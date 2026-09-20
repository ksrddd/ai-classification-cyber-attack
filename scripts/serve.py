"""Run the dashboard: the FastAPI backend and the Next.js frontend together.

The dashboard is two processes and neither is useful alone. The Next app reads
every number through the API, so starting it by itself produces a page that
renders its "could not reach the API" state -- which looks like a broken build
rather than a missing server. Starting them by hand means two terminals, two
Ctrl-C's, and remembering that stopping one leaves the other running.

What this adds over ``npm run dev`` in one window and ``uvicorn`` in another:

* **One lifetime.** Ctrl-C stops both. If either exits on its own the other is
  stopped too, so a crashed API never leaves a frontend behind still serving
  stale pages from its cache.
* **The browser opens when the API is actually answering**, not when the
  process has been spawned. Next compiles the first page on demand; opening
  too early shows the error state and the reader reloads by hand.
* **Interleaved, prefixed logs**, because the questions worth asking are
  cross-process ones -- which request the frontend made, and what the API said
  about it.
* **Windows process trees.** ``npm run dev`` spawns node as a child, and
  terminating npm on Windows orphans it; the port then stays busy and the next
  run fails with EADDRINUSE. Children are killed as a tree here.

Run::

    python scripts/serve.py                  # both, open a browser
    python scripts/serve.py --no-browser
    python scripts/serve.py --api-only       # e.g. to point the Streamlit app at it
    python scripts/serve.py --web-port 3001
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_API_PORT = 8000
DEFAULT_WEB_PORT = 3000

#: Origins ``api/main.py`` hands to the CORS middleware. A frontend on any
#: other port loads and then fails every fetch, which is a confusing way to
#: find out, so the mismatch is reported up front instead.
API_ALLOWED_WEB_PORTS = (3000, 3001)

#: How long to wait for the API to answer before opening a browser anyway.
#: Generous because the first request imports pandas and walks ``results/``.
API_READY_TIMEOUT_S = 60.0

ON_WINDOWS = os.name == "nt"


def api_command(port: int) -> list[str]:
    """Uvicorn through ``sys.executable``, so the active venv is the one used.

    Calling the ``uvicorn`` console script instead would resolve against PATH
    and could start the server in a different interpreter than the one this
    script is running in -- with a different set of installed packages.
    """
    return [sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(port)]


def web_command(port: int) -> list[str]:
    npm = "npm.cmd" if ON_WINDOWS else "npm"
    return [npm, "--prefix", "web", "run", "dev", "--", "--port", str(port)]


def spawn(command: list[str], env: dict[str, str]) -> subprocess.Popen[str]:
    """Start a child with its output captured and its own process group.

    The group matters for shutdown: on Windows a new process group lets the
    whole tree be killed together, and on POSIX it keeps the child from
    receiving the Ctrl-C the terminal sends to this script -- shutdown is
    driven from one place below rather than racing with the signal.
    """
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if ON_WINDOWS else 0
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        start_new_session=not ON_WINDOWS,
    )


def use_replacement_characters() -> None:
    """Never let an unprintable byte take the log with it.

    Next.js prints its logo as U+25B2, which a Windows console running cp1252
    cannot encode. Without this the print raises UnicodeEncodeError inside the
    forwarding thread, that thread dies, and the frontend simply stops
    logging -- while still running. Losing the glyph is the right trade;
    losing the log is not.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(OSError, ValueError):
                reconfigure(errors="replace")


def pump(name: str, process: subprocess.Popen[str], stop: threading.Event) -> None:
    """Forward one child's output, tagged, until it closes its pipe."""
    assert process.stdout is not None
    for line in process.stdout:
        if stop.is_set():
            break
        try:
            print(f"[{name}] {line.rstrip()}", flush=True)
        except UnicodeEncodeError:
            # A stream that could not be reconfigured above. Still better to
            # drop one line than to lose every line after it.
            print(f"[{name}] <line with unprintable characters>", flush=True)


def terminate(name: str, process: subprocess.Popen[str]) -> None:
    """Stop a child and everything it started.

    ``npm run dev`` is a launcher: the server that holds the port is a node
    process underneath it. Killing only the parent leaves the port bound, and
    the next run dies with EADDRINUSE pointing at a process the user cannot
    find. ``taskkill /T`` walks the tree on Windows; on POSIX the process
    group does the same job.
    """
    if process.poll() is not None:
        return
    print(f"[serve] stopping {name}", flush=True)
    try:
        if ON_WINDOWS:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
            )
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[serve] could not stop {name} cleanly: {exc}", flush=True)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def wait_for_api(port: int, deadline: float, process: subprocess.Popen[str] | None) -> bool:
    """Poll ``/api/bundles`` until it answers, exits, or the deadline passes.

    Any HTTP response counts as ready, including an error one: the server is
    up and the page can render whatever it says. Only a refused connection
    means "not yet".

    The process is watched alongside the socket because the most common way
    for this wait to fail is the port already being in use -- uvicorn prints
    the bind error and exits within a second, and polling a socket nobody is
    going to open would then sit out the whole timeout before reporting a
    failure that was visible immediately.
    """
    url = f"http://127.0.0.1:{port}/api/bundles"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2):  # noqa: S310 -- fixed localhost
                return True
        except urllib.error.HTTPError:
            return True
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.4)
    return False


def preflight(args: argparse.Namespace) -> int:
    """Refuse to start on the two problems that produce confusing symptoms."""
    if not args.api_only:
        if not (PROJECT_ROOT / "web" / "node_modules").is_dir():
            print(
                "web/node_modules is missing -- the frontend cannot start.\n"
                "Install it once with:  npm --prefix web install",
                file=sys.stderr,
            )
            return 1
        if args.web_port not in API_ALLOWED_WEB_PORTS:
            print(
                f"The API only allows browser origins on ports "
                f"{', '.join(str(p) for p in API_ALLOWED_WEB_PORTS)}, so a frontend on "
                f"{args.web_port} would load and then fail every request. Add the "
                f"origin to allow_origins in api/main.py first.",
                file=sys.stderr,
            )
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--api-port", type=int, default=DEFAULT_API_PORT)
    ap.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser")
    ap.add_argument("--api-only", action="store_true", help="start only the backend")
    ap.add_argument("--web-only", action="store_true", help="start only the frontend")
    args = ap.parse_args(argv)

    use_replacement_characters()

    if args.api_only and args.web_only:
        ap.error("--api-only and --web-only ask for opposite things")

    failed = preflight(args)
    if failed:
        return failed

    env = os.environ.copy()
    # The frontend reads this at build time for server components and at run
    # time in the browser, so a non-default API port has to be handed down
    # rather than assumed.
    env["NEXT_PUBLIC_API_URL"] = f"http://localhost:{args.api_port}"

    children: list[tuple[str, subprocess.Popen[str]]] = []
    stop = threading.Event()

    try:
        if not args.web_only:
            print(f"[serve] api  -> http://localhost:{args.api_port}", flush=True)
            children.append(("api", spawn(api_command(args.api_port), env)))
        if not args.api_only:
            print(f"[serve] web  -> http://localhost:{args.web_port}", flush=True)
            children.append(("web", spawn(web_command(args.web_port), env)))

        pumps = [
            threading.Thread(target=pump, args=(name, proc, stop), daemon=True)
            for name, proc in children
        ]
        for thread in pumps:
            thread.start()

        if not args.web_only:
            api = next((proc for name, proc in children if name == "api"), None)
            if not wait_for_api(args.api_port, time.monotonic() + API_READY_TIMEOUT_S, api):
                print(
                    (
                        "[serve] the API is not answering -- see its log above"
                        if api is not None and api.poll() is not None
                        else f"[serve] the API did not answer within "
                        f"{API_READY_TIMEOUT_S:.0f}s -- leaving both running so its "
                        "log above can be read"
                    ),
                    flush=True,
                )

        if not args.no_browser and not args.api_only:
            url = f"http://localhost:{args.web_port}"
            print(f"[serve] opening {url}", flush=True)
            webbrowser.open(url)

        print("[serve] Ctrl-C to stop both", flush=True)
        # Poll rather than wait() on one child: whichever exits first should
        # take the other down with it, and waiting on a single process would
        # miss the case where the other is the one that died.
        while all(proc.poll() is None for _, proc in children):
            time.sleep(0.5)

        for name, proc in children:
            if proc.poll() is not None:
                print(f"[serve] {name} exited with code {proc.returncode}", flush=True)
        return next((proc.returncode for _, proc in children if proc.poll() not in (None, 0)), 0)

    except KeyboardInterrupt:
        print("\n[serve] interrupted", flush=True)
        return 130
    finally:
        stop.set()
        for name, proc in reversed(children):
            terminate(name, proc)


if __name__ == "__main__":
    raise SystemExit(main())
