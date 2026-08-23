"""Convert the dashboard's CSS custom properties into Tokens Studio JSON.

web/src/app/globals.css is the single source of truth for colour in the
dashboard: :root carries the dark palette and [data-theme="light"] carries the
light one. Tokens Studio models exactly that shape as two token sets that a
Figma theme switches between, so the mapping is one-to-one and this script can
be re-run whenever globals.css changes.

    python scripts/css_tokens_to_figma.py

Writes docs/design/figma-export/tokens.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "web" / "src" / "app" / "globals.css"
OUT = ROOT / "docs" / "design" / "figma-export" / "tokens.json"

# --border-focus is a channel triple used inside rgb(); everything else in the
# --border-* family carries baked-in alpha. Group names are chosen so the Figma
# variable tree reads the way the CSS does.
GROUPS = {
    "color": "color",
    "chart": "chart",
    "sev": "severity",
    "border": "border",
    "bg": "bg",
    "text": "text",
}

COMMENT = re.compile(r"/\*.*?\*/", re.S)
DECL = re.compile(r"--([a-z0-9-]+)\s*:\s*([^;]+);")
CHANNELS = re.compile(r"^(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})$")
RGBA = re.compile(r"^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)$")
INDIRECT = re.compile(r"^rgba?\(\s*var\(\s*--([a-z0-9-]+)\s*\)\s*\)$")


def block(css: str, selector: str) -> str:
    """Return the body of the first `selector { ... }` rule, brace-matched."""
    start = css.index(selector)
    open_brace = css.index("{", start)
    depth, i = 0, open_brace
    while i < len(css):
        if css[i] == "{":
            depth += 1
        elif css[i] == "}":
            depth -= 1
            if depth == 0:
                return css[open_brace + 1 : i]
        i += 1
    raise ValueError(f"unbalanced braces after {selector}")


def to_hex(value: str) -> str | None:
    """Normalise a CSS colour value to #RRGGBB or #RRGGBBAA, else None."""
    value = value.strip()

    if value.startswith("#"):
        return value.upper()

    m = CHANNELS.match(value)
    if m:
        return "#{:02X}{:02X}{:02X}".format(*(int(c) for c in m.groups()))

    m = RGBA.match(value)
    if m:
        r, g, b, a = m.groups()
        hexv = "#{:02X}{:02X}{:02X}".format(int(float(r)), int(float(g)), int(float(b)))
        if a is not None and float(a) < 1:
            hexv += "{:02X}".format(round(float(a) * 255))
        return hexv

    return None


def parse(body: str) -> dict:
    """Build a nested {group: {name: token}} tree from one CSS rule body."""
    decls = dict(DECL.findall(COMMENT.sub("", body)))

    tokens: dict[str, dict] = {}
    for name, raw in decls.items():
        # Dark defines --accent as rgb(var(--color-brand-cyan)) while light
        # spells out a hex. Resolve one hop so both sets carry the same keys —
        # a Figma variable missing from one mode is a broken variable.
        hop = INDIRECT.match(raw.strip())
        if hop:
            raw = decls.get(hop.group(1), raw)

        hexv = to_hex(raw)
        if hexv is None:
            continue  # non-colour values (gradients, unresolved indirection)

        head, _, rest = name.partition("-")
        group = GROUPS.get(head)
        if group is None:
            group, leaf = "semantic", name
        else:
            leaf = rest or head

        tokens.setdefault(group, {})[leaf] = {"value": hexv, "type": "color"}
    return tokens


def main() -> None:
    css = CSS.read_text(encoding="utf-8")
    sets = {
        "dark": parse(block(css, ":root")),
        "light": parse(block(css, '[data-theme="light"]')),
    }

    doc = {
        **sets,
        "$themes": [
            {"id": name, "name": name.capitalize(), "selectedTokenSets": {name: "enabled"}}
            for name in sets
        ],
        "$metadata": {"tokenSetOrder": list(sets)},
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    for name, tree in sets.items():
        total = sum(len(v) for v in tree.values())
        print(f"{name}: {total} tokens across {len(tree)} groups -> {sorted(tree)}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
