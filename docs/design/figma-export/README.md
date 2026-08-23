# Figma export

Everything Figma needs to redesign the `web/` dashboard without guessing at a
colour, a label, or a number.

## What is here

| File | What it is | Feed it to |
|---|---|---|
| `make-brief.md` | The full design brief — IA, design language, component anatomy, all eight screens described panel by panel with real values, and what a redesign must preserve | Figma Make (paste as the prompt) |
| `screens/*.jpg` | Ten screenshots of the live dashboard, 1920px viewport — all 8 screens in dark, Overview and Model detail also in light | Figma Make (attach), or drop straight onto a design file canvas as reference |
| `tokens.json` | 120 colour variables in Tokens Studio format, two sets (`dark`, `light`) that one Figma theme switches between | Tokens Studio plugin → Import |
| `overview-dark.html` / `overview-light.html` | Self-contained DOM snapshots of the Overview screen with CSS inlined. **Local only — git-ignored**, since 46 KB of the 67 KB is the same compiled Tailwind reset in both files. Recapture from a running dev server if you want them | html.to.design plugin, or as a code reference |

## The two Figma links do different things

**Figma Make** — `figma.com/make/aEhobQi87EIZ8PoWjsAV4D/Project---AI-Cyber-Attack-Classification`

Prompt-driven. It generates working UI from a description plus reference images.
This is the right home for exploring redesigns of several screens at once.
Make files have no write API, so nothing in this folder can be loaded into one
automatically — the brief gets pasted in and the screenshots attached by hand.

**Figma Design** — `figma.com/design/vHqZzE0ZsqfaY7rBESpEtE/Dashboard`

A normal design file, and the dashboard is already built in it as real
auto-layout layers:

- **`CyberML Color`** — a variable collection of 60 colours with **Dark** and
  **Light** modes, generated from `tokens.json`. Every fill and stroke on the
  canvas is bound to it, so the whole file re-themes from the mode switcher.
- **`Redesign Web Dashboard (Desktop)`** — all 8 screens at 1440×1024, plus one
  `Overview — Light mode` frame as proof the modes resolve.
- **`Redesign Web Dashboard (IPad)`** — the same 8 screens at 834×1194: sidebar
  still visible, KPI grids at 2 columns, the 9-column comparison table clipped
  to a horizontal scroll.
- **`Redesign Web Dashboard (Mobile)`** — the same 8 at 390 wide, sidebar
  dropped for a hamburger, status bar trimmed, frames grown past 844 where the
  content is longer than the viewport.

The two tablet/phone pages follow the real breakpoints in the code
(`sm:640 / md:768 / lg:1024`), not a guess — the rail is `hidden md:flex`, the
KPI grids are `grid-cols-2 lg:grid-cols-4`, and the bundle context in the topbar
is `hidden sm:flex`.

Nav icons are drawn from primitives rather than imported from `lucide-react`;
they are named `icon/overview`, `icon/trophy` and so on, ready to be swapped.

## Using it with Figma Make

1. Open the Make link.
2. Paste the entire contents of `make-brief.md` into the prompt.
3. Attach the files in `screens/` — at minimum the screen you are redesigning,
   plus `01-overview-dark.jpg` and `01-overview-light.jpg` so Make sees both
   themes.
4. Add the actual ask on top, e.g. *"Redesign screen 5 (Comparison) so the cost
   of a false alarm is the first thing an analyst reads. Keep §5's
   non-negotiables."*

Do one screen per Make session. The brief carries the shared system, so screens
generated in separate sessions still agree with each other.

## Regenerating

Tokens, whenever `web/src/app/globals.css` changes:

```bash
python scripts/css_tokens_to_figma.py     # rewrites tokens.json
```

Screenshots — start both servers, then capture each route at a 1920px viewport
with `localStorage.theme` set to `dark` (or `light`):

```bash
.venv/Scripts/python.exe -m uvicorn api.main:app --port 8000
cd web && npm run dev
```

Routes, in the order the screenshots are numbered:
`/` · `/dataset` · `/eda` · `/performance` · `/compare` · `/shap` ·
`/predict` · `/contract`

The figures in `make-brief.md` are read off bundle `cicids2017_temporal_v1`. If
you recapture against a different bundle, update §4 and §6 to match — the brief
is only useful while its numbers are the real ones.
