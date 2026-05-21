# The Epistemic Arcade — frontend

A vanilla HTML / CSS / JS scrollytelling visualization of the Alpha (Internalist) vs Beta (Extended Mind) Tetris agents.

## Run it locally

The page fetches `../data/telemetry_{alpha,beta}.json` via relative URLs, so it must be served with the **project root** as the document root.

```bash
# From the project root:
python3 -m http.server 8000
```

Then open <http://localhost:8000/web/>.

That's it — no build step, no npm. Chart.js is loaded from a CDN.

## Files

- `index.html` — six-step scrollytelling layout (prose column + sticky visual column). Triggers: intro → Otto/demo → 10k → 250k → 1M → chart.
- `styles.css` — "Refined Monochrome CRT" design system (per `essay_spec.md` §2).
- `app.js` — data loading, JSON decoder, canvas renderer, IntersectionObserver step controller, Chart.js setup.

## Data dependencies

- `../data/telemetry_alpha.json` — produced by `uv run python -m epistemic_arcade.agents.evaluate --agent alpha`.
- `../data/telemetry_beta.json` — same, for Beta.

If either file is missing, the loading screen will replace itself with a hint pointing back here.

## Hotkeys

None — the experience is driven entirely by scroll position.

## Browser support

Tested on current Chrome and Firefox. Uses `fetch`, ES2020+ syntax, `IntersectionObserver`, and `requestAnimationFrame`. No polyfills. Mobile Safari should work; the layout collapses to a single column under 960 px.
