# Phase 3B.5: Custom Race-Control Component

## Purpose

Phase 3B.5 introduces a single custom race-control surface inside the existing Streamlit app.

The goal is not to replace Streamlit. The goal is to use a purpose-built component where Streamlit layout primitives were starting to limit alignment, interaction feel, and tactical density.

## Scope

The custom component owns only the primary race-control board:

- circuit rendering
- timing / order rendering
- selected-car tactical drawer
- row / marker selection interaction

The host Streamlit app still owns:

- sidebar controls
- source and snapshot selection
- Race Control vs Analysis mode switch
- deep-dive analysis tabs, tables, charts, and notes
- graceful fallback behavior

It also owns the asset-manifest loading and local-file inlining so the component only receives browser-safe asset sources.

The native Analysis selected-car summary also stays outside the component and now renders from structured Python data only. That keeps the component boundary clean and avoids leaking component/fallback HTML helpers into the native Streamlit analysis path.

## Location

Component files live in:

- `components/race_control_board/__init__.py`
- `components/race_control_board/race_control_board.html`
- `components/race_control_board/race_control_board.css`
- `components/race_control_board/race_control_board.js`

## Implementation Style

The component uses Streamlit Components v2 via `st.components.v2.component(...)`.

It is intentionally lightweight:

- local HTML/CSS/JavaScript assets in-repo
- no parallel app entrypoint
- no external frontend deployment
- no change to the core strategy engine

The local environment for this phase did not provide a Node build chain, so the component is build-free rather than a compiled React/TypeScript package. The important product decision is the move to a custom component boundary, not a heavyweight frontend toolchain.

## Current Payload Contract

The Python host builds a stable payload with these top-level keys:

```python
race_control_payload = {
    "meta": {...},
    "circuit": {...},
    "timing": {...},
    "selected_driver": {...},
}
```

### `meta`

- source mode
- board title
- subtitle
- compact tags / chips

### `circuit`

- schematic mode label
- circuit name
- start/finish label
- `path_points`
- `markers`

### `timing`

- compact `rows`
- optional low-salience note for reduced-fidelity sources

Each row carries the driver key plus compact leaderboard fields:

- position
- driver code
- team / short team label
- optional team logo source
- team accent color
- compound
- tyre age
- short current-call snippet
- selection state

The colour split is intentional:

- circuit markers use team colour for car identity
- timing-row tyre pills use compound colour for tyre identity

### `selected_driver`

- driver identity
- team
- optional team logo source
- optional driver photo source
- selected-state team accent color variants
- start / current position
- laps left
- tyre age
- stint
- track-status label
- current call title
- current call subtitle
- timing-window note
- support / confidence / risk summary
- nearby ahead / behind
- compact stint-strip items
- low-salience source note

## Return Contract

The component currently returns one state field:

- `selected_driver`

That is enough for Phase 3B.5 because the main interaction loop is row / marker selection.

Future optional additions could include:

- hovered driver
- drawer open / closed state
- timing-row scroll position

## Fallback

If the custom component is unavailable or fails during render, `app/streamlit_app.py` falls back to the prior native Streamlit race-control shell.

This keeps the app usable during development and prevents the Race Control mode from becoming a single point of failure.

Missing assets are handled the same way: no logo means text-only team rendering, no driver photo means a text-only header, and no explicit team color falls back to the board cyan accent.

## Assets

Phase 3B.5 now supports a local-first asset manifest at `assets/asset_manifest.json`.

The expected shape is:

```json
{
  "teams": {
    "Ferrari": {
      "logo": "assets/team_logos/ferrari.png",
      "primary_color": "#DC0000"
    }
  },
  "drivers": {
    "LEC": {
      "photo": "assets/driver_photos/lec.png",
      "team": "Ferrari"
    }
  }
}
```

Rules:

- local repo paths are preferred
- manual external URLs are allowed if the user adds them
- missing files fail quietly
- no official F1 assets are auto-downloaded

Local files are converted to data URIs in Python before they are passed to the component, which keeps the frontend simple and avoids broken relative-path assumptions inside the Streamlit component runtime.

## What This Enables Next

This component boundary sets Phase 3C up cleanly for:

- richer timing-row context
- clearer nearby-car annotations
- gap-aware labels once canonical inputs exist
- undercut / overcut context inside the tactical drawer

It still does not add competitor-aware strategy logic, live telemetry, or richer realism by itself. Those remain future model and data phases.
