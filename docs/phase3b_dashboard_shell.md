# Phase 3B / 3B.5: Replay-First Dashboard Shell

## Purpose

Phase 3B turns the Phase 3A race-state groundwork into the first real product shell inside `app/streamlit_app.py`.

This phase is about product structure and UI flow, not new strategy modeling.

It keeps the current deterministic strategy engine intact while presenting it in the shape of the future race-control experience.

The latest Phase 3B.5 refinement pass moves the main race-control surface into a dedicated Streamlit Components v2 board while keeping Streamlit as the host application. The primary screen is now tighter, more deliberately aligned, and less constrained by stacked native widgets. It still does not copy any official broadcast or game interface; it uses an original dark race-control style for the replay dashboard.

The current polish pass keeps that architecture intact while tightening the native Analysis layout, removing lap-suffix clutter from raw metrics, and adding a safe local-first asset path for optional logos/photos plus team-color selected-state accents.

## What Phase 3B Adds

Phase 3B adds four practical product layers:

### 1. Replay-first dashboard layout

The canonical Streamlit app now has a race-control shape instead of a single recommendation page.

It includes:

- a custom race-control board for the circuit, leaderboard, and selected-driver drawer
- a Race Control / Analysis view switch so deep-dive content does not sit in the primary tactical flow
- a lighter left control sidebar that stays subordinate to the race-control surface
- compact collapsed snapshot-limit notes instead of prominent assumptions text

The first screen is intended to feel useful without treating the app like a report. The circuit, timing/order, selected-driver state, stint strip, and current pit call are the primary surface.

The custom board lives under `components/race_control_board/` and is mounted only inside the Race Control mode. Analysis remains native Streamlit.

Optional visual assets live under `assets/` and are wired through `assets/asset_manifest.json`. Missing assets degrade cleanly to the text-only presentation.

### 2. Historical replay snapshot flow

The app now uses canonical Phase 3A race-state objects to render at least one real historical snapshot cleanly.

The main source mode is:

- Miami 2024 replay checkpoints from the canonical Pre-3 backtest artifact

Those checkpoints provide:

- lap-level driver order
- team identity
- current compound
- tyre age
- stint history
- replay checkpoint notes

The app overlays the current deterministic recommendation engine on top of that replay state.

That is an intentional product-shell choice:

- replay state is real
- recommendation overlay is current-model output
- the result is useful for shell-building without pretending the backend is already a full live race simulator

### 3. Driver detail experience

Selecting a driver now updates a compact tactical drawer-style panel inside the custom board. The current pit call sits high in the drawer, followed by the stint strip and support/risk framing. It prioritizes:

- driver code / name
- team
- start position
- current position
- current compound
- tyre age
- laps remaining
- stint history as a visual strategy strip
- current recommendation as a race-call card
- support / confidence / risk framing

Notes, placeholder caveats, and deeper model context remain accessible, but they are not the first thing the user sees.

When present, a local driver photo and team logo can appear in the selected-car header. When absent, the layout stays stable and text-first.

### 4. Secondary analysis mode

Advanced analysis remains available, but it is intentionally separated from the primary race-control view.

The Analysis view is separate from Race Control and stays native Streamlit. It uses compact summary cards plus tabs for:

- alternative strategies
- sensitivity context
- model/data context
- timing curves and lap-time grids

This keeps the tactical screen focused while preserving the existing diagnostics for users who want the deeper model readout.

The current polish pass also normalizes card sizing, wraps long values safely, and avoids overlapping/clipped summary content in Analysis mode.

The latest stabilization pass separates the Analysis selected-car summary from the HTML-heavy fallback/component presentation helpers. Analysis now renders from structured Python data only, which removes the risk of raw wrapper markup appearing as visible text in the native Streamlit view.

The final small Phase 3B polish pass keeps that native summary path intact while adding restrained native accents back into Analysis and separating team identity from tyre identity more clearly: circuit markers now use team color, while timing-table compound pills remain compound-colored.

### 5. Cleaner recommendation and stint presentation

The recommendation surface now translates engine output into product language:

- clear pit action
- target tyre
- strategy type
- compact model-time reference
- tight-margin note when the timing trace provides one
- risk/support details tucked behind expanders

The stint history now reads as a compact strategy strip rather than a text report.

### 6. Honest placeholder handling

Phase 3B does not fake missing data.

The shell explicitly treats these as placeholder or future integrations:

- exact interval gaps
- live telemetry coordinates
- tyre inventory
- full live race-control feed behavior

Driver photos and team logos are now optional local assets rather than built-in placeholders. The app does not auto-download official assets.

## Why A Custom Component Was Added

The earlier pure-Streamlit shell already had the right product structure, but the remaining gaps were mostly:

- alignment precision
- tighter selection behavior
- less stacked-widget appearance
- more integrated tactical composition

That made the race-control surface a good candidate for a single custom component, while leaving the rest of the app in Streamlit.

The current implementation uses `st.components.v2.component(...)` with local HTML/CSS/JavaScript assets in the repo. It is intentionally lightweight and does not introduce a separate frontend app or move Analysis mode out of Streamlit.

## What Is Real Versus Placeholder

### Real now

- historical replay race/session identity
- driver and team identity from canonical raw data
- current position
- current compound
- tyre age
- stint history reconstructed from lap-level data
- deterministic recommendation overlays from the existing strategy engine
- Phase 2C stability and timing-trace context for the selected driver
- support and risk metadata from the existing validation / defensibility stack

### Placeholder or intentionally limited

- circuit view coordinates are schematic order-based markers, not telemetry XY positions
- competitor context only exposes nearby identities where derivable; reliable gap timing is still missing
- photos and richer profile metadata are omitted because there is no canonical asset table yet
- live-feed behavior is not implemented
- advanced analysis is still deterministic and single-car, even though it is presented more cleanly

## What Phase 3B Does Not Add

Phase 3B does **not** add:

- competitor-aware strategy logic
- undercut / overcut modeling
- traffic-aware recommendations
- SC / VSC strategy behavior
- weather logic
- opponent-response logic
- Monte Carlo or probabilistic branching

Those remain future work.

## Python-to-Component Contract

The custom board is fed from a stable Python payload built in `app/streamlit_app.py`.

The contract currently has four top-level sections:

- `meta`
- `circuit`
- `timing`
- `selected_driver`

It includes replay/source metadata, schematic circuit path points, ordered markers, compact timing rows, and a selected-driver tactical payload with identity, tyre state, call summary, support/confidence/risk framing, nearby context, and stint-strip items.

The component currently returns:

- `selected_driver`

That keeps the selection loop inside the host Streamlit app simple and gives Phase 3C a stable board contract to extend later.

## How It Uses Phase 3A

Phase 3A introduced the canonical checkpoint model in `src/simulation/race_state.py`.

Phase 3B / 3B.5 now uses that model directly for:

- historical replay field snapshots
- selected-driver detail rendering
- timing / order panel rows
- recommendation payload display

That keeps the dashboard surface aligned to one canonical object shape instead of inventing a separate UI-only payload.

## Fallback Behavior

If the custom board is unavailable in the current environment or throws during render, the app falls back to the existing native Streamlit race-control shell instead of crashing.

That fallback preserves:

- circuit panel
- timing panel
- selected-driver drawer

The host app remains canonical either way.

## Relationship To Later Phases

### Phase 3C

Expected next additions:

- gap-aware context
- nearby-car framing
- undercut / overcut explanation
- richer timing-panel fields once reliable gap inputs exist

Phase 3B prepares for that by reserving nearby-car fields and by building a timing/order shell where that context can eventually live.

### Phase 3D

Expected later additions:

- SC / VSC handling
- weather
- traffic
- richer race realism
- stochastic branching

Phase 3B intentionally stops far before that line.

## Current Honest Read

After Phase 3B.5, the repo now has:

- the data foundation for dashboard checkpoints
- a replay-first product shell
- a custom race-control board embedded inside the host app
- a selected-driver workflow

But it still does **not** have:

- a live race-control backend
- reliable gap timing
- real track coordinates
- competitor-aware strategy recommendations

It now does have:

- optional local-first asset wiring for logos/photos
- selected-state team-color accents
- cleaner Analysis-mode layout behavior

That is the correct and honest scope boundary for this phase.
