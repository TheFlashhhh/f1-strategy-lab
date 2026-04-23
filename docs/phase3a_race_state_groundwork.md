# Phase 3A: Race-State And Dashboard Groundwork

## Purpose

Phase 3A starts building the **data and checkpoint foundation** for the future F1 Strategy Lab product without changing the current recommendation engine into a richer race simulator.

This is groundwork only.

It does **not** add:

- competitor-aware recommendation logic
- traffic or overtaking logic
- Safety Car / VSC strategy behavior
- weather handling
- Monte Carlo branching
- opponent modeling
- a full visual dashboard redesign

## Future Product Vision

The long-term product direction is a replay-first race-control dashboard that can eventually grow into a live race surface.

### Main race-control page

- top-down circuit map
- live or replay car positions on track
- timing / order panel beside the map
- driver selection from the order panel or track map

### Driver detail panel

When a user clicks a driver, the product should eventually show:

- driver photo
- name
- team
- start position
- current position
- current compound
- tyre age
- stint timeline / stint history
- recommendation: when to pit, and which tyre to pit to
- confidence / support / risk notes
- deeper strategy analysis in a secondary section

### Dynamic race-state concept

The final product needs a race-state model that can survive position re-ordering as cars pit, undercut, or overcut each other.

Phase 3A does **not** implement that dynamic logic yet.

Phase 3A only makes sure the canonical checkpoint schema is ready for that future behavior.

Phase 3B now uses this groundwork in the canonical Streamlit app, but this document stays focused on the Phase 3A foundation itself.

## Why Replay / Historical Support Comes First

Replay-first is the right build order because the repository already has:

- canonical historical lap-level data
- representative validation checkpoints
- held-out backtest checkpoints
- deterministic recommendation artifacts

That is enough to build inspectable replay checkpoints now.

The repository does **not** yet have:

- live timing ingestion
- reliable gap-to-car timing fields
- track-map coordinates
- explicit SC / VSC flags in the strategy stack
- weather inputs

So a live dashboard would be mostly shell with too many missing foundations. Replay support is the honest first step.

## What Phase 3A Adds

Phase 3A adds four main foundations:

### 1. Canonical race-state schema

`src/simulation/race_state.py` defines the canonical checkpoint object for future dashboard and replay work.

It includes typed structures for:

- race/session identity
- selected driver state
- stint history
- nearby competitor placeholders
- recommendation payload
- event-status placeholders

The module explicitly marks which fields are:

- active now
- derivable now with light work
- future placeholders

### 2. Extraction paths from existing repo artifacts

Phase 3A can now build canonical race-state objects from:

- the manual demo scenario shape
- Phase 2D representative validation scenarios
- Pre-3 held-out backtest checkpoints
- replay-style full-lap historical snapshots for the Phase 3B dashboard shell

This matters because future UI work should consume one canonical object shape instead of inventing separate ad hoc payloads for demo, validation, and replay cases.

### 3. Data availability audit

`scripts/run_phase3a_readiness.py` now writes a canonical availability artifact:

- `data/processed/phase3a_data_availability_summary.json`

This inventories what the current pipeline can already support for future dashboard fields, what is lightly derivable, and what still is not reliably available.

### 4. Roadmap clarity

The repository now explicitly distinguishes:

- the current deterministic single-car recommendation system
- the larger future dashboard product vision
- the phased build order from schema groundwork to richer race realism

## What Phase 3B / 3C / 3D Are Meant To Do

### Phase 3B: Dashboard shell and driver detail experience

This is now the direct follow-on phase that consumes the Phase 3A schema.

Expected focus:

- replay-first dashboard shell
- driver selection flow
- driver detail drawer / overlay
- presentation of canonical race-state objects
- schematic circuit placeholder until real coordinates exist

Still not the place for full rival-aware strategy logic.

### Phase 3C: Competitor-gap / undercut-overcut context

Expected focus:

- nearby-car context
- gap-aware framing
- undercut / overcut context in the recommendation surface

This is where the new schema starts paying off, because the checkpoint model already reserves nearby-car fields.

### Phase 3D: Richer race realism

Expected focus:

- Safety Car / VSC
- traffic
- weather
- stochastic branching

This is explicitly later because the deterministic core still needs to be treated honestly and because several key data feeds are not ready yet.

## What The Current Engine Still Does Not Support

The implemented system today is still:

- deterministic
- primarily Miami-anchored
- single-car in recommendation logic
- replay/offline rather than live

It still does **not** support:

- competitor-aware recommendation logic
- opponent reactions
- traffic-aware pit timing
- live track-map state
- gap-to-car timing logic
- SC / VSC strategy adaptation
- weather adaptation
- probabilistic outcome branching

## Readiness Interpretation

Phase 3A means:

- the repo is ready to standardize race-state checkpoints
- the repo is ready to support replay-first dashboard work
- the repo is **not** yet ready to claim rich race realism or competitor-aware strategy behavior

That is the correct scope boundary for this phase.
