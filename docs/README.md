# F1 Strategy Lab Documentation

## System Overview

**Current (Phase 0):** Lap-level race ingestion, linear compound-specific tyre degradation, empirical pit-loss estimation, and deterministic pit-window optimization. Produces PIT vs STAY OUT recommendations backed by exhaustive search.

**Planned (Phase 1):** Fuel-corrected pace modeling, representative-lap filtering, piecewise degradation with cliff detection, used-set history, and undercut evaluation to create a trustworthy foundation for strategy search.

## Design Documents

- **[phase1_spec.md](phase1_spec.md)** — Phase 1 specification for trustworthy lap-time prediction
  - Fuel-correction approach and validation
  - Representative-lap filtering protocol
  - Piecewise degradation and cliff detection
  - Used-set feature modeling
  - Undercut evaluation framework
  - Success and failure criteria

## Project README

See [../README.md](../README.md) for project overview, current capabilities, data sources, and results.
