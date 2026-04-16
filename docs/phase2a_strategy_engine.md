# Phase 2A: Strategy Engine Upgrade

## Status: ✅ COMPLETE

Phase 2A extends the Phase 1 pit-window optimizer with **automatic strategy search and recommendation**.

---

## What Phase 2A Adds

### Before Phase 2A
**User workflow:**
1. Choose current tyre (e.g., MEDIUM)
2. Choose target tyre (e.g., HARD) ← Manual choice
3. Get: optimal pit lap for that specific scenario
4. No search across compounds
5. No one-stop vs two-stop comparison

**Limitation:** The engine did not automatically recommend which tyre to switch to. User had to know the strategy beforehand or manually try all combinations.

### After Phase 2A
**New workflow:**
1. Input current state: compound, tyre life, laps remaining
2. Engine automatically searches:
   - ✅ ONE-STOP strategies: MEDIUM→SOFT, MEDIUM→HARD, etc.
   - ✅ TWO-STOP strategies: MEDIUM→SOFT→HARD, MEDIUM→HARD→SOFT, etc.
3. Ranks all strategies by total race time
4. Returns:
   - ✅ Best strategy recommendation
   - ✅ Next tyre to pit to
   - ✅ Pit lap(s)
   - ✅ Feasibility assessment
   - ✅ Top 5 alternatives with time deltas
   - ✅ Explanation for each strategy

**Improvement:** Strategy selection is now data-driven and automatic, not manual.

---

## Key Components

### 1. Strategy Plan Abstraction (`StrategyPlan` dataclass)

```python
@dataclass
class StrategyPlan:
    strategy_type: str           # "one-stop" or "two-stop"
    current_compound: str        # Starting tyre (e.g., "MEDIUM")
    current_tyre_life: int       # Starting tyre life
    next_compound: str           # First pit switch to
    final_compound: Optional[str] # Second pit switch (two-stop only)
    pit_lap: int                 # First pit lap
    second_pit_lap: Optional[int] # Second pit lap (two-stop only)
    total_race_time: float       # Estimated total time in seconds
    feasible: bool               # Is this strategy realistic?
    feasibility_reason: str      # Explanation
    one_stop_estimate: Optional[float] # For two-stop, one-stop time for comparison
    explanation: str             # Friendly strategy description
    model_info: Dict             # Metadata (model types, samples, etc.)
```

**Design goals:**
- Clean representation of a complete strategy
- Includes both optimization result and feasibility info
- Self-documenting (no hidden assumptions)
- Reusable across Streamlit, demo, and future tools

---

### 2. Feasibility Heuristic

**Question:** Can a tyre realistically complete a stint?

**Heuristic:** Conservative usable-tyre-life ranges per compound:

| Compound | Usable Tyre Life |
|----------|------------------|
| SOFT     | ~20 laps         |
| MEDIUM   | ~35 laps         |
| HARD     | ~50 laps         |

**Logic:**
```
For each stint:
  if stint_length > max_viable_for_compound:
    → INFEASIBLE (tyre will degrade too much)
  elif no degradation model for compound:
    → INFEASIBLE (no data to predict)
  else:
    → FEASIBLE
```

**Important:** This is a **reviewable heuristic**, not a black-box model.
- Ranges are conservative (errs on side of caution)
- Based on observed data patterns, not race-engineer expertise
- Can be easily adjusted if real data teaches us otherwise
- Explicitly reported to user

---

### 3. Search Space Design

**Constraint: Avoid combinatorial explosion**

#### One-Stop Search
- For each candidate next_compound ∈ {SOFT, MEDIUM, HARD}:
  - Skip if same as current (no pit)
  - Evaluate: current → next → finish
  - Rank by total time

**Result:** Up to 2 one-stop options (3 targets - 1 current)

#### Two-Stop Search
- For each (next_compound, final_compound) pair:
  - Skip if next == current (would be one-stop)
  - Skip if next == final (use one-stop instead)
  - Enforce minimum stint length (3 laps default) to prevent zigzagging
  - Simple pit-lap heuristic: first pit ~1/3 of remaining, second pit ~2/3

**Result:** Up to 6 two-stop options (9 pairs - 3 same-as-current - 3 consecutive same)

**Total: ~8 strategies evaluated per scenario (tractable)**

---

### 4. Strategy Ranking

**Primary sort:** Total race time (ascending)

**Secondary:** Feasibility priority option (feasible first within time-sorted list)

**Output:** "Top N" ranked list (typically top 5 shown to user)

---

## User-Facing Features

### Streamlit App

**Main section: "Recommended Strategy (Phase 2A)"**
```
Next Tyre: HARD
Pit Lap: 8
Strategy Type: ONE-STOP
Feasible: ✅ Yes
Est. Total Time: 2325.47 s

Explanation: Pit to HARD at lap 8, finish on HARD
Feasibility: HARD is viable for 17 laps (estimated max ~50)
```

**Alternative strategies table:**
```
Rank | Type      | Next Tyre | Pit(s)      | Final Tyre | Total Time | Feasible | vs Best
-----|-----------|-----------|-------------|-----------|------------|----------|--------
  1  | ONE-STOP  | HARD      | L8          | HARD      | 2325.47 s  | ✅       | BEST
  2  | ONE-STOP  | MEDIUM    | L7          | MEDIUM    | 2330.52 s  | ✅       | +5.05s
  3  | TWO-STOP  | SOFT      | L5 + L14    | HARD      | 2338.91 s  | ✅       | +13.44s
  4  | ONE-STOP  | SOFT      | L4          | SOFT      | 2350.22 s  | ⚠️       | +24.75s
  5  | TWO-STOP  | MEDIUM    | L7 + L16    | HARD      | 2365.18 s  | ⚠️       | +39.71s
```

**Advanced mode (optional):** Manual compound comparison
- Choose any target compound
- See pit-lap curve (original Phase 1 visualization)
- For power users who want to understand trade-offs

---

### Demo Output

```
  [Phase 2A - Automatic Strategy Search]

  Top Recommendation:
    Type: ONE-STOP
    Next Tyre: HARD
    Pit Lap: 8
    Total Time: 2325.47 s
    Feasible: ✓ Yes
    Rationale: Pit to HARD at lap 8, finish on HARD

  Top 5 Strategy Options (ranked by time):
    1. ✓ ONE-STOP  | MEDIUM→HARD @ L8 | Time: 2325.47s (BEST)
    2. ✓ ONE-STOP  | MEDIUM→MEDIUM @ L12 | Time: 2330.52s (+5.05s)
    3. ✓ TWO-STOP  | MEDIUM→SOFT→HARD @ L5,L14 | Time: 2338.91s (+13.44s)
    4. ⚠ ONE-STOP  | MEDIUM→SOFT @ L4 | Time: 2350.22s (+24.75s)
    5. ⚠ TWO-STOP  | MEDIUM→MEDIUM→HARD @ L7,L16 | Time: 2365.18s (+39.71s)
```

---

## Implementation

### New Module: `src/simulation/strategy_engine.py`

**Public API:**

```python
# Core entry point
best_plan, all_ranked_plans = recommend_best_strategy(
    degradation_models=deg_result,  # Phase 1 unified result
    pit_loss_value=pit_loss_value,  # Estimated pit time
    current_compound="MEDIUM",
    current_tyre_life=5,
    laps_remaining=25,
    candidate_compounds=["SOFT", "MEDIUM", "HARD"],
    include_two_stop=True,
)

# Lower-level builders (if you want to customize)
one_stop_plans = evaluate_one_stop_strategies(...)
two_stop_plans = evaluate_two_stop_strategies(...)
ranked_plans = rank_strategy_plans(one_stop_plans, two_stop_plans)

# Feasibility checker
feasible, reason = estimate_stint_feasibility(
    degradation_models=deg_result,
    compound="HARD",
    tyre_life=1,
    laps_remaining=17,
    pit_loss_value=pit_loss_value,
)
```

**Design philosophy:**
- Layers of abstraction: recommend_best_strategy() → evaluate_{one,two}_stop_strategies() → estimate_stint_feasibility()
- Each layer is testable and reusable
- Reuses existing Phase 1 pit-window optimizer (no reinvention)
- Feasibility is explicit and reviewable (not hidden in black-box)

---

## Limitations & Caveats

### What Phase 2A Does NOT Do

1. **Real race modeling:** No traffic, DRS, safety cars, undercut timing, traffic vs pit-lane time
2. **Driver capability:** Assumes consistent lap times (no driver errors, no heroics)
3. **Real-time adaptation:** Plans are static; doesn't re-optimize if circumstances change
4. **Pit crew efficiency:** Uses median empirical pit time; actual may vary
5. **Weather & track changes:** Assumes constant conditions
6. **Tyre temperature:** Assumes tyres perform to rating immediately after fit
7. **Fuel consumption:** Phase 1B corrects for fuel in lap times, but doesn't track remaining fuel
8. **Multi-lap pit stops:** Simplified model doesn't capture rare multi-stop outliers

### Design Trade-offs

**Search space simplification:**
- Two-stop uses heuristic pit laps (1/3 and 2/3) rather than exhaustive search
- Rationale: Exhaustive 3-stint search would be ~100+ combinations; heuristic gives ~6 options (tractable and good-enough)
- Could be upgraded later to more thorough search if needed

**Feasibility heuristic:**
- Conservative ranges (may mark some feasible strategies as "check"）
- Rationale: Better to warn user than falsely claim feasibility
- Ranges tunable based on new data

---

## Usage

### In Streamlit App

**Default behavior (what user sees):**
```python
best_plan, all_ranked_plans = recommend_best_strategy(
    degradation_models=deg_result,
    pit_loss_value=pit_loss_value,
    current_compound=compound,
    current_tyre_life=current_tyre_life,
    laps_remaining=laps_remaining,
    candidate_compounds=available_compounds,
    include_two_stop=True,  ← Optional
)

# Display best_plan prominently
# Show top 5 from all_ranked_plans in table
```

**Advanced mode (opt-in):**
```python
# User chooses a specific target compound
target_compound = st.selectbox("Compare to:", available_compounds)

# Fall back to Phase 1 pit-lap visualization
strategy_df = optimize_pit_window(..., post_pit_compound=target_compound)
plot_pit_lap_curve(strategy_df)
```

### In Demo

```python
# Replace manual strategy selection with automatic search
best_plan, all_ranked_plans = recommend_best_strategy(...)

# Print best plan
print(f"Recommendation: {best_plan.next_compound} @ L{best_plan.pit_lap}")

# Show top 5
for i, plan in enumerate(all_ranked_plans[:5]):
    print(f"{i}. {plan.strategy_type} | Time: {plan.total_race_time:.2f}s")
```

### Standalone Usage

```python
from src.simulation.strategy_engine import recommend_best_strategy

# Works with any Phase 1 degradation result
best_plan, ranked_plans = recommend_best_strategy(
    degradation_models=my_deg_result,
    pit_loss_value=0.30,
    current_compound="MEDIUM",
    current_tyre_life=5,
    laps_remaining=25,
)

print(f"Best strategy: Pit to {best_plan.next_compound} at lap {best_plan.pit_lap}")
print(f"Estimated time: {best_plan.total_race_time:.2f}s")
print(f"Feasible: {best_plan.feasible_reason}")
```

---

## Future Enhancements

### Phase 2B (Potential)
- **Pit-lap optimization for two-stop:** Instead of heuristic, search for optimal pit laps for all three stints simultaneously
- **Undercut evaluation:** Consider counterattacking via strategic pitstop timing
- **Margin calculations:** How much time gain/loss from being "off-line" during pit window?
- **Explicit lap-time variation:** Model pit-lane speed, tyre warm-up lag

### Phase 2C (Potential)
- **Real-time re-planning:** Live lap data allows recalculation as race progresses
- **Driver input:** Button to "execute pit now" vs "wait" with updated recommendation
- **Probability modeling:** Feasibility as likelihood rather than binary

### Phase 3 (Potential)
- **Multi-driver strategies:** What if my teammate pits first? Cost/benefit of covering off?
- **Safety car adaptation:** Update strategies on yellow flags
- **Opponent modeling:** Position-dependent decisions (e.g., "undercut if close behind")

---

## Testing & Validation

**Implemented in Phase 2A:**
- ✅ Strategy plan creation and validation
- ✅ Feasibility assessment logic (explicit, reviewable)
- ✅ One-stop and two-stop ranking
- ✅ Streamlit integration (no crashes)
- ✅ Demo integration (clear output)

**Recommended future:**
- Unit tests for strategy builders
- Integration tests with Phase 1 data
- Historical race backtesting (vs real outcomes)

---

## References

- Phase 1 Integration: `docs/PHASE_1_INTEGRATION.md`
- Degradation Modeling: `docs/phase1c_degradation_modeling.md`
- Fuel Correction: `docs/phase1b_fuel_correction.md`
