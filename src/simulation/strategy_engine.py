"""Phase 2A: Strategy Engine Upgrade - Automated strategy search and recommendation.

This module extends the Phase 1 pit-window optimizer to automatically recommend:
1. Which tyre to switch to next (SOFT/MEDIUM/HARD)
2. When to pit
3. Whether one-stop or two-stop is faster
4. A ranked list of top strategy options
5. Feasibility assessment for each strategy

**Design principles:**
- Reuse Phase 1 pit-window optimizer (optimize_pit_window)
- Add strategy plan abstraction for clean representation
- Implement explicit feasibility heuristics (not black-box)
- Rank by total race time, all times measured in same units
- Support one-stop and two-stop searches
- Keep search space tractable (no explosion of nonsense combinations)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.simulation.strategy import optimize_pit_window, predict_lap_time

logger = logging.getLogger(__name__)


@dataclass
class StrategyPlan:
    """Represents a complete pit strategy from current state to race end.
    
    Attributes:
        strategy_type: One of "one-stop" or "two-stop"
        current_compound: The compound the driver is currently on (e.g., "MEDIUM")
        current_tyre_life: Current tyre life in laps
        next_compound: The compound to switch to at first pit
        final_compound: The compound for final stint (only for two-stop; None for one-stop)
        pit_lap: Lap at which to execute first pit
        second_pit_lap: Lap for second pit (only for two-stop; None for one-stop)
        total_race_time: Estimated total race time in seconds
        feasible: Whether the strategy is realistically feasible
        feasibility_reason: Explanation of feasibility assessment
        one_stop_estimate: Total time if strategy were a one-stop (for comparison)
        explanation: Short rationale for the strategy
        model_info: Metadata about degradation models used (for transparency)
    """

    strategy_type: str  # "one-stop" or "two-stop"
    current_compound: str
    current_tyre_life: int
    next_compound: str
    final_compound: Optional[str]  # None for one-stop
    pit_lap: int  # First pit lap
    second_pit_lap: Optional[int]  # None for one-stop
    total_race_time: float
    feasible: bool
    feasibility_reason: str
    one_stop_estimate: Optional[float] = None  # For two-stop, show one-stop time for comparison
    explanation: str = ""
    model_info: Dict = None  # Stores model transparency info

    def __post_init__(self):
        """Ensure internal consistency."""
        if self.strategy_type == "one-stop":
            assert self.final_compound is None, "One-stop should have no final_compound"
            assert self.second_pit_lap is None, "One-stop should have no second_pit_lap"
        elif self.strategy_type == "two-stop":
            assert self.final_compound is not None, "Two-stop must have final_compound"
            assert self.second_pit_lap is not None, "Two-stop must have second_pit_lap"
        else:
            raise ValueError(f"Unknown strategy_type: {self.strategy_type}")

        if self.model_info is None:
            self.model_info = {}


def estimate_stint_feasibility(
    degradation_models: Union[Dict, object],
    compound: str,
    tyre_life: int,
    laps_remaining: int,
    pit_loss_value: float,
) -> Tuple[bool, str]:
    """Estimate whether a tyre compound can realistically complete a stint.
    
    Uses a simple heuristic based on observed/modeled tyre life ranges.
    **This is not race-engineer truth; it's an explicit, reviewable estimate.**
    
    **Feasibility heuristic:**
    1. Estimate usable tyre life per compound (conservative defaults from data)
    2. Check if tyre_life + estimated degradation allows finishing laps_remaining
    3. Return feasibility flag and explanation
    
    Args:
        degradation_models: Unified DegradationEvaluationResult or legacy dict
        compound: Target compound (e.g., "MEDIUM")
        tyre_life: Starting tyre life for this stint (typically 1 for new tyre)
        laps_remaining: Laps to complete
        pit_loss_value: Pit stop time cost (for very rough estimate)
        
    Returns:
        (feasible: bool, reason: str)
        - feasible=True: Strategy is plausible
        - feasible=False: Strategy is highly implausible (e.g., 60-lap stint on SOFT)
    """
    # Heuristic: Usable tyre life ranges per compound (from observed data patterns)
    # Conservative: assumes no fuel correction, worst-case degradation
    usable_tyre_life = {
        "SOFT": 20,    # SOFT typically viable up to ~20 laps
        "MEDIUM": 35,  # MEDIUM typically viable up to ~35 laps
        "HARD": 50,    # HARD typically viable up to ~50 laps
    }

    max_viable = usable_tyre_life.get(compound, 30)

    # Check: Does the stint require more laps than realistic tyre life?
    estimated_stint_length = laps_remaining  # Simplified: assume full remaining distance on this tyre

    if estimated_stint_length > max_viable:
        return (
            False,
            f"Unrealistic: {compound} tyre cannot cover {estimated_stint_length} laps "
            f"(max viable ~{max_viable} laps)",
        )

    # Check: Can we predict lap times for this compound?
    predicted_time = predict_lap_time(degradation_models, compound, tyre_life)
    if predicted_time is None:
        return (
            False,
            f"No degradation model available for {compound}",
        )

    # If we got here, the strategy is plausible
    return (
        True,
        f"{compound} is viable for {estimated_stint_length} laps (estimated max ~{max_viable})",
    )


def evaluate_one_stop_strategies(
    degradation_models: Union[Dict, object],
    pit_loss_value: float,
    current_compound: str,
    current_tyre_life: int,
    laps_remaining: int,
    candidate_compounds: Optional[List[str]] = None,
) -> List[StrategyPlan]:
    """Evaluate all candidate one-stop strategies (current -> next -> end).
    
    Args:
        degradation_models: Unified result or legacy dict
        pit_loss_value: Estimated pit stop time (seconds)
        current_compound: Current tyre compound
        current_tyre_life: Current tyre life
        laps_remaining: Laps from now until race end
        candidate_compounds: List of compounds to try (default: ["SOFT", "MEDIUM", "HARD"])
        
    Returns:
        List of StrategyPlan objects, sorted by total_race_time (best first)
    """
    if candidate_compounds is None:
        candidate_compounds = ["SOFT", "MEDIUM", "HARD"]

    plans = []

    for next_compound in candidate_compounds:
        # Skip if same as current (no point in pitting to same compound)
        if next_compound == current_compound:
            continue

        # Evaluate this one-stop option: stay current, pit to next, finish on next
        from src.simulation.strategy import find_optimal_pit_lap

        strategy_df = optimize_pit_window(
            degradation_models=degradation_models,
            pit_loss_value=pit_loss_value,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            compound=current_compound,
            post_pit_compound=next_compound,
        )

        # Validate that strategy_df has no all-NaN data before finding optimal
        if strategy_df.empty or strategy_df["TotalTime"].isna().all():
            logger.warning(
                f"One-stop strategy {current_compound}->{next_compound} produced "
                f"invalid strategy_df (empty or all-NaN). Skipping this option."
            )
            continue

        try:
            optimal_pit_lap, total_race_time = find_optimal_pit_lap(strategy_df)
        except ValueError as e:
            logger.warning(
                f"One-stop strategy {current_compound}->{next_compound} failed to find optimal pit lap: {e}"
            )
            continue

        # Check feasibility
        # For one-stop, the stint on next_compound starts at tyre_life=1 after pit
        remaining_after_pit = laps_remaining - optimal_pit_lap
        feasible, feasibility_reason = estimate_stint_feasibility(
            degradation_models=degradation_models,
            compound=next_compound,
            tyre_life=1,
            laps_remaining=remaining_after_pit,
            pit_loss_value=pit_loss_value,
        )

        plan = StrategyPlan(
            strategy_type="one-stop",
            current_compound=current_compound,
            current_tyre_life=current_tyre_life,
            next_compound=next_compound,
            final_compound=None,
            pit_lap=optimal_pit_lap,
            second_pit_lap=None,
            total_race_time=total_race_time,
            feasible=feasible,
            feasibility_reason=feasibility_reason,
            explanation=f"Pit to {next_compound} at lap {optimal_pit_lap}, finish on {next_compound}",
        )

        plans.append(plan)

    # Sort by total race time (best first)
    plans.sort(key=lambda p: p.total_race_time)
    return plans


def evaluate_two_stop_strategies(
    degradation_models: Union[Dict, object],
    pit_loss_value: float,
    current_compound: str,
    current_tyre_life: int,
    laps_remaining: int,
    candidate_compounds: Optional[List[str]] = None,
    min_stint_length: int = 3,
) -> List[StrategyPlan]:
    """Evaluate candidate two-stop strategies (current -> next -> final -> end).
    
    This is a simplified search that avoids combinatorial explosion:
    - Tries each {next_compound, final_compound} pair from candidates
    - Skips obviously bad combinations (e.g., SOFT -> SOFT)
    - Enforces minimum stint length to avoid zigzagging
    - Uses simplified pit-lap search (not exhaustive for all combinations)
    
    Args:
        degradation_models: Unified result or legacy dict
        pit_loss_value: Estimated pit stop time (seconds)
        current_compound: Current tyre compound
        current_tyre_life: Current tyre life
        laps_remaining: Laps from now until race end
        candidate_compounds: List of compounds to try
        min_stint_length: Minimum laps per stint (prevents short zigzag stints)
        
    Returns:
        List of StrategyPlan objects, sorted by total_race_time (best first)
    """
    if candidate_compounds is None:
        candidate_compounds = ["SOFT", "MEDIUM", "HARD"]

    plans = []

    for next_compound in candidate_compounds:
        # Skip if same as current
        if next_compound == current_compound:
            continue

        for final_compound in candidate_compounds:
            # Skip if final same as next (use one-stop instead)
            if final_compound == next_compound:
                continue

            # Simple pit-lap search for this two-stop:
            # Assume we pit around lap 1/2 of laps_remaining for first pit,
            # then estimate second pit around 3/4 for this simplified version
            first_pit_lap = max(min_stint_length, laps_remaining // 3)
            second_pit_lap = max(first_pit_lap + min_stint_length, 2 * laps_remaining // 3)

            # Ensure valid lap numbers
            if second_pit_lap >= laps_remaining - min_stint_length:
                second_pit_lap = laps_remaining - min_stint_length

            if first_pit_lap >= second_pit_lap:
                continue  # Invalid: first pit must be before second

            # Estimate total time for this two-stop plan
            # Simplified approach: break into three stints, sum times, add pit losses
            total_time = 0.0

            # Stint 1: current compound from now until first pit
            for lap in range(first_pit_lap):
                tyre_age = current_tyre_life + lap
                lap_time = predict_lap_time(degradation_models, current_compound, tyre_age)
                if lap_time is None:
                    lap_time = 95.0  # Fallback
                total_time += lap_time

            # First pit
            total_time += pit_loss_value

            # Stint 2: next compound for (second_pit_lap - first_pit_lap) laps
            stint2_laps = second_pit_lap - first_pit_lap
            for lap in range(stint2_laps):
                tyre_age = 1 + lap
                lap_time = predict_lap_time(degradation_models, next_compound, tyre_age)
                if lap_time is None:
                    lap_time = 95.0
                total_time += lap_time

            # Second pit
            total_time += pit_loss_value

            # Stint 3: final compound for remaining laps
            stint3_laps = laps_remaining - second_pit_lap
            for lap in range(stint3_laps):
                tyre_age = 1 + lap
                lap_time = predict_lap_time(degradation_models, final_compound, tyre_age)
                if lap_time is None:
                    lap_time = 95.0
                total_time += lap_time

            # Check feasibility for all three stints
            feasible_1, reason_1 = estimate_stint_feasibility(
                degradation_models,
                current_compound,
                current_tyre_life,
                first_pit_lap,
                pit_loss_value,
            )
            feasible_2, reason_2 = estimate_stint_feasibility(
                degradation_models,
                next_compound,
                1,
                stint2_laps,
                pit_loss_value,
            )
            feasible_3, reason_3 = estimate_stint_feasibility(
                degradation_models,
                final_compound,
                1,
                stint3_laps,
                pit_loss_value,
            )

            feasible = feasible_1 and feasible_2 and feasible_3
            feasibility_reason = "; ".join([reason_1, reason_2, reason_3])

            # For two-stop, also estimate what a one-stop to next would cost
            one_stop_df = optimize_pit_window(
                degradation_models=degradation_models,
                pit_loss_value=pit_loss_value,
                current_tyre_life=current_tyre_life,
                laps_remaining=laps_remaining,
                compound=current_compound,
                post_pit_compound=next_compound,
            )
            from src.simulation.strategy import find_optimal_pit_lap

            _, one_stop_time = find_optimal_pit_lap(one_stop_df)

            plan = StrategyPlan(
                strategy_type="two-stop",
                current_compound=current_compound,
                current_tyre_life=current_tyre_life,
                next_compound=next_compound,
                final_compound=final_compound,
                pit_lap=first_pit_lap,
                second_pit_lap=second_pit_lap,
                total_race_time=total_time,
                feasible=feasible,
                feasibility_reason=feasibility_reason,
                one_stop_estimate=one_stop_time,
                explanation=f"Pit to {next_compound} (lap {first_pit_lap}), "
                f"then {final_compound} (lap {second_pit_lap}), finish on {final_compound}",
            )

            plans.append(plan)

    # Sort by total race time
    plans.sort(key=lambda p: p.total_race_time)
    return plans


def rank_strategy_plans(
    one_stop_plans: List[StrategyPlan],
    two_stop_plans: List[StrategyPlan],
    prioritize_feasible: bool = True,
) -> List[StrategyPlan]:
    """Rank all strategy options by total time, with optional feasibility prioritization.
    
    Args:
        one_stop_plans: List of one-stop plans (should be pre-sorted)
        two_stop_plans: List of two-stop plans (should be pre-sorted)
        prioritize_feasible: If True, show feasible plans first (still sorted by time within group)
        
    Returns:
        Sorted list of all plans (one-stop + two-stop) with best option first
    """
    all_plans = one_stop_plans + two_stop_plans

    if prioritize_feasible:
        # Separate feasible and infeasible
        feasible_plans = [p for p in all_plans if p.feasible]
        infeasible_plans = [p for p in all_plans if not p.feasible]

        # Sort each group by time
        feasible_plans.sort(key=lambda p: p.total_race_time)
        infeasible_plans.sort(key=lambda p: p.total_race_time)

        return feasible_plans + infeasible_plans
    else:
        # Just sort by time overall
        all_plans.sort(key=lambda p: p.total_race_time)
        return all_plans


def recommend_best_strategy(
    degradation_models: Union[Dict, object],
    pit_loss_value: float,
    current_compound: str,
    current_tyre_life: int,
    laps_remaining: int,
    candidate_compounds: Optional[List[str]] = None,
    include_two_stop: bool = True,
) -> Tuple[StrategyPlan, List[StrategyPlan]]:
    """Automatically recommend the best strategy from ranked options.
    
    **Phase 2A Main Entry Point:**
    Replaces the old manual workflow of:
      "What tyre do you want? Let me pit to that."
    
    With automated workflow of:
      "I recommend this tyre, at this time, for this reason. Here are alternatives."
    
    Args:
        degradation_models: Unified DegradationEvaluationResult or legacy dict
        pit_loss_value: Estimated pit stop time
        current_compound: Current tyre
        current_tyre_life: Current tyre life
        laps_remaining: Laps remaining
        candidate_compounds: None defaults to ["SOFT", "MEDIUM", "HARD"]
        include_two_stop: If True, evaluate two-stop options; else one-stop only
        
    Returns:
        (best_plan: StrategyPlan, all_ranked_plans: List[StrategyPlan])
        best_plan: Top recommendation
        all_ranked_plans: All options ranked by time (best first)
    """
    if candidate_compounds is None:
        candidate_compounds = ["SOFT", "MEDIUM", "HARD"]

    # Evaluate one-stop strategies
    one_stop_plans = evaluate_one_stop_strategies(
        degradation_models=degradation_models,
        pit_loss_value=pit_loss_value,
        current_compound=current_compound,
        current_tyre_life=current_tyre_life,
        laps_remaining=laps_remaining,
        candidate_compounds=candidate_compounds,
    )

    # Evaluate two-stop strategies (if enabled)
    two_stop_plans = []
    if include_two_stop:
        two_stop_plans = evaluate_two_stop_strategies(
            degradation_models=degradation_models,
            pit_loss_value=pit_loss_value,
            current_compound=current_compound,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            candidate_compounds=candidate_compounds,
        )

    # Rank all options
    all_ranked_plans = rank_strategy_plans(one_stop_plans, two_stop_plans, prioritize_feasible=True)

    if not all_ranked_plans:
        raise ValueError("No valid strategies were generated.")

    # Best option is first in ranked list
    best_plan = all_ranked_plans[0]

    return best_plan, all_ranked_plans
