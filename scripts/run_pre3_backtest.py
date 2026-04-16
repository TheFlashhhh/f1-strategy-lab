#!/usr/bin/env python
"""Run the Pre-3 defensibility audit and held-out Miami backtest."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import DataLoader
from src.data.preprocess import build_model_df, clean_laps, detect_pit_stops, select_relevant_columns
from src.features.evaluate_degradation import evaluate_all_degradation
from src.features.hybrid_modeling import build_role_based_hybrid_model, summarize_hybrid_context
from src.simulation.strategy import estimate_pit_loss_window, predict_lap_time
from src.simulation.strategy_engine import StrategyPlan, recommend_best_strategy


def _to_python(value):
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    return value


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_to_python)


def _plan_to_dict(plan: Optional[StrategyPlan]) -> Optional[dict]:
    if plan is None:
        return None
    return {
        "strategy_type": plan.strategy_type,
        "current_compound": plan.current_compound,
        "current_tyre_life": plan.current_tyre_life,
        "next_compound": plan.next_compound,
        "final_compound": plan.final_compound,
        "pit_lap": plan.pit_lap,
        "second_pit_lap": plan.second_pit_lap,
        "total_race_time": round(float(plan.total_race_time), 3),
        "feasible": bool(plan.feasible),
        "feasibility_reason": plan.feasibility_reason,
        "explanation": plan.explanation,
    }


def _simulate_plan_time(
    degradation_models,
    pit_loss_value: float,
    current_compound: str,
    current_tyre_life: int,
    laps_remaining: int,
    next_compound: str,
    pit_lap: int,
    final_compound: Optional[str] = None,
    second_pit_lap: Optional[int] = None,
) -> Optional[float]:
    total_time = 0.0

    for lap in range(pit_lap):
        lap_time = predict_lap_time(degradation_models, current_compound, current_tyre_life + lap)
        if lap_time is None:
            return None
        total_time += float(lap_time)

    total_time += float(pit_loss_value)

    if final_compound is None or second_pit_lap is None:
        for lap in range(laps_remaining - pit_lap):
            lap_time = predict_lap_time(degradation_models, next_compound, 1 + lap)
            if lap_time is None:
                return None
            total_time += float(lap_time)
        return total_time

    second_stint_laps = second_pit_lap - pit_lap
    final_stint_laps = laps_remaining - second_pit_lap

    for lap in range(second_stint_laps):
        lap_time = predict_lap_time(degradation_models, next_compound, 1 + lap)
        if lap_time is None:
            return None
        total_time += float(lap_time)

    total_time += float(pit_loss_value)

    for lap in range(final_stint_laps):
        lap_time = predict_lap_time(degradation_models, final_compound, 1 + lap)
        if lap_time is None:
            return None
        total_time += float(lap_time)

    return total_time


def _match_actual_rank(ranked_plans: list[StrategyPlan], actual_plan: dict) -> tuple[Optional[int], Optional[StrategyPlan]]:
    for index, plan in enumerate(ranked_plans, start=1):
        if (
            plan.strategy_type == actual_plan["strategy_type"]
            and plan.next_compound == actual_plan["next_compound"]
            and plan.final_compound == actual_plan["final_compound"]
            and plan.pit_lap == actual_plan["pit_lap"]
            and plan.second_pit_lap == actual_plan["second_pit_lap"]
        ):
            return index, plan
    return None, None


def _build_actual_remaining_plan(stints: pd.DataFrame, checkpoint_lap: int) -> dict:
    first = stints.iloc[0]
    second = stints.iloc[1]
    actual = {
        "strategy_type": "one-stop" if len(stints) == 2 else "two-stop",
        "next_compound": second["Compound"],
        "pit_lap": int(first["end_lap"] - checkpoint_lap + 1),
        "final_compound": None,
        "second_pit_lap": None,
    }
    if len(stints) >= 3:
        third = stints.iloc[2]
        actual["final_compound"] = third["Compound"]
        actual["second_pit_lap"] = int(second["end_lap"] - checkpoint_lap + 1)
    return actual


def main() -> None:
    print("\n" + "=" * 88)
    print("PRE-3 DEFENSIBILITY AUDIT + MIAMI HOLDOUT BACKTEST")
    print("=" * 88)

    loader = DataLoader(project_root=ROOT)
    miami_raw = loader.load_data("miami_historical")

    role_model, hybrid_context = build_role_based_hybrid_model(project_root=ROOT)
    support_summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "phase": "pre3",
        "blend_summary": summarize_hybrid_context(hybrid_context),
        "compound_support": role_model.to_dict(),
        "warnings": [
            "SOFT should not be treated as equally supported as MEDIUM/HARD unless its tier improves.",
            "2026 data is now framed as bounded recency support rather than direct Miami truth.",
        ],
    }
    support_path = ROOT / "data" / "processed" / "pre3_compound_support_summary.json"
    _json_dump(support_path, support_summary)

    holdout_year = 2024
    train_raw = miami_raw[miami_raw["season"] < holdout_year].copy()
    holdout_raw = miami_raw[miami_raw["season"] == holdout_year].copy()
    if train_raw.empty or holdout_raw.empty:
        raise RuntimeError("Held-out Miami backtest could not be constructed from local data.")

    train_model_df = build_model_df(clean_laps(detect_pit_stops(select_relevant_columns(train_raw))))
    holdout_df = detect_pit_stops(select_relevant_columns(holdout_raw))
    train_models = evaluate_all_degradation(train_model_df, use_fuel_correction=True, use_piecewise=True)

    pit_loss_samples = estimate_pit_loss_window(detect_pit_stops(select_relevant_columns(train_raw)))
    if len(pit_loss_samples) == 0:
        raise RuntimeError("No pit-loss samples were available in the Miami training subset.")
    pit_loss_value = float(np.median(pit_loss_samples))

    print(f"Holdout year: Miami {holdout_year}")
    print(f"Training years: {sorted(train_raw['season'].unique().tolist())}")
    print(f"Training Miami pit-loss baseline: {pit_loss_value:.2f}s")

    checkpoints = [
        {
            "checkpoint_id": "bot_soft_early",
            "driver": "BOT",
            "checkpoint_lap": 5,
            "rationale": "Early first-stint SOFT checkpoint in a real two-stop race.",
        },
        {
            "checkpoint_id": "lec_medium_mid",
            "driver": "LEC",
            "checkpoint_lap": 12,
            "rationale": "Mid first-stint MEDIUM checkpoint in a one-stop race.",
        },
        {
            "checkpoint_id": "ham_hard_mid",
            "driver": "HAM",
            "checkpoint_lap": 15,
            "rationale": "HARD-compound checkpoint before a medium switch.",
        },
        {
            "checkpoint_id": "nor_medium_late",
            "driver": "NOR",
            "checkpoint_lap": 24,
            "rationale": "Late first-stint MEDIUM checkpoint close to the actual stop.",
        },
    ]

    holdout_max_lap = int(holdout_df["LapNumber"].max())
    backtest_results = []

    for checkpoint in checkpoints:
        driver_df = holdout_df[holdout_df["Driver"] == checkpoint["driver"]].copy()
        stints = (
            driver_df.dropna(subset=["Compound"])
            .groupby(["Stint", "Compound"], as_index=False)
            .agg(start_lap=("LapNumber", "min"), end_lap=("LapNumber", "max"))
            .sort_values("start_lap")
            .reset_index(drop=True)
        )
        if len(stints) < 2:
            continue
        first_stint = stints.iloc[0]
        if not (int(first_stint["start_lap"]) <= checkpoint["checkpoint_lap"] <= int(first_stint["end_lap"])):
            continue

        row = driver_df[driver_df["LapNumber"] == checkpoint["checkpoint_lap"]].iloc[0]
        current_compound = row["Compound"]
        current_tyre_life = int(row["TyreLife"])
        laps_remaining = holdout_max_lap - checkpoint["checkpoint_lap"] + 1

        best_plan, ranked_plans = recommend_best_strategy(
            degradation_models=train_models,
            pit_loss_value=pit_loss_value,
            current_compound=current_compound,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            candidate_compounds=["SOFT", "MEDIUM", "HARD"],
            include_two_stop=True,
        )

        actual_plan = _build_actual_remaining_plan(stints, checkpoint["checkpoint_lap"])
        actual_rank, actual_ranked_plan = _match_actual_rank(ranked_plans, actual_plan)
        actual_plan_time = _simulate_plan_time(
            degradation_models=train_models,
            pit_loss_value=pit_loss_value,
            current_compound=current_compound,
            current_tyre_life=current_tyre_life,
            laps_remaining=laps_remaining,
            next_compound=actual_plan["next_compound"],
            pit_lap=actual_plan["pit_lap"],
            final_compound=actual_plan["final_compound"],
            second_pit_lap=actual_plan["second_pit_lap"],
        )
        delta_vs_actual = (
            round(float(actual_plan_time - best_plan.total_race_time), 3)
            if actual_plan_time is not None
            else None
        )

        result = {
            "checkpoint_id": checkpoint["checkpoint_id"],
            "driver": checkpoint["driver"],
            "checkpoint_lap": checkpoint["checkpoint_lap"],
            "rationale": checkpoint["rationale"],
            "current_compound": current_compound,
            "current_tyre_life": current_tyre_life,
            "laps_remaining": laps_remaining,
            "model_best": _plan_to_dict(best_plan),
            "actual_remaining_strategy": actual_plan,
            "actual_strategy_rank": actual_rank,
            "actual_strategy_in_top_3": actual_rank is not None and actual_rank <= 3,
            "actual_strategy_estimated_time": round(float(actual_plan_time), 3) if actual_plan_time is not None else None,
            "delta_model_best_vs_actual_s": delta_vs_actual,
            "top_3_model_plans": [_plan_to_dict(plan) for plan in ranked_plans[:3]],
            "actual_ranked_plan": _plan_to_dict(actual_ranked_plan),
        }
        backtest_results.append(result)

    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "phase": "pre3",
        "holdout_year": holdout_year,
        "training_years": sorted(train_raw["season"].unique().tolist()),
        "design": {
            "target": "held-out Miami decision-support backtest",
            "why_2024": "Miami 2024 is cleaner in the local data than Miami 2025, which contains large early-race compound gaps.",
            "leakage_control": "The backtest trains on earlier Miami races only and excludes 2026 recency data to avoid future leakage.",
        },
        "training_pit_loss_value": round(pit_loss_value, 3),
        "checkpoints": backtest_results,
        "summary": {
            "checkpoint_count": len(backtest_results),
            "actual_strategy_top_3_hits": sum(1 for item in backtest_results if item["actual_strategy_in_top_3"]),
            "actual_strategy_rank_available": sum(1 for item in backtest_results if item["actual_strategy_rank"] is not None),
            "average_delta_model_best_vs_actual_s": (
                round(
                    float(
                        np.mean(
                            [
                                item["delta_model_best_vs_actual_s"]
                                for item in backtest_results
                                if item["delta_model_best_vs_actual_s"] is not None
                            ]
                        )
                    ),
                    3,
                )
                if any(item["delta_model_best_vs_actual_s"] is not None for item in backtest_results)
                else None
            ),
        },
    }

    backtest_path = ROOT / "data" / "processed" / "pre3_backtest_summary.json"
    _json_dump(backtest_path, summary)

    print("\nCompound support tiers:")
    for compound, support in role_model.to_dict().items():
        print(
            f"  {compound}: {support['support_tier']} | "
            f"Miami model laps {support['miami']['model_laps']} | "
            f"2026 model laps {support['recency']['model_laps']}"
        )

    print("\nBacktest checkpoints:")
    for item in backtest_results:
        actual = item["actual_remaining_strategy"]
        top3 = "yes" if item["actual_strategy_in_top_3"] else "no"
        print(
            f"  {item['checkpoint_id']}: model={item['model_best']['next_compound']} @ L{item['model_best']['pit_lap']} | "
            f"actual={actual['next_compound']} @ L{actual['pit_lap']} | top3={top3}"
        )

    print("\nArtifacts:")
    print(f"  - {support_path}")
    print(f"  - {backtest_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
