from __future__ import annotations

import numpy as np
import pandas as pd


CHAMPIONSHIP_DOMAINS = {
    "Rate production": "PRODUCTION_SCORE",
    "Role burden": "ROLE_BURDEN_SCORE",
    "Cumulative impact": "CUMULATIVE_IMPACT_SCORE",
    "Terminal-series responsibility": "TERMINAL_RESPONSIBILITY_SCORE",
    "Historical defense evidence": "HISTORICAL_DEFENSE_EVIDENCE_SCORE",
    "Primary-star compatibility": "ROLE_COMPATIBILITY_SCORE",
}


def championship_weight_sensitivity(
    frame: pd.DataFrame,
    simulations: int = 50_000,
    seed: int = 20_260_910,
) -> pd.DataFrame:
    """Stress-test rankings across uncertain definitions of second-option value.

    Uniform Dirichlet draws span the full six-domain weight simplex. The resulting
    interval is uncertainty about the *value judgment* encoded by weights; it is
    not a player-performance confidence interval.
    """
    required = {"POSTSEASON_FINISH", "RANKING_ELIGIBLE", *CHAMPIONSHIP_DOMAINS.values()}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Sensitivity model missing columns: {sorted(missing)}")
    champions = frame[
        (frame.POSTSEASON_FINISH == "Champion")
        & frame.RANKING_ELIGIBLE.fillna(False).astype(bool)
    ].copy()
    values = champions[list(CHAMPIONSHIP_DOMAINS.values())].apply(
        pd.to_numeric, errors="coerce")
    complete = values.notna().all(axis=1)
    champions, values = champions[complete].copy(), values[complete]
    if champions.empty:
        return champions

    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(values.shape[1]), size=simulations)
    scores = values.to_numpy() @ weights.T
    ranks = np.empty_like(scores)
    for simulation in range(simulations):
        ranks[:, simulation] = (-scores[:, simulation]).argsort().argsort() + 1

    champions["SENSITIVITY_MEDIAN_RANK"] = np.median(ranks, axis=1)
    champions["SENSITIVITY_P10_RANK"] = np.quantile(ranks, .10, axis=1)
    champions["SENSITIVITY_P90_RANK"] = np.quantile(ranks, .90, axis=1)
    champions["SENSITIVITY_TOP_10_PROBABILITY"] = (ranks <= 10).mean(axis=1)
    champions["SENSITIVITY_TOP_5_PROBABILITY"] = (ranks <= 5).mean(axis=1)
    champions["SENSITIVITY_FIRST_PROBABILITY"] = (ranks == 1).mean(axis=1)
    champions["SENSITIVITY_MEAN_SCORE"] = scores.mean(axis=1)
    champions["SENSITIVITY_SIMULATIONS"] = simulations
    champions["SENSITIVITY_SEED"] = seed
    return champions.sort_values(
        ["SENSITIVITY_MEDIAN_RANK", "SENSITIVITY_MEAN_SCORE"],
        ascending=[True, False],
    )
