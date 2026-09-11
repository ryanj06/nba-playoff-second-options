from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def season_end_year(season: pd.Series) -> pd.Series:
    """Return the ending year for NBA season labels such as ``2019-20``."""
    return pd.to_numeric(season.astype(str).str[:4], errors="coerce") + 1


def _safe_rate(numerator: pd.Series, denominator: pd.Series, scale: float = 1.0) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / denominator * scale


def _percentile_of(values: pd.Series, observations: pd.Series) -> pd.Series:
    reference = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    result = pd.Series(np.nan, index=observations.index, dtype=float)
    if not len(reference):
        return result
    for index, value in pd.to_numeric(observations, errors="coerce").items():
        if pd.notna(value):
            below = np.count_nonzero(reference < value)
            tied = np.count_nonzero(reference == value)
            result.loc[index] = (below + .5 * tied) / len(reference) * 100
    return result


def attach_season_environment(players: pd.DataFrame, runs: pd.DataFrame,
                              config: PipelineConfig) -> pd.DataFrame:
    """Attach same-season playoff context from every qualifying rotation player.

    Conference Finalists remain the reporting population, but the era baseline is
    deliberately built from all playoff players meeting the minutes/games filter.
    """
    population = players.copy()
    population["MPG"] = _safe_rate(population.MIN, population.GP)
    population["PPG_ENV"] = _safe_rate(population.PTS, population.GP)
    population["TS_ENV"] = _safe_rate(
        population.PTS, 2 * (population.FGA + .44 * population.FTA))
    population["PTS_PER_75_ENV"] = _safe_rate(population.PTS, population.POSS, 75)
    population["FG3A_PER_75_ENV"] = _safe_rate(population.FG3A, population.POSS, 75)
    population_stl = pd.to_numeric(
        population.get("STL", pd.Series(np.nan, index=population.index)), errors="coerce")
    population_blk = pd.to_numeric(
        population.get("BLK", pd.Series(np.nan, index=population.index)), errors="coerce")
    population["STOCKS_PER_75_ENV"] = _safe_rate(
        population_stl + population_blk, population.POSS, 75)
    population = population[
        (population.GP >= config.min_games) & (population.MPG >= config.min_mpg)
    ].copy()
    out = runs.copy()
    if "MPG" not in out:
        out["MPG"] = _safe_rate(out.MIN, out.GP)
    out["PTS_PER_75"] = _safe_rate(out.PTS, out.POSS, 75)
    out["FG3A_PER_75"] = _safe_rate(out.FG3A, out.POSS, 75)
    out_stl = pd.to_numeric(
        out.get("STL", pd.Series(np.nan, index=out.index)), errors="coerce")
    out_blk = pd.to_numeric(
        out.get("BLK", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["STOCKS_PER_75"] = _safe_rate(out_stl + out_blk, out.POSS, 75)
    if "TEAM_ID" in players and "TEAM_ID" in out:
        available_team_stats = [
            stat for stat in ["PTS", "FGA", "AST", "MIN", "REB", "BLK"]
            if stat in players and stat in out]
        team_totals = players.groupby("TEAM_ID")[available_team_stats].sum()
        for stat in available_team_stats:
            denominator = out.TEAM_ID.map(team_totals[stat])
            out[f"TEAM_{stat}_SHARE"] = _safe_rate(out[stat], denominator)
    for stat in ["PTS", "FGA", "AST", "MIN", "REB", "BLK"]:
        if f"TEAM_{stat}_SHARE" not in out:
            out[f"TEAM_{stat}_SHARE"] = np.nan
    metric_map = {
        "PPG": "PPG_ENV",
        "PTS75": "PTS_PER_75_ENV",
        "TS": "TS_ENV",
        "USG": "USG_PCT",
        "AST": "AST_PCT",
        "FG3A75": "FG3A_PER_75_ENV",
        "MPG": "MPG",
        "STOCKS75": "STOCKS_PER_75_ENV",
    }
    run_columns = {
        "PPG": "PPG",
        "PTS75": "PTS_PER_75",
        "TS": "TS_PCT",
        "USG": "USG_PCT",
        "AST": "AST_PCT",
        "FG3A75": "FG3A_PER_75",
        "MPG": "MPG",
        "STOCKS75": "STOCKS_PER_75",
    }
    for label, population_column in metric_map.items():
        reference = pd.to_numeric(population[population_column], errors="coerce")
        out[f"ERA_{label}_MEAN"] = reference.mean()
        out[f"ERA_{label}_STD"] = reference.std(ddof=0)
        out[f"ERA_{label}_PERCENTILE"] = _percentile_of(
            reference, out[run_columns[label]])
    total_possessions = pd.to_numeric(population.POSS, errors="coerce").sum()
    out["ERA_ENVIRONMENT_SAMPLE_PLAYERS"] = len(population)
    out["ERA_ENVIRONMENT_SAMPLE_POSSESSIONS"] = total_possessions
    era_ts_denominator = 2 * (population.FGA.sum() + .44 * population.FTA.sum())
    out["ERA_PLAYOFF_TS_PCT"] = (
        population.PTS.sum() / era_ts_denominator if era_ts_denominator else np.nan)
    out["ERA_PLAYOFF_3PA_RATE"] = (
        population.FG3A.sum() / population.FGA.sum() if population.FGA.sum() else np.nan)
    out["ERA_PLAYOFF_FTA_RATE"] = (
        population.FTA.sum() / population.FGA.sum() if population.FGA.sum() else np.nan)
    out["ERA_PLAYOFF_PACE"] = pd.to_numeric(population.PACE, errors="coerce").mean()
    out["ERA_PLAYOFF_OFF_RATING"] = pd.to_numeric(
        population.OFF_RATING, errors="coerce").mean()
    out["RELATIVE_TS_PCT"] = out.TS_PCT - out.ERA_PLAYOFF_TS_PCT
    out["ERA_BASELINE_METHOD"] = "ALL_PLAYOFF_ROTATION_PLAYERS"
    return out


def rolling_era_percentile(frame: pd.DataFrame, column: str, *, higher: bool = True,
                           radius: int = 1) -> pd.Series:
    """Percentile against a centered, current-season-weighted era window."""
    if column not in frame or "SEASON" not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    years = season_end_year(frame.SEASON)
    values = pd.to_numeric(frame[column], errors="coerce")
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for index in frame.index:
        value, year = values.loc[index], years.loc[index]
        if pd.isna(value) or pd.isna(year):
            continue
        nearby = values[(years - year).abs() <= radius].dropna()
        current = values[years == year].dropna()
        reference = pd.concat([nearby, current], ignore_index=True)
        if reference.empty:
            continue
        below = (reference < value).sum()
        tied = (reference == value).sum()
        score = (below + .5 * tied) / len(reference) * 100
        result.loc[index] = score if higher else 100 - score
    return result
