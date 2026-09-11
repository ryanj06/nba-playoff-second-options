"""Build constant-lineup scoring stints from official NBA game feeds."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import SchemaError
from .sources import require_columns

_CLOCK = re.compile(r"PT(?P<minutes>\d+)M(?P<seconds>\d+(?:\.\d+)?)S")


def elapsed_deciseconds(period: int, clock: str) -> float:
    """Convert an NBA ISO clock to elapsed game deciseconds."""
    match = _CLOCK.fullmatch(str(clock))
    if not match or period < 1:
        raise ValueError(f"Invalid NBA clock: period={period!r}, clock={clock!r}")
    remaining = 60 * int(match.group("minutes")) + float(match.group("seconds"))
    period_length = 720 if period <= 4 else 300
    if remaining > period_length:
        raise ValueError(f"Clock exceeds period length: {clock!r}")
    prior = 720 * min(period - 1, 4) + 300 * max(period - 5, 0)
    return round((prior + period_length - remaining) * 10, 3)


def _score_timeline(play_by_play: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_columns(
        play_by_play,
        ["period", "clock", "actionNumber", "scoreHome", "scoreAway"],
        "play-by-play",
    )
    scores = play_by_play.copy()
    scores["ELAPSED"] = [
        elapsed_deciseconds(int(period), clock)
        for period, clock in zip(scores.period, scores.clock, strict=True)
    ]
    for column in ("scoreHome", "scoreAway"):
        scores[column] = pd.to_numeric(
            scores[column].replace("", np.nan), errors="coerce"
        )
    scores = scores.sort_values(["ELAPSED", "actionNumber"])
    scores[["scoreHome", "scoreAway"]] = scores[
        ["scoreHome", "scoreAway"]
    ].ffill().fillna(0)
    scores = scores.groupby("ELAPSED", as_index=False).tail(1)
    return (
        scores.ELAPSED.to_numpy(dtype=float),
        scores.scoreHome.to_numpy(dtype=float),
        scores.scoreAway.to_numpy(dtype=float),
    )


def _score_at(
    elapsed: float,
    timeline: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float]:
    times, home, away = timeline
    position = int(np.searchsorted(times, elapsed, side="right") - 1)
    if position < 0:
        return 0.0, 0.0
    return float(home[position]), float(away[position])


def build_constant_lineup_stints(
    rotation: pd.DataFrame,
    play_by_play: pd.DataFrame,
) -> pd.DataFrame:
    """Return exact lineup intervals and their observed scoring margin.

    This function intentionally does not label the observed margin as causal or
    convert it to per-100 possessions. Possessions and opponent controls belong
    in the later model-building stage.
    """
    require_columns(
        rotation,
        [
            "GAME_ID",
            "TEAM_ID",
            "PERSON_ID",
            "IN_TIME_REAL",
            "OUT_TIME_REAL",
            "IS_HOME",
        ],
        "game rotation",
    )
    if rotation.GAME_ID.astype(str).nunique() != 1:
        raise SchemaError("Rotation input must contain exactly one game")
    team_sides = rotation[["TEAM_ID", "IS_HOME"]].drop_duplicates()
    if len(team_sides) != 2 or team_sides.IS_HOME.nunique() != 2:
        raise SchemaError("Rotation input must contain one home and one away team")

    clean = rotation.copy()
    clean["IN_TIME_REAL"] = pd.to_numeric(clean.IN_TIME_REAL, errors="coerce")
    clean["OUT_TIME_REAL"] = pd.to_numeric(clean.OUT_TIME_REAL, errors="coerce")
    if clean[["IN_TIME_REAL", "OUT_TIME_REAL"]].isna().any().any():
        raise SchemaError("Rotation input contains invalid in/out times")
    boundaries = np.unique(
        np.concatenate([clean.IN_TIME_REAL.to_numpy(), clean.OUT_TIME_REAL.to_numpy()])
    )
    boundaries.sort()
    timeline = _score_timeline(play_by_play)
    game_id = str(clean.GAME_ID.iloc[0])
    home_team = int(team_sides.loc[team_sides.IS_HOME, "TEAM_ID"].iloc[0])
    away_team = int(team_sides.loc[~team_sides.IS_HOME, "TEAM_ID"].iloc[0])

    records: list[dict[str, object]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if end <= start:
            continue
        active = clean[(clean.IN_TIME_REAL <= start) & (clean.OUT_TIME_REAL >= end)]
        home_players = sorted(
            active.loc[active.IS_HOME, "PERSON_ID"].astype(int).unique().tolist()
        )
        away_players = sorted(
            active.loc[~active.IS_HOME, "PERSON_ID"].astype(int).unique().tolist()
        )
        if len(home_players) != 5 or len(away_players) != 5:
            raise SchemaError(
                f"{game_id} {start:g}-{end:g}: expected five players per team; "
                f"found home={len(home_players)}, away={len(away_players)}"
            )
        start_home, start_away = _score_at(float(start), timeline)
        end_home, end_away = _score_at(float(end), timeline)
        records.append({
            "GAME_ID": game_id,
            "START_DECISECONDS": float(start),
            "END_DECISECONDS": float(end),
            "DURATION_SECONDS": float((end - start) / 10),
            "HOME_TEAM_ID": home_team,
            "AWAY_TEAM_ID": away_team,
            "HOME_PLAYER_IDS": "|".join(map(str, home_players)),
            "AWAY_PLAYER_IDS": "|".join(map(str, away_players)),
            "START_HOME_SCORE": start_home,
            "START_AWAY_SCORE": start_away,
            "END_HOME_SCORE": end_home,
            "END_AWAY_SCORE": end_away,
            "HOME_POINT_MARGIN": (end_home - start_home) - (end_away - start_away),
            "OBSERVATION_STATUS": "OBSERVED_NOT_CAUSAL",
        })
    return pd.DataFrame(records)
