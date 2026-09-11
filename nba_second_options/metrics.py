from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

from .config import COMPLETE, NOT_MODELED, PARTIAL, PipelineConfig, PipelineError
from .era import rolling_era_percentile
from .sources import require_columns

SCORING_LOAD_WEIGHTS = {"PPG": .45, "USG_PCT": .30, "FGA_PER_GAME": .25}
CREATION_ENGINE_WEIGHTS = {"AST_PCT": .75, "AST_TO_RATIO": .25}
PRIMARY_ROLE_WEIGHTS = {"SCORING_LOAD": .40, "CREATION_ENGINE": .30,
                        "IMPACT_SIGNAL": .25, "MINUTES_LOAD": .05}
SECONDARY_ROLE_WEIGHTS = {"SCORING_LOAD": .65, "CREATION_ENGINE": .10,
                          "IMPACT_SIGNAL": .20, "MINUTES_LOAD": .05}


def normalized_name(value: Any) -> str:
    text = str(value)
    if "Ã" in text or "Â" in text:
        try:
            text = text.encode("latin-1").decode("utf-8")
        except UnicodeError:
            pass
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    # Only remove generational suffixes at the end. "JR Smith" is a first name,
    # while "Gary Payton II" contains an actual suffix.
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\.?\s*\*?$", "", text.lower())
    return re.sub(r"[^a-z0-9]+", "", text)


def series_wins_and_finish(games: pd.DataFrame) -> pd.DataFrame:
    require_columns(games, ["TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP", "WL"], "games")
    work = games.copy()
    work["OPP"] = work["MATCHUP"].astype(str).str.extract(r"(?:vs\.|@)\s*([A-Z]{2,3})")
    records = []
    for (team_id, team, opp), group in work.dropna(subset=["OPP"]).groupby(
            ["TEAM_ID", "TEAM_ABBREVIATION", "OPP"]):
        wins, losses = int((group.WL == "W").sum()), int((group.WL == "L").sum())
        records.append({"TEAM_ID": int(team_id), "TEAM_ABBREVIATION": team,
                        "OPP": opp, "SERIES_WON": int(wins > losses)})
    if not records:
        return pd.DataFrame(columns=["TEAM_ID", "TEAM_ABBREVIATION", "SERIES_WINS",
                                     "POSTSEASON_FINISH"])
    result = (pd.DataFrame(records).groupby(
        ["TEAM_ID", "TEAM_ABBREVIATION"], as_index=False).SERIES_WON.sum()
        .rename(columns={"SERIES_WON": "SERIES_WINS"}))
    result["POSTSEASON_FINISH"] = np.select(
        [result.SERIES_WINS >= 4, result.SERIES_WINS >= 3, result.SERIES_WINS >= 2],
        ["Champion", "Finals", "Conference Finals"], default="Earlier Round")
    return result


def conference_finalists(all_games: pd.DataFrame, round_three: pd.DataFrame) -> pd.DataFrame:
    inferred = series_wins_and_finish(all_games)
    inferred_ids = set(inferred.loc[inferred.SERIES_WINS >= 2, "TEAM_ID"].astype(int))
    round_ids = (set(round_three.TEAM_ID.astype(int))
                 if not round_three.empty and "TEAM_ID" in round_three else set())
    selected = round_ids if len(round_ids) == 4 else inferred_ids
    if len(selected) != 4:
        raise PipelineError(f"Expected four Conference Finalists; found {len(selected)}: {selected}")
    result = inferred[inferred.TEAM_ID.isin(selected)].copy()
    result["QUALIFIER_METHOD"] = "NBA_POROUND_3" if len(round_ids) == 4 else "SERIES_INFERENCE"
    result["QUALIFIER_VALIDATION"] = (
        "AGREES" if round_ids == inferred_ids else
        "ROUND_UNAVAILABLE" if not round_ids else "POROUND_FILTER_IGNORED")
    return result


def merge_player_stats(base: pd.DataFrame, advanced: pd.DataFrame) -> pd.DataFrame:
    keys = ["PLAYER_ID", "TEAM_ID"]
    require_columns(base, keys + ["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN",
                                 "PTS", "FGA", "FTA", "FG3M", "FG3A"], "base stats")
    require_columns(advanced, keys + ["USG_PCT"], "advanced stats")
    keep = [c for c in advanced.columns if c not in base.columns or c in keys]
    return base.merge(advanced[keep], on=keys, how="left", validate="one_to_one")


def rank_offensive_roles(players: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Calculate distinct primary-engine and secondary-option role scores.

    A primary is the structural star/offensive engine; a secondary is the next
    scoring and creation option. This remains a proposal generator rather than
    ground truth, so close calls are flagged for editorial review.
    """
    work = players.copy()
    required = ["TEAM_ID", "PLAYER_ID", "PLAYER_NAME", "GP", "MIN", "PTS",
                "FGA", "AST", "TOV", "USG_PCT", "AST_PCT", "PIE"]
    require_columns(work, required, "offensive role inputs")
    for column in ["GP", "MIN", "PTS", "FGA", "AST", "TOV", "USG_PCT",
                   "AST_PCT", "PIE"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["MPG"] = work.MIN / work.GP
    work["PPG"] = work.PTS / work.GP
    work["FGA_PER_GAME"] = work.FGA / work.GP
    work["AST_TO_RATIO"] = work.AST / work.TOV.where(work.TOV > 0)
    work = work[(work.GP >= config.min_games) & (work.MPG >= config.min_mpg)].copy()
    score_columns = ["SCORING_LOAD", "CREATION_ENGINE", "IMPACT_SIGNAL",
                     "MINUTES_LOAD", "PRIMARY_SCORE", "SECONDARY_SCORE"]
    work[score_columns] = 0.0
    for _, group in work.groupby("TEAM_ID"):
        percentiles: dict[str, pd.Series] = {}
        for metric in [*SCORING_LOAD_WEIGHTS, *CREATION_ENGINE_WEIGHTS,
                       "PIE", "MPG"]:
            low, high = group[metric].min(), group[metric].max()
            percentiles[metric] = (
                (group[metric] - low) / (high - low)
                if pd.notna(low) and pd.notna(high) and high > low
                else pd.Series(.5, index=group.index)
            ).fillna(.5)
        scoring = sum(percentiles[m] * w for m, w in SCORING_LOAD_WEIGHTS.items())
        creation = sum(percentiles[m] * w for m, w in CREATION_ENGINE_WEIGHTS.items())
        components = {
            "SCORING_LOAD": scoring,
            "CREATION_ENGINE": creation,
            "IMPACT_SIGNAL": percentiles["PIE"],
            "MINUTES_LOAD": percentiles["MPG"],
        }
        for column, values in components.items():
            work.loc[group.index, column] = values * 100
        work.loc[group.index, "PRIMARY_SCORE"] = sum(
            components[m] * w * 100 for m, w in PRIMARY_ROLE_WEIGHTS.items()
        )
        work.loc[group.index, "SECONDARY_SCORE"] = sum(
            components[m] * w * 100 for m, w in SECONDARY_ROLE_WEIGHTS.items()
        )
    work["ROLE_SCORE"] = work["PRIMARY_SCORE"]
    work = work.sort_values(["TEAM_ID", "PRIMARY_SCORE", "PPG", "USG_PCT"],
                            ascending=[True, False, False, False])
    work["ROLE_RANK"] = work.groupby("TEAM_ID").cumcount() + 1
    return work


def role_pairing_proposals(ranked: pd.DataFrame) -> pd.DataFrame:
    """Return one proposed #1/#2 pairing plus #3 challenger per team."""
    records = []
    for team_id, group in ranked.groupby("TEAM_ID"):
        ordered = group.sort_values("PRIMARY_SCORE", ascending=False)
        if len(ordered) < 2:
            continue
        first = ordered.iloc[0]
        primary_runner_up = ordered.iloc[1]
        remaining = group[group.PLAYER_ID != first.PLAYER_ID].sort_values(
            "SECONDARY_SCORE", ascending=False
        )
        second = remaining.iloc[0]
        third = remaining.iloc[1] if len(remaining) > 1 else None
        hierarchy_margin = float(first.PRIMARY_SCORE - primary_runner_up.PRIMARY_SCORE)
        secondary_margin = (float(second.SECONDARY_SCORE - third.SECONDARY_SCORE)
                            if third is not None else np.nan)
        if hierarchy_margin < 5:
            status = "AMBIGUOUS_HIERARCHY"
        elif pd.notna(secondary_margin) and secondary_margin < 8:
            status = "AMBIGUOUS_SECONDARY"
        else:
            status = "CLEAR"
        record = {"TEAM_ID": int(team_id),
            "TEAM_ABBREVIATION": first.get("TEAM_ABBREVIATION", ""),
            "PRIMARY_PLAYER_ID": int(first.PLAYER_ID), "PRIMARY": first.PLAYER_NAME,
            "PRIMARY_ROLE_SCORE": first.PRIMARY_SCORE, "PRIMARY_PPG": first.PPG,
            "PRIMARY_USG_PCT": first.USG_PCT, "PRIMARY_AST_PCT": first.AST_PCT,
            "PRIMARY_SCORING_LOAD": first.SCORING_LOAD,
            "PRIMARY_CREATION_ENGINE": first.CREATION_ENGINE,
            "PRIMARY_IMPACT_SIGNAL": first.IMPACT_SIGNAL,
            "SECONDARY_PLAYER_ID": int(second.PLAYER_ID), "SECONDARY": second.PLAYER_NAME,
            "SECONDARY_ROLE_SCORE": second.SECONDARY_SCORE, "SECONDARY_PPG": second.PPG,
            "SECONDARY_USG_PCT": second.USG_PCT, "SECONDARY_AST_PCT": second.AST_PCT,
            "SECONDARY_SCORING_LOAD": second.SCORING_LOAD,
            "SECONDARY_CREATION_ENGINE": second.CREATION_ENGINE,
            "SECONDARY_IMPACT_SIGNAL": second.IMPACT_SIGNAL,
            "THIRD": third.PLAYER_NAME if third is not None else "",
            "THIRD_ROLE_SCORE": third.SECONDARY_SCORE if third is not None else np.nan,
            "HIERARCHY_MARGIN": hierarchy_margin, "SECONDARY_MARGIN": secondary_margin,
            "ROLE_REVIEW_STATUS": status}
        records.append(record)
    return pd.DataFrame(records)


def select_options(players: pd.DataFrame, qualifiers: pd.DataFrame,
                   config: PipelineConfig, season: str | None = None) -> pd.DataFrame:
    """Select reviewed roles, preserving the model proposal beside the final label."""
    from .role_audit import REVIEWED_PAIRINGS

    work = players.merge(qualifiers, on=["TEAM_ID", "TEAM_ABBREVIATION"], how="inner")
    numeric = ["GP", "MIN", "USG_PCT", "PTS", "FGA", "FTA", "FG3M", "FG3A",
               "AST", "TOV", "AST_PCT", "PIE"]
    work[numeric] = work[numeric].apply(pd.to_numeric, errors="coerce")
    ranked = rank_offensive_roles(work, config)
    proposals = role_pairing_proposals(ranked)
    records: list[dict[str, Any]] = []
    for _, proposal in proposals.iterrows():
        team_id = int(proposal.TEAM_ID)
        team = str(proposal.TEAM_ABBREVIATION)
        reviewed = REVIEWED_PAIRINGS.get((season, team)) if season else None
        primary_name = reviewed.primary if reviewed else str(proposal.PRIMARY)
        second_name = reviewed.secondary if reviewed else str(proposal.SECONDARY)
        team_players = ranked[ranked.TEAM_ID == team_id]
        primary_rows = team_players[
            team_players.PLAYER_NAME.map(normalized_name) == normalized_name(primary_name)]
        second_rows = team_players[
            team_players.PLAYER_NAME.map(normalized_name) == normalized_name(second_name)]
        if primary_rows.empty or second_rows.empty:
            raise PipelineError(
                f"{season} {team}: reviewed role player missing or ineligible")
        primary, second = primary_rows.iloc[0], second_rows.iloc[0]
        row = second.to_dict()
        row.update({
            "OPTION_RANK": 2,
            "PRIMARY_PLAYER_ID": int(primary.PLAYER_ID),
            "PRIMARY_PLAYER_NAME": primary.PLAYER_NAME,
            "PRIMARY_USG_PCT": primary.USG_PCT,
            "PRIMARY_AST_PCT": primary.AST_PCT,
            "MODEL_PRIMARY_PLAYER_NAME": proposal.PRIMARY,
            "MODEL_SECONDARY_PLAYER_NAME": proposal.SECONDARY,
            "ROLE_DECISION_STATUS": reviewed.status if reviewed else proposal.ROLE_REVIEW_STATUS,
            "ROLE_DECISION_REASON": reviewed.note if reviewed else "Model proposal retained.",
            "ROLE_PAIRING_SOURCE": reviewed.status if reviewed else "MODEL_PROPOSAL",
        })
        records.append(row)
    second = pd.DataFrame(records)
    missing = set(qualifiers.TEAM_ID) - set(second.TEAM_ID)
    if missing:
        raise PipelineError(f"No eligible second option for team IDs: {sorted(missing)}")
    return second


def true_shooting(points: pd.Series, fga: pd.Series, fta: pd.Series) -> pd.Series:
    denominator = 2 * (fga + .44 * fta)
    return points.div(denominator.where(denominator > 0))


def beta_binomial_shrinkage(made: pd.Series, attempts: pd.Series,
                            fallback_strength: float = 75) -> tuple[pd.Series, float, float]:
    made = pd.to_numeric(made, errors="coerce").fillna(0)
    attempts = pd.to_numeric(attempts, errors="coerce").fillna(0)
    valid = attempts > 0
    mean = float(made.sum() / attempts.sum()) if attempts.sum() else .36
    rates = (made[valid] / attempts[valid]).clip(1e-6, 1 - 1e-6)
    if len(rates) >= 8 and rates.var(ddof=1) > 0:
        noise = float((mean * (1 - mean) / attempts[valid]).mean())
        between = max(float(rates.var(ddof=1)) - noise, 1e-5)
        strength = max(5., min(250., mean * (1 - mean) / between - 1))
    else:
        strength = fallback_strength
    alpha, beta = mean * strength, (1 - mean) * strength
    posterior = ((made + alpha) / (attempts + alpha + beta)).where(valid, mean)
    return posterior, alpha, beta


def parse_lineup_ids(row: pd.Series) -> set[int]:
    values = str(row.get("GROUP_ID", ""))
    if not re.search(r"\d", values):
        values = str(row.get("GROUP_NAME", ""))
    return {int(value) for value in re.findall(r"\d+", values)}


def weighted_rating(frame: pd.DataFrame, column: str) -> tuple[float, float, str]:
    if frame.empty or column not in frame:
        return math.nan, 0., NOT_MODELED
    weight_col = "POSS" if "POSS" in frame else "MIN" if "MIN" in frame else None
    if not weight_col:
        return math.nan, 0., NOT_MODELED
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0)
    valid = values.notna() & (weights > 0)
    if not valid.any():
        return math.nan, 0., NOT_MODELED
    return float(np.average(values[valid], weights=weights[valid])), float(weights[valid].sum()), COMPLETE


def pairing_synergy(lineups: pd.DataFrame, primary_id: int, second_id: int) -> dict[str, Any]:
    work = lineups.copy()
    if not work.empty:
        work["PLAYER_IDS"] = work.apply(parse_lineup_ids, axis=1)
    if work.empty:
        return {"PAIR_NET_RATING": np.nan, "PRIMARY_WITHOUT_SECOND_NET_RATING": np.nan,
                "SECOND_WITHOUT_PRIMARY_NET_RATING": np.nan, "NEITHER_NET_RATING": np.nan,
                "PAIR_SYNERGY_DELTA": np.nan, "CONTEXTUAL_INTERACTION": np.nan,
                "PAIR_SAMPLE": 0., "PRIMARY_WITHOUT_SECOND_SAMPLE": 0.,
                "SECOND_WITHOUT_PRIMARY_SAMPLE": 0., "NEITHER_SAMPLE": 0.,
                "SYNERGY_STATUS": NOT_MODELED, "SYNERGY_INTERPRETATION": NOT_MODELED}
    has_primary = work.PLAYER_IDS.map(lambda ids: primary_id in ids)
    has_second = work.PLAYER_IDS.map(lambda ids: second_id in ids)
    states = {
        "PAIR": work[has_primary & has_second],
        "PRIMARY_WITHOUT_SECOND": work[has_primary & ~has_second],
        "SECOND_WITHOUT_PRIMARY": work[~has_primary & has_second],
        "NEITHER": work[~has_primary & ~has_second],
    }
    ratings = {name: weighted_rating(group, "NET_RATING") for name, group in states.items()}
    complete = all(result[2] == COMPLETE for result in ratings.values())
    pair, primary_solo, second_solo, neither = (
        ratings[name][0] for name in states)
    raw_delta = pair - primary_solo if ratings["PAIR"][2] == ratings[
        "PRIMARY_WITHOUT_SECOND"][2] == COMPLETE else np.nan
    interaction = ((pair - primary_solo) - (second_solo - neither)
                   if complete else np.nan)
    return {
        "PAIR_NET_RATING": pair,
        "PRIMARY_WITHOUT_SECOND_NET_RATING": primary_solo,
        "SECOND_WITHOUT_PRIMARY_NET_RATING": second_solo,
        "NEITHER_NET_RATING": neither,
        "PAIR_SYNERGY_DELTA": raw_delta,
        "CONTEXTUAL_INTERACTION": interaction,
        "PAIR_SAMPLE": ratings["PAIR"][1],
        "PRIMARY_WITHOUT_SECOND_SAMPLE": ratings["PRIMARY_WITHOUT_SECOND"][1],
        "SECOND_WITHOUT_PRIMARY_SAMPLE": ratings["SECOND_WITHOUT_PRIMARY"][1],
        "NEITHER_SAMPLE": ratings["NEITHER"][1],
        "SYNERGY_STATUS": COMPLETE if complete else NOT_MODELED,
        "SYNERGY_INTERPRETATION": "CONTEXTUAL_NOT_CAUSAL" if complete else NOT_MODELED,
    }


def confidence_label(minutes: float) -> str:
    if pd.isna(minutes) or minutes <= 0:
        return NOT_MODELED
    return "HIGH" if minutes >= 500 else "MEDIUM" if minutes >= 300 else "LOW"


def _position_group(value: Any) -> str:
    """Collapse Basketball-Reference positions into stable comparison groups."""
    position = str(value).upper()
    if "C" in position or position.startswith("PF"):
        return "BIG"
    if position.startswith("PG") or position.startswith("SG"):
        return "GUARD"
    return "WING"


def match_bpm(runs: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    out = runs.copy()
    out[["BPM_STATUS", "BPM_MATCH_METHOD"]] = [NOT_MODELED, "NONE"]
    for column in ["BPM", "OBPM", "DBPM", "PER", "WS_PER_48", "OWS", "DWS",
                   "WS", "VORP", "DRB_PCT", "TOV_PCT",
                   "BBR_BPM_PERCENTILE", "BBR_OBPM_PERCENTILE",
                   "BBR_DBPM_PERCENTILE", "BBR_VORP_PERCENTILE",
                   "BBR_WS_PERCENTILE", "BBR_DWS_PERCENTILE",
                   "POSITION_3PAR_PERCENTILE",
                   "POSITION_TS_PERCENTILE"]:
        out[column] = np.nan
    out["POSITION"] = NOT_MODELED
    out["POSITION_GROUP"] = NOT_MODELED
    if source.empty or not {"Player", "BPM"}.issubset(source):
        return out
    bbr = source.copy()
    bbr["_NAME"] = bbr.Player.map(normalized_name)
    numeric_source = {
        "G": "_G", "MP": "_MP", "BPM": "_BPM", "OBPM": "_OBPM",
        "DBPM": "_DBPM", "PER": "_PER", "WS/48": "_WS48",
        "OWS": "_OWS", "DWS": "_DWS", "WS": "_WS", "VORP": "_VORP",
        "DRB%": "_DRB", "TOV%": "_TOV",
        "3PAr": "_3PAR", "TS%": "_TS",
    }
    for original, target in numeric_source.items():
        bbr[target] = pd.to_numeric(bbr.get(original), errors="coerce")
    bbr["_POSITION"] = bbr.get("Pos", pd.Series(NOT_MODELED, index=bbr.index))
    bbr["_POSITION_GROUP"] = bbr._POSITION.map(_position_group)
    rotation = bbr[(bbr._G >= 8) & ((bbr._MP / bbr._G) >= 15)].copy()
    rotation["_BPM_PCTL"] = percentile(rotation._BPM)
    rotation["_OBPM_PCTL"] = percentile(rotation._OBPM)
    rotation["_DBPM_PCTL"] = percentile(rotation._DBPM)
    rotation["_VORP_PCTL"] = percentile(rotation._VORP)
    rotation["_WS_PCTL"] = percentile(rotation._WS)
    rotation["_DWS_PCTL"] = percentile(rotation._DWS)
    rotation["_POSITION_3PAR_PCTL"] = rotation.groupby(
        "_POSITION_GROUP")["_3PAR"].rank(pct=True) * 100
    rotation["_POSITION_TS_PCTL"] = rotation.groupby(
        "_POSITION_GROUP")["_TS"].rank(pct=True) * 100
    team_col = "Tm" if "Tm" in bbr else "Team" if "Team" in bbr else None
    for index, row in out.iterrows():
        candidates = bbr[bbr._NAME == normalized_name(row.PLAYER_NAME)]
        method = "NORMALIZED_NAME"
        if team_col and len(candidates) > 1:
            matched = candidates[candidates[team_col].astype(str) == str(row.TEAM_ABBREVIATION)]
            if not matched.empty:
                candidates, method = matched, "NAME_AND_TEAM"
        if len(candidates) == 1:
            selected = candidates.iloc[0]
            value = pd.to_numeric(selected.BPM, errors="coerce")
            if pd.notna(value):
                out.loc[index, ["BPM", "BPM_STATUS", "BPM_MATCH_METHOD"]] = [value, COMPLETE, method]
                qualified = rotation[rotation.index == selected.name]
                q = qualified.iloc[0] if not qualified.empty else selected
                values = {
                    "OBPM": q.get("_OBPM"), "DBPM": q.get("_DBPM"),
                    "PER": q.get("_PER"), "WS_PER_48": q.get("_WS48"),
                    "OWS": q.get("_OWS"), "DWS": q.get("_DWS"),
                    "WS": q.get("_WS"), "VORP": q.get("_VORP"),
                    "DRB_PCT": q.get("_DRB"), "TOV_PCT": q.get("_TOV"),
                    "BBR_BPM_PERCENTILE": q.get("_BPM_PCTL"),
                    "BBR_OBPM_PERCENTILE": q.get("_OBPM_PCTL"),
                    "BBR_DBPM_PERCENTILE": q.get("_DBPM_PCTL"),
                    "BBR_VORP_PERCENTILE": q.get("_VORP_PCTL"),
                    "BBR_WS_PERCENTILE": q.get("_WS_PCTL"),
                    "BBR_DWS_PERCENTILE": q.get("_DWS_PCTL"),
                    "POSITION_3PAR_PERCENTILE": q.get("_POSITION_3PAR_PCTL"),
                    "POSITION_TS_PERCENTILE": q.get("_POSITION_TS_PCTL"),
                    "POSITION": q.get("_POSITION", NOT_MODELED),
                    "POSITION_GROUP": q.get("_POSITION_GROUP", NOT_MODELED),
                }
                for column, matched_value in values.items():
                    out.loc[index, column] = matched_value
    return out


def deepest_round_availability(team_games: pd.DataFrame, player_games: pd.DataFrame,
                               runs: pd.DataFrame) -> pd.DataFrame:
    """Measure whether the selected player was present in the terminal series.

    Team qualification and player value are separate questions. A player who
    missed most of the Finals/Conference Finals does not represent the team's
    complete deep run, even if his earlier-round aggregate clears 8 GP.
    """
    records: list[dict[str, Any]] = []
    required_team = {"TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP"}
    required_player = {"PLAYER_ID", "GAME_ID", "MIN"}
    modeled = required_team.issubset(team_games) and required_player.issubset(player_games)
    for _, run in runs.iterrows():
        record = {
            "DEEPEST_ROUND_TEAM_GAMES": np.nan,
            "DEEPEST_ROUND_GAMES_PLAYED": np.nan,
            "DEEPEST_ROUND_AVAILABILITY": np.nan,
            "DEEPEST_ROUND_MPG": np.nan,
            "DEEPEST_ROUND_PPG": np.nan,
            "DEEPEST_ROUND_TS_PCT": np.nan,
            "DEEPEST_ROUND_AST_PG": np.nan,
            "DEEPEST_ROUND_REB_PG": np.nan,
            "DEEPEST_ROUND_TEAM_PTS_SHARE": np.nan,
            "DEEPEST_ROUND_TEAM_FGA_SHARE": np.nan,
            "PRIMARY_DEEPEST_ROUND_GAMES_PLAYED": np.nan,
            "PRIMARY_DEEPEST_ROUND_AVAILABILITY": np.nan,
            "PRIMARY_DEEPEST_ROUND_MPG": np.nan,
            "DEEPEST_ROUND_STATUS": NOT_MODELED,
            "RANKING_ELIGIBLE": True,
            "RANKING_ELIGIBILITY_REASON": "Availability not modeled",
        }
        if not modeled:
            records.append(record)
            continue
        team = team_games[pd.to_numeric(team_games.TEAM_ID, errors="coerce") == int(run.TEAM_ID)].copy()
        if team.empty:
            records.append(record)
            continue
        team["_DATE"] = pd.to_datetime(team.GAME_DATE, errors="coerce")
        latest = team.sort_values("_DATE").iloc[-1]
        opponent = str(latest.MATCHUP).split()[-1]
        terminal = team[team.MATCHUP.astype(str).str.endswith(opponent)]
        game_ids = set(terminal.GAME_ID.astype(str))
        games = len(game_ids)
        def participation(player_id: int) -> dict[str, float]:
            appearances = player_games[
                (pd.to_numeric(player_games.PLAYER_ID, errors="coerce") == player_id)
                & player_games.GAME_ID.astype(str).isin(game_ids)
            ]
            minutes = pd.to_numeric(appearances.MIN, errors="coerce")
            played = int(minutes.notna().sum())
            totals = {
                stat: pd.to_numeric(appearances[stat], errors="coerce").sum()
                if stat in appearances else np.nan
                for stat in ["PTS", "FGA", "FTA", "AST", "REB"]
            }
            tsa = totals["FGA"] + .44 * totals["FTA"]
            return {
                "played": played,
                "share": played / games if games else np.nan,
                "mpg": float(minutes.mean()) if played else 0.0,
                "ppg": totals["PTS"] / played if played else 0.0,
                "ts": totals["PTS"] / (2 * tsa) if tsa and pd.notna(tsa) else np.nan,
                "ast_pg": totals["AST"] / played if played else 0.0,
                "reb_pg": totals["REB"] / played if played else 0.0,
                "points": totals["PTS"],
                "fga": totals["FGA"],
            }

        second_stats = participation(int(run.PLAYER_ID))
        primary_stats = participation(int(run.PRIMARY_PLAYER_ID))
        played, share, mpg = (second_stats["played"], second_stats["share"],
                              second_stats["mpg"])
        primary_played, primary_share, primary_mpg = (
            primary_stats["played"], primary_stats["share"], primary_stats["mpg"])
        eligible = bool(
            games and share >= .5 and mpg >= 15
            and primary_share >= .5 and primary_mpg >= 15)
        record.update({
            "DEEPEST_ROUND_TEAM_GAMES": games,
            "DEEPEST_ROUND_GAMES_PLAYED": played,
            "DEEPEST_ROUND_AVAILABILITY": share,
            "DEEPEST_ROUND_MPG": mpg,
            "DEEPEST_ROUND_PPG": second_stats["ppg"],
            "DEEPEST_ROUND_TS_PCT": second_stats["ts"],
            "DEEPEST_ROUND_AST_PG": second_stats["ast_pg"],
            "DEEPEST_ROUND_REB_PG": second_stats["reb_pg"],
            "DEEPEST_ROUND_TEAM_PTS_SHARE": (
                second_stats["points"] / pd.to_numeric(
                    terminal.PTS, errors="coerce").sum()
                if "PTS" in terminal and pd.to_numeric(
                    terminal.PTS, errors="coerce").sum() else np.nan),
            "DEEPEST_ROUND_TEAM_FGA_SHARE": (
                second_stats["fga"] / pd.to_numeric(
                    terminal.FGA, errors="coerce").sum()
                if "FGA" in terminal and pd.to_numeric(
                    terminal.FGA, errors="coerce").sum() else np.nan),
            "PRIMARY_DEEPEST_ROUND_GAMES_PLAYED": primary_played,
            "PRIMARY_DEEPEST_ROUND_AVAILABILITY": primary_share,
            "PRIMARY_DEEPEST_ROUND_MPG": primary_mpg,
            "DEEPEST_ROUND_STATUS": COMPLETE,
            "RANKING_ELIGIBLE": eligible,
            "RANKING_ELIGIBILITY_REASON": (
                "Both stars played at least half of terminal-series games at 15+ MPG"
                if eligible else "Incomplete #1/#2 pairing in terminal series"),
        })
        records.append(record)
    return pd.DataFrame(records)


def attach_optional_metrics(runs: pd.DataFrame, isolation: pd.DataFrame,
                            catch: pd.DataFrame, rim: pd.DataFrame,
                            late: pd.DataFrame) -> pd.DataFrame:
    out = runs.copy()
    specs = {
        "ISO_POSS_PCT": (isolation, ["PLAYER_ID"], ["POSS_PCT"]),
        "ISO_PPP": (isolation, ["PLAYER_ID"], ["PPP"]),
        "CATCH_SHOOT_FG3A": (catch, ["PLAYER_ID"], ["CATCH_SHOOT_FG3A", "FG3A"]),
        "CATCH_SHOOT_FG3_PCT": (catch, ["PLAYER_ID"], ["CATCH_SHOOT_FG3_PCT", "FG3_PCT"]),
        "RIM_DFG_PCT": (rim, ["CLOSE_DEF_PERSON_ID", "PLAYER_ID"],
                        ["D_FG_PCT", "LT_06_PCT"]),
        "RIM_DFG_DIFF": (rim, ["CLOSE_DEF_PERSON_ID", "PLAYER_ID"],
                         ["PCT_PLUSMINUS", "PLUSMINUS"]),
        "LATE_CLOCK_FGA": (late, ["PLAYER_ID"], ["FGA"]),
        "LATE_CLOCK_PTS": (late, ["PLAYER_ID"], ["PTS"])}
    for target, (frame, ids, values) in specs.items():
        out[target] = np.nan
        id_col = next((c for c in ids if c in frame), None)
        value_col = next((c for c in values if c in frame), None)
        if id_col and value_col:
            lookup = frame.drop_duplicates(id_col).set_index(id_col)[value_col]
            out[target] = out.PLAYER_ID.map(lookup)
    modeled = out[["CATCH_SHOOT_FG3A", "RIM_DFG_PCT"]].notna().sum(axis=1)
    out["TRACKING_STATUS"] = np.select([modeled == 2, modeled > 0],
                                       [COMPLETE, PARTIAL], default=NOT_MODELED)
    return out


def percentile(series: pd.Series, higher: bool = True) -> pd.Series:
    score = pd.to_numeric(series, errors="coerce").rank(pct=True) * 100
    return score if higher else 100 - score


def opponent_srs_context(
    team_games: pd.DataFrame,
    team_srs: pd.DataFrame,
    runs: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the regular-season SRS of each run's playoff opponents.

    Games are deduplicated to the focal team's rows. Later series receive a
    modest 15% per-round leverage increase, reflecting deeper-round pressure
    without allowing schedule context to overwhelm player performance.
    """
    columns = [
        "OPPONENT_SRS_WEIGHTED", "OPPONENT_SRS_MAX", "OPPONENT_SERIES_COUNT",
        "OPPONENT_SRS_STATUS",
    ]
    if team_games.empty or team_srs.empty:
        return pd.DataFrame({column: [np.nan] * len(runs) for column in columns})
    require_columns(team_games, ["TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP"],
                    "opponent SRS team games")
    require_columns(team_srs, ["TEAM_ABBREVIATION", "SRS"], "team SRS")
    srs = team_srs.drop_duplicates("TEAM_ABBREVIATION").set_index(
        "TEAM_ABBREVIATION").SRS
    records: list[dict[str, object]] = []
    for _, run in runs.iterrows():
        games = team_games[team_games.TEAM_ID.eq(run.TEAM_ID)].copy()
        games = games.drop_duplicates("GAME_ID")
        games["OPPONENT"] = games.MATCHUP.astype(str).str.extract(
            r"(?:vs\.|@)\s*([A-Z]{2,3})")
        games["OPPONENT_SRS"] = pd.to_numeric(games.OPPONENT.map(srs), errors="coerce")
        series = (games.dropna(subset=["OPPONENT_SRS"])
                  .groupby("OPPONENT", as_index=False)
                  .agg(GAMES=("GAME_ID", "nunique"),
                       FIRST_GAME=("GAME_DATE", "min"),
                       OPPONENT_SRS=("OPPONENT_SRS", "first"))
                  .sort_values("FIRST_GAME"))
        if series.empty:
            records.append({column: np.nan for column in columns})
            continue
        series["ROUND_ORDER"] = np.arange(1, len(series) + 1)
        series["WEIGHT"] = series.GAMES * (1 + .15 * (series.ROUND_ORDER - 1))
        records.append({
            "OPPONENT_SRS_WEIGHTED": np.average(
                series.OPPONENT_SRS, weights=series.WEIGHT),
            "OPPONENT_SRS_MAX": series.OPPONENT_SRS.max(),
            "OPPONENT_SERIES_COUNT": len(series),
            "OPPONENT_SRS_STATUS": COMPLETE,
        })
    return pd.DataFrame(records, index=runs.index).reset_index(drop=True)


def role_adjusted_era_percentile(
    frame: pd.DataFrame,
    column: str,
    *,
    group: str = "POSITION_GROUP",
    radius: int = 1,
    minimum_reference: int = 4,
) -> pd.Series:
    """Compare a skill with nearby-era players who share its broad role.

    Small role/era cells fall back to the all-role era percentile instead of
    producing unstable one- or two-player rankings.
    """
    fallback = rolling_era_percentile(frame, column, radius=radius)
    if column not in frame or group not in frame or "SEASON" not in frame:
        return fallback
    years = pd.to_numeric(frame.SEASON.astype(str).str[:4], errors="coerce") + 1
    values = pd.to_numeric(frame[column], errors="coerce")
    groups = frame[group].astype(str)
    result = fallback.copy()
    for index in frame.index:
        value, year, role = values.loc[index], years.loc[index], groups.loc[index]
        if pd.isna(value) or pd.isna(year) or role == NOT_MODELED:
            continue
        nearby = values[((years - year).abs() <= radius) & groups.eq(role)].dropna()
        current = values[years.eq(year) & groups.eq(role)].dropna()
        reference = pd.concat([nearby, current], ignore_index=True)
        if len(reference) < minimum_reference:
            continue
        below = (reference < value).sum()
        tied = (reference == value).sum()
        result.loc[index] = (below + .5 * tied) / len(reference) * 100
    return result


def score_archetypes(frame: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    out = frame.copy()
    out["FG3A_PER_36"] = out.FG3A / out.MIN * 36
    out["BLK_PER_36"] = pd.to_numeric(out.get("BLK", np.nan), errors="coerce") / out.MIN * 36
    out["CATCH_SHOOT_FG3A_PER_36"] = (
        pd.to_numeric(out.get("CATCH_SHOOT_FG3A", np.nan), errors="coerce") / out.MIN * 36)
    out["LATE_CLOCK_FGA_PER_36"] = (
        pd.to_numeric(out.get("LATE_CLOCK_FGA", np.nan), errors="coerce") / out.MIN * 36)
    out["FG2A_PER_36"] = (out.FGA - out.FG3A) / out.MIN * 36
    out["FG2_PCT"] = ((out.FGM - out.FG3M) / (out.FGA - out.FG3A).replace(0, np.nan))
    out["FTA_PER_36"] = out.FTA / out.MIN * 36
    creation_components = pd.DataFrame({
        "usage_load": out.get(
            "ERA_USG_PERCENTILE", rolling_era_percentile(out, "USG_PCT")),
        "playmaking_load": out.get(
            "ERA_AST_PERCENTILE", rolling_era_percentile(out, "AST_PCT")),
        "decision_quality": rolling_era_percentile(out, "AST_TO_RATIO"),
        "isolation_volume": rolling_era_percentile(out, "ISO_POSS_PCT"),
        "isolation_efficiency": rolling_era_percentile(out, "ISO_PPP"),
        "late_clock_volume": rolling_era_percentile(out, "LATE_CLOCK_FGA_PER_36"),
    })
    off_ball_components = pd.DataFrame({
        "three_volume": pd.concat([
            out.get("ERA_FG3A75_PERCENTILE", rolling_era_percentile(out, "FG3A_PER_36")),
            pd.to_numeric(out.get("POSITION_3PAR_PERCENTILE", np.nan), errors="coerce"),
        ], axis=1).mean(axis=1),
        "three_accuracy": rolling_era_percentile(out, "EB_FG3_PCT"),
        "catch_shoot_volume": rolling_era_percentile(out, "CATCH_SHOOT_FG3A_PER_36"),
    })
    interior_components = pd.DataFrame({
        "two_point_volume": rolling_era_percentile(out, "FG2A_PER_36"),
        "two_point_efficiency": rolling_era_percentile(out, "FG2_PCT"),
        "free_throw_pressure": rolling_era_percentile(out, "FTA_PER_36"),
    })
    defensive_components = pd.DataFrame({
        "rim_deterrence": rolling_era_percentile(out, "RIM_DFG_DIFF", higher=False),
        "block_rate": rolling_era_percentile(out, "BLK_PER_36"),
        "box_defense": pd.to_numeric(out.get("BBR_DBPM_PERCENTILE", np.nan), errors="coerce"),
    })
    groups = {
        "CREATION_SCORE": creation_components,
        "OFF_BALL_SCORE": off_ball_components,
        "DEFENSIVE_SCORE": defensive_components,
    }
    for score, components in groups.items():
        count = components.notna().sum(axis=1)
        out[f"{score}_COVERAGE"] = count / len(components.columns)
        out[f"{score}_STATUS"] = np.select([count == len(components.columns), count > 0],
                                                   [COMPLETE, PARTIAL], default=NOT_MODELED)
        out[score] = components.mean(axis=1).where(count > 0)
        if strict:
            out.loc[count < len(components.columns), score] = np.nan
    out["DEFENSIVE_SCORE_RAW"] = out.DEFENSIVE_SCORE
    out["DEFENSIVE_SCORE"] = role_adjusted_era_percentile(
        out, "DEFENSIVE_SCORE_RAW"
    )
    out["DEFENSIVE_SCORE_METHOD"] = "POSITION_GROUP_NEARBY_ERA_PERCENTILE"
    out["INTERIOR_GRAVITY_SCORE"] = interior_components.mean(axis=1)
    out["INTERIOR_GRAVITY_COVERAGE"] = interior_components.notna().mean(axis=1)
    # Complementary offense can scale through perimeter gravity or through rim
    # pressure/finishing. Taking the stronger observed route avoids requiring a
    # big to create value in the same way as a guard.
    out["SCALABLE_OFFENSE_SCORE"] = pd.concat(
        [out.OFF_BALL_SCORE, out.INTERIOR_GRAVITY_SCORE], axis=1).max(axis=1)
    archetypes = ["CREATION_SCORE", "OFF_BALL_SCORE", "DEFENSIVE_SCORE"]
    out["DOMINANT_ARCHETYPE"] = out[archetypes].idxmax(axis=1).map({
        "CREATION_SCORE": "Pressure Valve", "OFF_BALL_SCORE": "Gravity Engine",
        "DEFENSIVE_SCORE": "Defensive Anchor"})
    out.loc[out[archetypes].notna().sum(axis=1) == 0, "DOMINANT_ARCHETYPE"] = NOT_MODELED
    out["PRIMARY_FG3A_PER_36"] = (
        pd.to_numeric(out.get("PRIMARY_FG3A", np.nan), errors="coerce")
        / pd.to_numeric(out.get("PRIMARY_MIN", np.nan), errors="coerce") * 36)
    out["PRIMARY_BLK_PER_36"] = (
        pd.to_numeric(out.get("PRIMARY_BLK", np.nan), errors="coerce")
        / pd.to_numeric(out.get("PRIMARY_MIN", np.nan), errors="coerce") * 36)
    out["PRIMARY_CATCH_SHOOT_FG3A_PER_36"] = (
        pd.to_numeric(out.get("PRIMARY_CATCH_SHOOT_FG3A", np.nan), errors="coerce")
        / pd.to_numeric(out.get("PRIMARY_MIN", np.nan), errors="coerce") * 36)

    primary_off_ball_components = pd.DataFrame({
        "three_volume": rolling_era_percentile(out, "PRIMARY_FG3A_PER_36"),
        "three_accuracy": rolling_era_percentile(out, "PRIMARY_EB_FG3_PCT"),
        "catch_shoot": rolling_era_percentile(
            out, "PRIMARY_CATCH_SHOOT_FG3A_PER_36"),
    })
    primary_defense_components = pd.DataFrame({
        "rim": rolling_era_percentile(out, "PRIMARY_RIM_DFG_DIFF", higher=False),
        "blocks": rolling_era_percentile(out, "PRIMARY_BLK_PER_36"),
    })
    primary_off_ball = primary_off_ball_components.mean(axis=1)
    primary_defense = primary_defense_components.mean(axis=1)
    primary_burden = (
        rolling_era_percentile(out, "PRIMARY_USG_PCT") * .60
        + rolling_era_percentile(out, "PRIMARY_AST_PCT") * .40)
    primary_creation = primary_burden
    out["PRIMARY_CREATION_CAPABILITY"] = primary_creation
    out["PRIMARY_OFF_BALL_CAPABILITY"] = primary_off_ball
    out["PRIMARY_DEFENSIVE_CAPABILITY"] = primary_defense
    out["PRIMARY_PRESSURE_VALVE_NEED"] = primary_burden
    out["PRIMARY_SPACING_NEED"] = 100 - primary_off_ball
    out["PRIMARY_DEFENSIVE_COVER_NEED"] = 100 - primary_defense
    out["PRESSURE_VALVE_FIT"] = out["CREATION_SCORE"] * primary_burden / 100
    out["GRAVITY_FIT"] = out["OFF_BALL_SCORE"] * out["PRIMARY_SPACING_NEED"] / 100
    out["DEFENSIVE_COVER_FIT"] = (
        out["DEFENSIVE_SCORE"] * out["PRIMARY_DEFENSIVE_COVER_NEED"] / 100)
    fit_components = out[["PRESSURE_VALVE_FIT", "GRAVITY_FIT", "DEFENSIVE_COVER_FIT"]]
    out["LEGACY_GAP_FILL_FIT"] = fit_components.mean(axis=1)
    needs = pd.DataFrame({
        "creation": out.PRIMARY_PRESSURE_VALVE_NEED,
        "gravity": out.PRIMARY_SPACING_NEED,
        "defense": out.PRIMARY_DEFENSIVE_COVER_NEED,
    })
    skills = pd.DataFrame({
        "creation": out.CREATION_SCORE,
        "gravity": out.SCALABLE_OFFENSE_SCORE,
        "defense": out.DEFENSIVE_SCORE,
    })
    available = needs.notna() & skills.notna()
    weighted_supply = (needs * skills).where(available).sum(axis=1, min_count=1)
    modeled_need = needs.where(available).sum(axis=1, min_count=1)
    out["NEED_NORMALIZED_FIT"] = weighted_supply / modeled_need.replace(0, np.nan)
    out["DEFENSIVE_NEED_SHARE"] = (
        needs.defense.where(available.defense) / modeled_need.replace(0, np.nan)
    )

    amplification = pd.DataFrame({
        "shared_creation": np.sqrt(out.CREATION_SCORE * primary_creation),
        "shared_gravity": np.sqrt(out.SCALABLE_OFFENSE_SCORE * primary_off_ball),
    })
    out["STRENGTH_AMPLIFICATION_SCORE"] = amplification.mean(axis=1)
    out["STRENGTH_AMPLIFICATION_COVERAGE"] = amplification.notna().sum(axis=1) / 2
    out["ROLE_COMPATIBILITY_SCORE"] = np.sqrt(
        out.NEED_NORMALIZED_FIT * out.STRENGTH_AMPLIFICATION_SCORE)

    secondary_coverage = out[[
        "CREATION_SCORE_COVERAGE", "OFF_BALL_SCORE_COVERAGE", "DEFENSIVE_SCORE_COVERAGE"]]
    primary_coverage = pd.DataFrame({
        "creation": pd.DataFrame({
            "usg": pd.to_numeric(out.PRIMARY_USG_PCT, errors="coerce"),
            "ast": pd.to_numeric(out.PRIMARY_AST_PCT, errors="coerce"),
        }).notna().mean(axis=1),
        "gravity": primary_off_ball_components.notna().mean(axis=1),
        "defense": primary_defense_components.notna().mean(axis=1),
    })
    out["COMPLEMENT_FIT_COVERAGE"] = pd.concat(
        [secondary_coverage, primary_coverage], axis=1).mean(axis=1)
    out["COMPLEMENT_FIT_SCORE"] = out.ROLE_COMPATIBILITY_SCORE
    out["COMPLEMENT_FIT_STATUS"] = np.select(
        [out.COMPLEMENT_FIT_COVERAGE == 1, out.COMPLEMENT_FIT_COVERAGE > 0],
        [COMPLETE, PARTIAL], default=NOT_MODELED)
    if "ERA_PTS75_PERCENTILE" not in out:
        out["ERA_PTS75_PERCENTILE"] = rolling_era_percentile(out, "PPG")
    if "ERA_TS_PERCENTILE" not in out:
        out["ERA_TS_PERCENTILE"] = rolling_era_percentile(out, "TS_PCT")
    out["ERA_BPM_PERCENTILE"] = pd.to_numeric(
        out.get("BBR_BPM_PERCENTILE", np.nan), errors="coerce")
    missing_bpm_reference = out.ERA_BPM_PERCENTILE.isna()
    out.loc[missing_bpm_reference, "ERA_BPM_PERCENTILE"] = rolling_era_percentile(
        out, "BPM")[missing_bpm_reference]
    production_components = pd.DataFrame({
        "BPM": out.ERA_BPM_PERCENTILE,
        "SCORING": out.ERA_PTS75_PERCENTILE,
        "EFFICIENCY": out.ERA_TS_PERCENTILE,
        "OFFENSIVE_BURDEN": out.ERA_USG_PERCENTILE,
    })
    # Equal-domain geometric mean: each observable domain must be strong, and no
    # fitted complementarity weight can overwhelm the underlying run.
    clipped = production_components.clip(lower=1e-6)
    out["IN_ERA_PRODUCTION_SCORE"] = np.exp(np.log(clipped).mean(axis=1))
    out["PRODUCTION_SCORE"] = out.IN_ERA_PRODUCTION_SCORE
    out["PRODUCTION_SCORE_STATUS"] = np.where(
        production_components.notna().all(axis=1), COMPLETE, NOT_MODELED)
    out["CONTEXTUAL_INTERACTION_SCORE"] = percentile(out["CONTEXTUAL_INTERACTION"])
    # Context is capped at 10% and omitted rather than imputed. The production
    # score remains the comparable all-era result.
    out["OVERALL_EVIDENCE_SCORE"] = out["PRODUCTION_SCORE"]
    contextual = out["CONTEXTUAL_INTERACTION_SCORE"].notna()
    out.loc[contextual, "OVERALL_EVIDENCE_SCORE"] = (
        out.loc[contextual, "PRODUCTION_SCORE"] * .90
        + out.loc[contextual, "CONTEXTUAL_INTERACTION_SCORE"] * .10)
    out["NET_IMPACT_SCORE"] = out["OVERALL_EVIDENCE_SCORE"]
    out["NET_IMPACT_SCORE_STATUS"] = out["PRODUCTION_SCORE_STATUS"]
    out["PROVISIONAL_EVIDENCE_SCORE"] = out.PRODUCTION_SCORE
    out["BEST_SECOND_OPTION_SCORE"] = out.PRODUCTION_SCORE.where(
        out.get("RANKING_ELIGIBLE", True))
    out["RANKING_METHOD"] = "OBSERVED_PRODUCTION_GEOMEAN_FIT_DESCRIPTIVE_ONLY"
    role_components = pd.DataFrame({
        "scoring_load": out.ERA_PTS75_PERCENTILE,
        "usage_load": out.ERA_USG_PERCENTILE,
        "minutes_load": out.get("ERA_MPG_PERCENTILE"),
        "team_scoring_share": rolling_era_percentile(out, "TEAM_PTS_SHARE"),
        "team_shot_share": rolling_era_percentile(out, "TEAM_FGA_SHARE"),
    }).clip(lower=1e-6)
    out["ROLE_BURDEN_SCORE"] = np.exp(np.log(role_components).mean(axis=1))
    cumulative_components = pd.DataFrame({
        "vorp": pd.to_numeric(out.get("BBR_VORP_PERCENTILE"), errors="coerce"),
        "win_shares": pd.to_numeric(out.get("BBR_WS_PERCENTILE"), errors="coerce"),
    }).clip(lower=1e-6)
    out["CUMULATIVE_IMPACT_SCORE"] = np.exp(
        np.log(cumulative_components).mean(axis=1))
    terminal_components = pd.DataFrame({
        "scoring": rolling_era_percentile(out, "DEEPEST_ROUND_PPG"),
        "efficiency": rolling_era_percentile(out, "DEEPEST_ROUND_TS_PCT"),
        "minutes": rolling_era_percentile(out, "DEEPEST_ROUND_MPG"),
        "team_scoring_share": rolling_era_percentile(
            out, "DEEPEST_ROUND_TEAM_PTS_SHARE"),
    }).clip(lower=1e-6)
    out["TERMINAL_RESPONSIBILITY_SCORE"] = np.exp(
        np.log(terminal_components).mean(axis=1))
    defense_components = pd.DataFrame({
        "dbpm": pd.to_numeric(out.get("BBR_DBPM_PERCENTILE"), errors="coerce"),
        "defensive_win_shares": pd.to_numeric(
            out.get("BBR_DWS_PERCENTILE"), errors="coerce"),
        "stocks": out.get("ERA_STOCKS75_PERCENTILE"),
    }).clip(lower=1e-6)
    out["HISTORICAL_DEFENSE_EVIDENCE_SCORE"] = np.exp(
        np.log(defense_components).mean(axis=1))
    holistic_components = pd.DataFrame({
        "rate_production": out.PRODUCTION_SCORE,
        "role_burden": out.ROLE_BURDEN_SCORE,
        "cumulative_impact": out.CUMULATIVE_IMPACT_SCORE,
        "terminal_responsibility": out.TERMINAL_RESPONSIBILITY_SCORE,
        "historical_defense": out.HISTORICAL_DEFENSE_EVIDENCE_SCORE,
    }).clip(lower=1e-6)
    out["BALANCED_SCORECARD_SCORE"] = np.exp(
        np.log(holistic_components).mean(axis=1))
    out["BALANCED_SCORECARD_COVERAGE"] = holistic_components.notna().mean(axis=1)
    out["ERA_ADJUSTMENT_STATUS"] = np.where(
        out.get("ERA_ENVIRONMENT_SAMPLE_PLAYERS", pd.Series(np.nan, index=out.index)).notna(),
        COMPLETE, PARTIAL)
    if strict:
        out.loc[out.COMPLEMENT_FIT_STATUS != COMPLETE, [
            "COMPLEMENT_FIT_SCORE", "ROLE_COMPATIBILITY_SCORE",
            "PROVISIONAL_EVIDENCE_SCORE", "BEST_SECOND_OPTION_SCORE"]] = np.nan
    return out
