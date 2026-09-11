from __future__ import annotations

import json
import logging
from typing import Callable

import pandas as pd

from .config import (COMPLETE, PipelineConfig, PipelineError, SchemaError,
                     SourceUnavailable, has_tracking_coverage, seasons_between)
from .era import attach_season_environment
from .metrics import (attach_optional_metrics, beta_binomial_shrinkage, confidence_label,
                      conference_finalists, deepest_round_availability, match_bpm, merge_player_stats,
                      opponent_srs_context, pairing_synergy, score_archetypes,
                      select_options, true_shooting)
from .sources import BasketballReferenceSource, FrameCache, NBAStatsSource

LOG = logging.getLogger(__name__)


def optional(call: Callable[[], pd.DataFrame], label: str) -> pd.DataFrame:
    try:
        return call()
    except (SourceUnavailable, SchemaError, ValueError, KeyError) as exc:
        LOG.warning("%s unavailable; recording NOT_MODELED: %s", label, exc)
        return pd.DataFrame()


def process_season(season: str, nba: NBAStatsSource,
                   bbr: BasketballReferenceSource, config: PipelineConfig) -> pd.DataFrame:
    LOG.info("Processing %s", season)
    team_games = nba.playoff_games(season)
    qualifiers = conference_finalists(
        team_games,
        optional(lambda: nba.playoff_games(season, 3), f"{season} round 3"))
    players = merge_player_stats(nba.player_stats(season, "Base"),
                                 nba.player_stats(season, "Advanced"))
    runs = select_options(players, qualifiers, config, season)
    runs.insert(0, "SEASON", season)
    runs["PPG"] = runs.PTS / runs.GP
    runs["TS_PCT"] = true_shooting(runs.PTS, runs.FGA, runs.FTA)
    runs = attach_season_environment(players, runs, config)
    player_games = optional(lambda: nba.playoff_player_games(season),
                            f"{season} player game logs")
    availability = deepest_round_availability(team_games, player_games, runs)
    runs = pd.concat([runs.reset_index(drop=True), availability], axis=1)
    league_ts = float(players.PTS.sum() / (2 * (players.FGA.sum() + .44 * players.FTA.sum())))
    runs["PLAYOFF_LEAGUE_TS_PCT"] = league_ts
    runs["TS_PLUS"] = runs.TS_PCT / league_ts * 100
    posterior, alpha, beta = beta_binomial_shrinkage(players.FG3M, players.FG3A)
    lookup = pd.DataFrame({"PLAYER_ID": players.PLAYER_ID,
                           "POSTERIOR": posterior}).groupby("PLAYER_ID").POSTERIOR.mean()
    runs["EB_FG3_PCT"] = runs.PLAYER_ID.map(lookup)
    runs["EB_PRIOR_ALPHA"], runs["EB_PRIOR_BETA"] = alpha, beta
    runs["RUN_CONFIDENCE"] = runs.MIN.map(confidence_label)

    srs = optional(lambda: bbr.regular_season_srs(season), f"{season} opponent SRS")
    competition = opponent_srs_context(team_games, srs, runs)
    runs = pd.concat([runs.reset_index(drop=True), competition], axis=1)

    synergy = []
    for _, row in runs.iterrows():
        lineups = optional(lambda row=row: nba.lineups(season, int(row.TEAM_ID)),
                           f"{season} {row.TEAM_ABBREVIATION} lineups")
        synergy.append(pairing_synergy(lineups, int(row.PRIMARY_PLAYER_ID), int(row.PLAYER_ID)))
    runs = pd.concat([runs.reset_index(drop=True), pd.DataFrame(synergy)], axis=1)
    runs["SYNERGY_CONFIDENCE"] = runs[[
        "PAIR_SAMPLE", "PRIMARY_WITHOUT_SECOND_SAMPLE",
        "SECOND_WITHOUT_PRIMARY_SAMPLE", "NEITHER_SAMPLE",
    ]].min(axis=1).map(confidence_label)
    runs = match_bpm(runs, optional(lambda: bbr.playoff_advanced(season), f"{season} BPM"))

    tracking = has_tracking_coverage(season)
    isolation = (optional(lambda: nba.isolation(season), f"{season} isolation")
                 if tracking else pd.DataFrame())
    catch = (optional(lambda: nba.catch_shoot(season), f"{season} catch-and-shoot")
             if tracking else pd.DataFrame())
    rim = (optional(lambda: nba.rim_defense(season), f"{season} rim defense")
           if tracking else pd.DataFrame())
    late_parts = ([optional(lambda clock=clock: nba.player_stats(season, "Base", clock),
                            f"{season} {clock}")
                   for clock in ("7-4 Late", "4-0 Very Late")] if tracking else [])
    available = [part for part in late_parts if not part.empty]
    late = pd.concat(available, ignore_index=True) if available else pd.DataFrame()
    if not late.empty:
        columns = [c for c in ["FGA", "PTS"] if c in late]
        late = late.groupby("PLAYER_ID", as_index=False)[columns].sum()
    runs = attach_optional_metrics(runs, isolation, catch, rim, late)

    primary = players[players.PLAYER_ID.isin(runs.PRIMARY_PLAYER_ID)].copy()
    primary["EB_FG3_PCT"] = primary.PLAYER_ID.map(lookup)
    primary = attach_optional_metrics(primary, isolation, catch, rim, late)
    primary_columns = [
        "PLAYER_ID", "TEAM_ID", "MIN", "FG3A", "BLK", "EB_FG3_PCT",
        "ISO_POSS_PCT", "ISO_PPP", "CATCH_SHOOT_FG3A",
        "RIM_DFG_DIFF", "LATE_CLOCK_FGA",
    ]
    primary = primary[[column for column in primary_columns if column in primary]].rename(
        columns={
            column: f"PRIMARY_{column}"
            for column in primary_columns
            if column not in {"PLAYER_ID", "TEAM_ID"}
        } | {"PLAYER_ID": "PRIMARY_PLAYER_ID"}
    )
    return runs.merge(primary, on=["PRIMARY_PLAYER_ID", "TEAM_ID"], how="left",
                      validate="one_to_one")


def demo_data() -> pd.DataFrame:
    """Synthetic smoke-test data; values are not research results."""
    examples = [
        ("2000-01", "LAL", "Kobe Bryant", "Shaquille O'Neal", 29.4, .555, 8.2, 6.5),
        ("2007-08", "BOS", "Paul Pierce", "Kevin Garnett", 19.7, .570, 5.1, 4.0),
        ("2015-16", "CLE", "Kyrie Irving", "LeBron James", 25.2, .574, 9.0, 4.8),
        ("2019-20", "LAL", "Anthony Davis", "LeBron James", 27.7, .665, 7.5, 8.7),
        ("2022-23", "DEN", "Jamal Murray", "Nikola Jokic", 26.1, .586, 10.8, 5.2),
        ("2023-24", "BOS", "Jaylen Brown", "Jayson Tatum", 23.9, .535, 6.2, 3.0)]
    rows = []
    for i, (season, team, player, primary, ppg, ts, delta, bpm) in enumerate(examples):
        rows.append({"SEASON": season, "TEAM_ABBREVIATION": team, "PLAYER_ID": 1000 + i,
            "PLAYER_NAME": player, "PRIMARY_PLAYER_NAME": primary, "PPG": ppg, "TS_PCT": ts,
            "PAIR_SYNERGY_DELTA": delta, "BPM": bpm, "MIN": 550 - i * 20,
            "RUN_CONFIDENCE": "HIGH", "SYNERGY_CONFIDENCE": "MEDIUM",
            "POSTSEASON_FINISH": "Champion", "CREATION_SCORE": 58 + i * 6,
            "OFF_BALL_SCORE": 48 + i * 7,
            "DEFENSIVE_SCORE": 72 if player == "Anthony Davis" else 35 + i * 3,
            "NET_IMPACT_SCORE": 60 + i * 5, "TRACKING_STATUS": COMPLETE,
            "BPM_STATUS": COMPLETE, "SYNERGY_STATUS": COMPLETE,
            "CREATION_SCORE_STATUS": COMPLETE, "OFF_BALL_SCORE_STATUS": COMPLETE,
            "DEFENSIVE_SCORE_STATUS": COMPLETE, "NET_IMPACT_SCORE_STATUS": COMPLETE})
    result = pd.DataFrame(rows)
    result["DOMINANT_ARCHETYPE"] = result[
        ["CREATION_SCORE", "OFF_BALL_SCORE", "DEFENSIVE_SCORE"]].idxmax(axis=1).map({
        "CREATION_SCORE": "Pressure Valve", "OFF_BALL_SCORE": "Gravity Engine",
        "DEFENSIVE_SCORE": "Defensive Anchor"})
    # Keep the smoke test independent of network/cache state while exercising
    # every reporting path with visibly synthetic, internally complete inputs.
    scale = pd.Series(range(len(result)), index=result.index, dtype=float)
    result["TEAM_ID"] = 1_610_610_000 + scale.astype(int)
    result["RANKING_ELIGIBLE"] = True
    result["PTS_PER_75"] = result.PPG * 1.02
    result["RELATIVE_TS_PCT"] = result.TS_PCT - .56
    result["ERA_PTS75_PERCENTILE"] = 68 + scale * 5
    result["ERA_TS_PERCENTILE"] = 64 + scale * 5
    result["ERA_BPM_PERCENTILE"] = 70 + scale * 4
    result["ERA_USG_PERCENTILE"] = 72 + scale * 3
    result["TEAM_PTS_SHARE"] = .19 + scale * .012
    result["TEAM_AST_SHARE"] = .14 + scale * .013
    result["TEAM_REB_SHARE"] = .08 + scale * .009
    result["TEAM_BLK_SHARE"] = .08 + scale * .012
    result["POSITION_GROUP"] = "GUARD"
    result.loc[result.PLAYER_NAME.eq("Anthony Davis"), "POSITION_GROUP"] = "BIG"
    result["DEEPEST_ROUND_MPG"] = 34 + scale
    result["NEED_NORMALIZED_FIT"] = 55 + scale * 5
    result["STRENGTH_AMPLIFICATION_SCORE"] = 58 + scale * 4
    result["ROLE_COMPATIBILITY_SCORE"] = (
        result.NEED_NORMALIZED_FIT * result.STRENGTH_AMPLIFICATION_SCORE
    ) ** .5
    result["COMPLEMENT_FIT_COVERAGE"] = 1.0
    result["IN_ERA_PRODUCTION_SCORE"] = 66 + scale * 5
    result["PRODUCTION_SCORE"] = result.IN_ERA_PRODUCTION_SCORE
    result["BEST_SECOND_OPTION_SCORE"] = result.PRODUCTION_SCORE
    result["ROLE_BURDEN_SCORE"] = 65 + scale * 4
    result["BBR_VORP_PERCENTILE"] = 70 + scale * 4
    result["BBR_WS_PERCENTILE"] = 72 + scale * 4
    result["CUMULATIVE_IMPACT_SCORE"] = (
        result.BBR_VORP_PERCENTILE * result.BBR_WS_PERCENTILE
    ) ** .5
    result["TERMINAL_RESPONSIBILITY_SCORE"] = 68 + scale * 4
    result["HISTORICAL_DEFENSE_EVIDENCE_SCORE"] = 55 + scale * 4
    result["OPPONENT_SRS_WEIGHTED"] = 1 + scale * .4
    result["OPPONENT_SRS_MAX"] = 3 + scale * .5
    return result


def run_pipeline(config: PipelineConfig) -> pd.DataFrame:
    if config.demo:
        return demo_data()
    cache = FrameCache(config.cache_dir)
    nba, bbr = NBAStatsSource(config, cache), BasketballReferenceSource(config, cache)
    results, failures = [], []
    for season in seasons_between(config.start_season, config.resolved_end_season()):
        try:
            results.append(process_season(season, nba, bbr, config))
        except (PipelineError, ValueError, KeyError) as exc:
            LOG.error("Skipping %s: %s", season, exc)
            failures.append({"season": season, "error": str(exc)})
    if failures:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "season_failures.json").write_text(json.dumps(failures, indent=2))
    if not results:
        raise PipelineError("No seasons completed; inspect logs/cache or validate with --demo")
    return score_archetypes(pd.concat(results, ignore_index=True), config.strict_metrics)
