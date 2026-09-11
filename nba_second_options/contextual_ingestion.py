"""Resumable ingestion for the game-level inputs required by CRV."""

from __future__ import annotations

import logging

import pandas as pd

from .config import PipelineConfig, PipelineError, seasons_between, utc_now
from .sources import FrameCache, NBAStatsSource, require_columns
from .stints import build_constant_lineup_stints

LOG = logging.getLogger(__name__)


def cache_contextual_games(config: PipelineConfig) -> pd.DataFrame:
    """Cache official play-by-play and rotation feeds for every playoff game.

    Existing cache entries are reused. Individual failures are recorded and do
    not discard successful games, which makes long historical pulls resumable.
    """
    cache = FrameCache(config.cache_dir)
    source = NBAStatsSource(config, cache)
    records: list[dict[str, object]] = []
    for season in seasons_between(config.start_season, config.resolved_end_season()):
        try:
            games = source.playoff_games(season)
            require_columns(games, ["GAME_ID"], f"{season} playoff games")
        except (PipelineError, ValueError, KeyError) as exc:
            LOG.error("Unable to enumerate %s playoff games: %s", season, exc)
            records.append({
                "SEASON": season,
                "GAME_ID": "",
                "STATUS": "SEASON_FAILED",
                "ERROR": str(exc),
            })
            continue

        game_ids = games.GAME_ID.astype(str).drop_duplicates().sort_values()
        for position, game_id in enumerate(game_ids, start=1):
            try:
                play_by_play = source.play_by_play(game_id)
                rotation = source.game_rotation(game_id)
                stint_key = f"constant_lineup_stints_{game_id}"
                stints = (
                    None
                    if config.refresh_cache
                    else cache.read("derived_contextual", stint_key)
                )
                if stints is None:
                    stints = build_constant_lineup_stints(rotation, play_by_play)
                    cache.write(
                        "derived_contextual",
                        stint_key,
                        stints,
                        {
                            "source": "derived from cached NBA play-by-play v3 and game rotation",
                            "retrieved_at": utc_now(),
                            "rows": len(stints),
                            "observation_status": "OBSERVED_NOT_CAUSAL",
                        },
                    )
                records.append({
                    "SEASON": season,
                    "GAME_ID": game_id,
                    "STATUS": "COMPLETE",
                    "PLAY_BY_PLAY_ROWS": len(play_by_play),
                    "ROTATION_ROWS": len(rotation),
                    "STINT_ROWS": len(stints),
                    "ERROR": "",
                })
                LOG.info(
                    "%s game %s/%s cached: %s",
                    season,
                    position,
                    len(game_ids),
                    game_id,
                )
            except (PipelineError, ValueError, KeyError) as exc:
                LOG.warning("%s %s unavailable: %s", season, game_id, exc)
                records.append({
                    "SEASON": season,
                    "GAME_ID": game_id,
                    "STATUS": "FAILED",
                    "PLAY_BY_PLAY_ROWS": 0,
                    "ROTATION_ROWS": 0,
                    "STINT_ROWS": 0,
                    "ERROR": str(exc),
                })

    manifest = pd.DataFrame(records)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(config.output_dir / "contextual_ingestion_manifest.csv", index=False)
    return manifest
