#!/usr/bin/env python3
"""CLI for the NBA playoff second-option analytics pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from nba_second_options import PipelineConfig
from nba_second_options.config import PipelineError
from nba_second_options.contextual_ingestion import cache_contextual_games
from nba_second_options.pipeline import run_pipeline
from nba_second_options.reporting import save_outputs


def parse_args(argv: Sequence[str] | None = None) -> tuple[PipelineConfig, bool, bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", default="1999-00")
    parser.add_argument("--end-season")
    parser.add_argument("--min-games", type=int, default=8)
    parser.add_argument("--min-mpg", type=float, default=15)
    parser.add_argument("--request-delay", type=float, default=1.25,
                        help="Minimum delay between successful source requests")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--strict-metrics", action="store_true",
                        help="Only score archetypes with complete component coverage")
    parser.add_argument("--demo", action="store_true",
                        help="Render synthetic smoke-test artifacts, never research results")
    parser.add_argument(
        "--cache-contextual-games",
        action="store_true",
        help="Cache playoff play-by-play and rotations, then write an ingestion manifest",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    config = PipelineConfig(
        start_season=args.start_season, end_season=args.end_season,
        min_games=args.min_games, min_mpg=args.min_mpg,
        request_delay=args.request_delay,
        cache_dir=args.cache_dir, output_dir=args.output_dir,
        offline=args.offline, refresh_cache=args.refresh_cache,
        strict_metrics=args.strict_metrics, demo=args.demo)
    return config, args.verbose, args.cache_contextual_games


def main(argv: Sequence[str] | None = None) -> int:
    config, verbose, cache_games = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if cache_games:
            manifest = cache_contextual_games(config)
            completed = int(manifest.STATUS.eq("COMPLETE").sum())
            failed = int((~manifest.STATUS.eq("COMPLETE")).sum())
            logging.info(
                "Contextual inputs cached for %s games; %s failures are documented in %s",
                completed,
                failed,
                (config.output_dir / "contextual_ingestion_manifest.csv").resolve(),
            )
            return 0 if completed else 2
        frame = run_pipeline(config)
        paths = save_outputs(frame, config)
        logging.info("Completed %s runs and wrote %s artifacts to %s",
                     len(frame), len(paths), config.output_dir.resolve())
        return 0
    except (PipelineError, OSError, ValueError) as exc:
        logging.error("Pipeline failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
