from __future__ import annotations

import hashlib
import io
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from .config import PipelineConfig, SchemaError, SourceUnavailable, utc_now

LOG = logging.getLogger(__name__)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise SchemaError(f"{context}: missing required columns {missing}")


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [str(parts[-1]) for parts in result.columns]
    return result


class FrameCache:
    """Parquet responses plus adjacent, human-readable provenance metadata."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def paths(self, source: str, key: str) -> tuple[Path, Path]:
        folder = self.root / source
        folder.mkdir(parents=True, exist_ok=True)
        readable = re.sub(r"[^a-zA-Z0-9_.-]+", "_", key).strip("_")[:120]
        stem = f"{readable}-{hashlib.sha256(key.encode()).hexdigest()[:10]}"
        return folder / f"{stem}.parquet", folder / f"{stem}.json"

    def read(self, source: str, key: str) -> pd.DataFrame | None:
        frame_path, _ = self.paths(source, key)
        return pd.read_parquet(frame_path) if frame_path.exists() else None

    def write(self, source: str, key: str, frame: pd.DataFrame,
              metadata: Mapping[str, Any]) -> None:
        frame_path, meta_path = self.paths(source, key)
        frame.to_parquet(frame_path, index=False)
        meta_path.write_text(json.dumps(dict(metadata), indent=2, default=str))


def retry_call(call: Callable[[], Any], *, attempts: int, base_delay: float,
               label: str) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = base_delay * 2 ** (attempt - 1) + random.uniform(0, 0.4)
            LOG.warning("%s failed (%s/%s): %s; retrying in %.1fs",
                        label, attempt, attempts, exc, delay)
            time.sleep(delay)
    raise SourceUnavailable(f"{label} failed after {attempts} attempts: {last_error}")


class NBAStatsSource:
    """Typed adapter around the unstable public NBA Stats endpoints."""

    def __init__(self, config: PipelineConfig, cache: FrameCache):
        self.config, self.cache = config, cache

    def _frame(self, key: str, factory: Callable[[], Any], dataset: int = 0,
               retry: bool = True) -> pd.DataFrame:
        cached = None if self.config.refresh_cache else self.cache.read("nba_stats", key)
        if cached is not None:
            return cached
        if self.config.offline:
            raise SourceUnavailable(f"Offline cache miss: nba_stats/{key}")

        def request() -> pd.DataFrame:
            frames = factory().get_data_frames()
            if len(frames) <= dataset:
                raise SchemaError(f"{key}: expected data set {dataset}, got {len(frames)}")
            return frames[dataset]

        # Optional tracking endpoints often return a deterministic 400 for old
        # seasons; retrying those wastes time and cannot create missing history.
        frame = (retry_call(request, attempts=self.config.max_retries,
                            base_delay=self.config.request_delay, label=f"NBA Stats {key}")
                 if retry else request())
        self.cache.write("nba_stats", key, frame, {
            "source": "https://stats.nba.com", "retrieved_at": utc_now(),
            "rows": len(frame), "columns": list(frame.columns)})
        time.sleep(self.config.request_delay)
        return frame

    def playoff_games(self, season: str, po_round: int | None = None) -> pd.DataFrame:
        from nba_api.stats.endpoints import leaguegamefinder
        key = f"games_{season}_playoffs_round_{po_round or 'all'}"
        return self._frame(key, lambda: leaguegamefinder.LeagueGameFinder(
            player_or_team_abbreviation="T", season_nullable=season,
            season_type_nullable="Playoffs", po_round_nullable=str(po_round or ""),
            timeout=self.config.timeout))

    def playoff_player_games(self, season: str) -> pd.DataFrame:
        """Return one row per player appearance for the postseason.

        This is intentionally a league-wide request so deepest-round availability
        can be audited without making one request per player.
        """
        from nba_api.stats.endpoints import leaguegamefinder
        key = f"player_games_{season}_playoffs"
        return self._frame(key, lambda: leaguegamefinder.LeagueGameFinder(
            player_or_team_abbreviation="P", season_nullable=season,
            season_type_nullable="Playoffs", timeout=self.config.timeout))

    def player_stats(self, season: str, measure: str,
                     shot_clock: str = "") -> pd.DataFrame:
        from nba_api.stats.endpoints import leaguedashplayerstats
        return self._frame(f"players_{season}_{measure}_{shot_clock or 'all'}", lambda:
            leaguedashplayerstats.LeagueDashPlayerStats(
                season=season, season_type_all_star="Playoffs",
                measure_type_detailed_defense=measure, per_mode_detailed="Totals",
                shot_clock_range_nullable=shot_clock, timeout=self.config.timeout),
            retry=not bool(shot_clock))

    def lineups(self, season: str, team_id: int) -> pd.DataFrame:
        from nba_api.stats.endpoints import teamdashlineups
        return self._frame(f"lineups_v2_{season}_{team_id}_advanced_5", lambda:
            teamdashlineups.TeamDashLineups(
                team_id=team_id, group_quantity="5", season=season,
                season_type_all_star="Playoffs", measure_type_detailed_defense="Advanced",
                per_mode_detailed="Totals", timeout=self.config.timeout), dataset=1)

    def play_by_play(self, game_id: str) -> pd.DataFrame:
        """Return time-stamped actions from the current NBA play-by-play feed."""
        from nba_api.stats.endpoints import playbyplayv3

        key = f"play_by_play_v3_{game_id}"
        return self._frame(
            key,
            lambda: playbyplayv3.PlayByPlayV3(
                game_id=str(game_id), timeout=self.config.timeout
            ),
        )

    def game_rotation(self, game_id: str) -> pd.DataFrame:
        """Return both teams' official in/out intervals in one cached frame."""
        key = f"game_rotation_{game_id}"
        cached = None if self.config.refresh_cache else self.cache.read("nba_stats", key)
        if cached is not None:
            return cached
        if self.config.offline:
            raise SourceUnavailable(f"Offline cache miss: nba_stats/{key}")

        from nba_api.stats.endpoints import gamerotation

        def request() -> pd.DataFrame:
            frames = gamerotation.GameRotation(
                game_id=str(game_id), timeout=self.config.timeout
            ).get_data_frames()
            if len(frames) < 2:
                raise SchemaError(f"{key}: expected away and home rotation tables")
            away, home = frames[0].copy(), frames[1].copy()
            away["IS_HOME"] = False
            home["IS_HOME"] = True
            return pd.concat([away, home], ignore_index=True)

        frame = retry_call(
            request,
            attempts=self.config.max_retries,
            base_delay=self.config.request_delay,
            label=f"NBA Stats {key}",
        )
        require_columns(
            frame,
            [
                "GAME_ID",
                "TEAM_ID",
                "PERSON_ID",
                "IN_TIME_REAL",
                "OUT_TIME_REAL",
                "IS_HOME",
            ],
            key,
        )
        self.cache.write(
            "nba_stats",
            key,
            frame,
            {
                "source": "https://stats.nba.com",
                "endpoint": "gamerotation",
                "retrieved_at": utc_now(),
                "rows": len(frame),
                "columns": list(frame.columns),
            },
        )
        time.sleep(self.config.request_delay)
        return frame

    def isolation(self, season: str) -> pd.DataFrame:
        from nba_api.stats.endpoints import synergyplaytypes
        return self._frame(f"isolation_{season}", lambda: synergyplaytypes.SynergyPlayTypes(
            player_or_team_abbreviation="P", season=season,
            season_type_all_star="Playoffs", per_mode_simple="Totals",
            play_type_nullable="Isolation", timeout=self.config.timeout), retry=False)

    def catch_shoot(self, season: str) -> pd.DataFrame:
        from nba_api.stats.endpoints import leaguedashptstats
        return self._frame(f"catch_shoot_{season}", lambda: leaguedashptstats.LeagueDashPtStats(
            player_or_team="Player", pt_measure_type="CatchShoot", season=season,
            season_type_all_star="Playoffs", per_mode_simple="Totals",
            timeout=self.config.timeout), retry=False)

    def rim_defense(self, season: str) -> pd.DataFrame:
        from nba_api.stats.endpoints import leaguedashptdefend
        return self._frame(f"rim_defense_{season}", lambda:
            leaguedashptdefend.LeagueDashPtDefend(
                defense_category="Less Than 6Ft", season=season,
                season_type_all_star="Playoffs", per_mode_simple="Totals",
                timeout=self.config.timeout), retry=False)


class BasketballReferenceSource:
    """Canonical playoff BPM source; never substitutes NBA raw plus-minus."""

    def __init__(self, config: PipelineConfig, cache: FrameCache):
        self.config, self.cache = config, cache

    def playoff_advanced(self, season: str) -> pd.DataFrame:
        ending_year = int(season[:4]) + 1
        url = f"https://www.basketball-reference.com/playoffs/NBA_{ending_year}_advanced.html"
        key = f"playoffs_{ending_year}_advanced"
        cached = None if self.config.refresh_cache else self.cache.read(
            "basketball_reference", key)
        if cached is not None:
            return cached
        if self.config.offline:
            raise SourceUnavailable(f"Offline cache miss: basketball_reference/{key}")

        import requests
        from bs4 import BeautifulSoup, Comment

        def request_html() -> str:
            response = requests.get(url, timeout=self.config.timeout, headers={
                "User-Agent": "nba-second-options-research/1.0 (personal analytics project)"})
            response.raise_for_status()
            return response.text

        html = retry_call(request_html, attempts=self.config.max_retries,
                          base_delay=max(3.0, self.config.request_delay), label=url)
        soup = BeautifulSoup(html, "lxml")
        candidates = [html] + [str(c) for c in soup.find_all(
            string=lambda v: isinstance(v, Comment)) if "<table" in str(c)]
        tables: list[pd.DataFrame] = []
        for candidate in candidates:
            try:
                tables.extend(pd.read_html(io.StringIO(candidate)))
            except ValueError:
                pass
        table = next((flatten_columns(t) for t in tables
                      if {"Player", "BPM"}.issubset(flatten_columns(t).columns)), None)
        if table is None:
            raise SchemaError(f"No Player/BPM table found at {url}")
        table = table[table["Player"].astype(str) != "Player"].copy()
        self.cache.write("basketball_reference", key, table, {
            "source": url, "retrieved_at": utc_now(), "rows": len(table),
            "metric": "Basketball-Reference BPM 2.0"})
        time.sleep(max(3.0, self.config.request_delay))
        return table

    def regular_season_srs(self, season: str) -> pd.DataFrame:
        """Return team Simple Rating System values for the preceding regular season.

        SRS is used only as a bounded description of the opponents a team faced;
        it is not treated as player production or as a causal estimate of difficulty.
        """
        ending_year = int(season[:4]) + 1
        url = f"https://www.basketball-reference.com/leagues/NBA_{ending_year}.html"
        key = f"regular_season_{ending_year}_srs"
        cached = None if self.config.refresh_cache else self.cache.read(
            "basketball_reference", key)
        if cached is not None:
            return cached
        if self.config.offline:
            raise SourceUnavailable(f"Offline cache miss: basketball_reference/{key}")

        import requests
        from bs4 import BeautifulSoup

        def request_html() -> str:
            response = requests.get(url, timeout=self.config.timeout, headers={
                "User-Agent": "nba-second-options-research/1.0 (personal analytics project)"})
            response.raise_for_status()
            return response.text

        html = retry_call(request_html, attempts=self.config.max_retries,
                          base_delay=max(3.0, self.config.request_delay), label=url)
        # Basketball-Reference sometimes places standings tables inside comments.
        soup = BeautifulSoup(html.replace("<!--", "").replace("-->", ""), "lxml")
        records: list[dict[str, object]] = []
        for row in soup.find_all("tr"):
            team_cell = row.find(["th", "td"], attrs={"data-stat": "team_name"})
            srs_cell = row.find("td", attrs={"data-stat": "srs"})
            link = team_cell.find("a") if team_cell else None
            if not link or srs_cell is None:
                continue
            match = re.search(r"/teams/([A-Z]{2,3})/", str(link.get("href", "")))
            if not match:
                continue
            try:
                srs = float(srs_cell.get_text(strip=True))
            except ValueError:
                continue
            records.append({
                "TEAM_ABBREVIATION": match.group(1),
                "TEAM_NAME": team_cell.get_text(" ", strip=True).rstrip("*"),
                "SRS": srs,
            })
        table = pd.DataFrame(records).drop_duplicates("TEAM_ABBREVIATION")
        if len(table) < 20:
            raise SchemaError(f"No complete team SRS table found at {url}")
        self.cache.write("basketball_reference", key, table, {
            "source": url, "retrieved_at": utc_now(), "rows": len(table),
            "metric": "Basketball-Reference regular-season SRS"})
        time.sleep(max(3.0, self.config.request_delay))
        return table
