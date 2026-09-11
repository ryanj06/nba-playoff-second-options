from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

NOT_MODELED = "NOT_MODELED"
COMPLETE = "COMPLETE"
PARTIAL = "PARTIAL"
TRACKING_START_SEASON = 2013  # 2013-14; optical player tracking era


class PipelineError(RuntimeError):
    """Base error with a user-actionable message."""


class SourceUnavailable(PipelineError):
    """A source failed and no valid cache entry exists."""


class SchemaError(PipelineError):
    """An upstream response no longer satisfies its data contract."""


@dataclass(frozen=True)
class PipelineConfig:
    start_season: str = "1999-00"
    end_season: str | None = None
    min_games: int = 8
    min_mpg: float = 15.0
    timeout: int = 45
    max_retries: int = 5
    request_delay: float = 1.25
    cache_dir: Path = Path("data/cache")
    output_dir: Path = Path("outputs")
    offline: bool = False
    refresh_cache: bool = False
    strict_metrics: bool = False
    demo: bool = False

    def resolved_end_season(self) -> str:
        if self.end_season:
            return self.end_season
        now = datetime.now()
        ending_year = now.year if now.month >= 7 else now.year - 1
        return f"{ending_year - 1}-{str(ending_year)[-2:]}"


def seasons_between(start: str, end: str) -> list[str]:
    def first_year(value: str) -> int:
        if not re.fullmatch(r"\d{4}-\d{2}", value):
            raise ValueError(f"Invalid NBA season {value!r}; expected YYYY-YY")
        return int(value[:4])

    start_year, end_year = first_year(start), first_year(end)
    if start_year > end_year:
        raise ValueError("start_season must not be later than end_season")
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(start_year, end_year + 1)]


def has_tracking_coverage(season: str) -> bool:
    return int(season[:4]) >= TRACKING_START_SEASON


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
