"""Download and document official NBA player portraits used by the visuals.

The NBA archive contains season/team-specific portraits for many recent seasons.
For older runs, this module can retrieve official NBA Player File portraits from
the nearest archived snapshot. A non-season portrait is used only as an explicitly
labelled last resort; it is never presented as season-specific.
"""

from __future__ import annotations

import argparse
import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

LOG = logging.getLogger(__name__)

# Official NBA Player File images preserved by the Internet Archive. Each
# timestamp falls inside the listed NBA season, so these are exact-season—not
# modern "latest"—portraits. The small original dimensions reflect the NBA's
# site design at the time.
ARCHIVED_SEASON_PORTRAIT_URLS: dict[tuple[str, int], str] = {
    ("2000-01", 977): (
        "https://web.archive.org/web/20001003031416id_/"
        "http://www.nba.com:80/playerfile/images/kobe_bryant.jpg"
    ),
    ("2003-04", 406): (
        "https://web.archive.org/web/20031231000114id_/"
        "http://www.nba.com:80/playerfile/images/shaquille_oneal.jpg"
    ),
    ("2004-05", 1938): (
        "https://web.archive.org/web/20040921143432id_/"
        "http://www.nba.com:80/media/playerfile/emanuel_ginobili.jpg"
    ),
    ("2009-10", 2200): (
        "https://web.archive.org/web/20100219021936id_/"
        "http://www.nba.com/media/playerfile/pau_gasol.jpg"
    ),
    ("2010-11", 2548): (
        "https://web.archive.org/web/20101117011155id_/"
        "http://www.nba.com/media/playerfile/dwyane_wade.jpg"
    ),
}


def season_portrait_url(row: pd.Series) -> str:
    """Return the official NBA season/team archive URL for a run."""
    year = str(row["SEASON"]).split("-", maxsplit=1)[0]
    return (
        "https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/"
        f"{int(row['TEAM_ID'])}/{year}/260x190/{int(row['PLAYER_ID'])}.png"
    )


def fallback_portrait_url(row: pd.Series) -> str:
    """Return the official NBA general portrait URL for a player."""
    return (
        "https://cdn.nba.com/headshots/nba/latest/260x190/"
        f"{int(row['PLAYER_ID'])}.png"
    )


def portrait_path(directory: Path, season: str, player_id: int) -> Path:
    return directory / f"{season.replace('-', '_')}_{int(player_id)}.png"


def _valid_image(response: requests.Response) -> bool:
    if response.status_code != 200:
        return False
    if not response.headers.get("content-type", "").lower().startswith("image/"):
        return False
    try:
        with Image.open(io.BytesIO(response.content)) as image:
            image.verify()
    except (OSError, ValueError):
        return False
    return True


def _fetch(session: requests.Session, url: str, attempts: int = 3) -> bytes | None:
    for attempt in range(attempts):
        try:
            response = session.get(url, timeout=20)
            if _valid_image(response):
                return response.content
            # A missing/forbidden archive key is definitive; retry only transient errors.
            if response.status_code not in {429, 500, 502, 503, 504}:
                return None
        except requests.RequestException as exc:
            LOG.warning("Portrait request failed (%s): %s", url, exc)
        if attempt + 1 < attempts:
            time.sleep(1.5 * (attempt + 1))
    return None


def _png_bytes(content: bytes) -> bytes:
    """Normalize archived JPEGs and current NBA PNGs to a real PNG file."""
    with Image.open(io.BytesIO(content)) as image:
        output = io.BytesIO()
        image.convert("RGBA").save(output, format="PNG")
    return output.getvalue()


def download_portraits(
    runs: pd.DataFrame,
    directory: Path,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Cache portraits and return provenance for every requested run."""
    required = {"SEASON", "TEAM_ID", "PLAYER_ID", "PLAYER_NAME"}
    missing = required.difference(runs.columns)
    if missing:
        raise ValueError(f"Portrait input missing columns: {sorted(missing)}")
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "headshot_manifest.csv"
    previous: dict[tuple[str, int], dict[str, object]] = {}
    if manifest_path.exists() and not refresh:
        prior = pd.read_csv(manifest_path)
        previous = {
            (str(row.SEASON), int(row.PLAYER_ID)): row._asdict()
            for row in prior.itertuples(index=False)
        }
    session = requests.Session()
    session.headers.update({
        "User-Agent": "nba-second-options-portfolio/1.0 (+educational analytics)",
        "Accept": "image/avif,image/webp,image/png,image/*,*/*;q=0.8",
    })
    records: list[dict[str, object]] = []
    unique = runs.drop_duplicates(["SEASON", "PLAYER_ID"])
    for _, row in unique.iterrows():
        target = portrait_path(directory, str(row.SEASON), int(row.PLAYER_ID))
        prior_record = previous.get((str(row.SEASON), int(row.PLAYER_ID)))
        if target.exists() and not refresh and prior_record:
            records.append(prior_record)
            continue
        key = (str(row.SEASON), int(row.PLAYER_ID))
        source_url = season_portrait_url(row)
        source_type = "SEASON_SPECIFIC_OFFICIAL_NBA"
        content = None if target.exists() and not refresh else _fetch(session, source_url)
        archived_url = ARCHIVED_SEASON_PORTRAIT_URLS.get(key)
        if content is None and archived_url and (refresh or not target.exists()):
            source_url = archived_url
            source_type = "SEASON_SPECIFIC_OFFICIAL_NBA_ARCHIVE"
            content = _fetch(session, source_url)
        if content is None and (refresh or not target.exists()):
            source_url = fallback_portrait_url(row)
            source_type = "CLOSEST_AVAILABLE_OFFICIAL_NBA_FALLBACK"
            content = _fetch(session, source_url)
        if content is not None:
            temporary = target.with_suffix(".tmp")
            temporary.write_bytes(_png_bytes(content))
            temporary.replace(target)
        status = source_type if target.exists() else "NOT_AVAILABLE"
        records.append({
            "SEASON": row.SEASON,
            "PLAYER_ID": int(row.PLAYER_ID),
            "PLAYER_NAME": row.PLAYER_NAME,
            "LOCAL_PATH": str(target),
            "PORTRAIT_STATUS": status,
            "SOURCE_URL": source_url,
        })
        time.sleep(0.15)
    manifest = pd.DataFrame(records)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ranking_csv", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("assets/headshots"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    frames = [pd.read_csv(path) for path in args.ranking_csv]
    manifest = download_portraits(pd.concat(frames, ignore_index=True),
                                  args.output_dir, refresh=args.refresh)
    LOG.info("Cached %s portraits in %s", manifest.LOCAL_PATH.notna().sum(),
             args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
