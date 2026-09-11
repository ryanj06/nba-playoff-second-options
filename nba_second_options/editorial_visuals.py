"""Editorial, social-ready graphics for the final transparent scorecard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw

from .headshots import portrait_path

BACKGROUND = "#07111f"
PANEL = "#0d1b2d"
INK = "#f8fafc"
MUTED = "#9fb0c5"
GRID = "#26384f"
GOLD = "#f8c35c"
PERFORMANCE = "#63c5ff"
RESPONSIBILITY = "#ff9966"
RUN_VALUE = "#b69cff"
FIT = "#5ee1a3"
SCHEDULE = "#f472b6"

TEAM_COLORS = {
    "BOS": "#007a33", "CLE": "#fdbb30", "DEN": "#fec524",
    "GSW": "#1d70c9", "LAL": "#f5b335", "MIA": "#f43f5e",
    "NYK": "#f58426", "OKC": "#48a9e6", "SAS": "#b8c4ce",
}

def _portrait(directory: Path, row: pd.Series) -> np.ndarray | None:
    path = portrait_path(directory, str(row.SEASON), int(row.PLAYER_ID))
    if not path.exists():
        return None
    image = Image.open(path).convert("RGBA")
    alpha = np.asarray(image.getchannel("A"))
    if alpha.max() > 0:
        box = image.getbbox()
        if box:
            image = image.crop(box)
    # NBA Player File portraits from the early 2000s were only 65x90. Pad
    # rather than stretch them, then upscale so they occupy the same visual
    # area as newer 260x190 league headshots.
    if image.width < 120 and image.height < 140:
        side = max(image.size)
        square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        square.alpha_composite(image, ((side - image.width) // 2, 0))
        mask = Image.new("L", square.size, 0)
        ImageDraw.Draw(mask).ellipse((0, 0, side - 1, side - 1), fill=255)
        square.putalpha(ImageChops.multiply(square.getchannel("A"), mask))
        image = square.resize((160, 160), Image.Resampling.LANCZOS)
    return np.asarray(image)


def _place_portrait(ax, image: np.ndarray | None, xy: tuple[float, float],
                    *, zoom: float, accent: str, initials: str = "") -> None:
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.patches import Circle

    ax.add_patch(Circle(xy, .37, transform=ax.transData, facecolor=PANEL,
                        edgecolor=accent, linewidth=2.2, zorder=4))
    if image is not None:
        artist = AnnotationBbox(OffsetImage(image, zoom=zoom), xy,
                                frameon=False, pad=0, zorder=5)
        ax.add_artist(artist)
    elif initials:
        ax.text(*xy, initials, color=INK, fontsize=10, fontweight="bold",
                ha="center", va="center", zorder=5)


def _photo_note(directory: Path, runs: pd.DataFrame) -> str:
    manifest_path = directory / "headshot_manifest.csv"
    if not manifest_path.exists():
        return "Portraits unavailable; initials are used where needed."
    manifest = pd.read_csv(manifest_path).merge(
        runs[["SEASON", "PLAYER_ID"]].drop_duplicates(),
        on=["SEASON", "PLAYER_ID"], how="inner",
    )
    exact = int(manifest.PORTRAIT_STATUS.str.startswith("SEASON_SPECIFIC", na=False).sum())
    fallback = int(manifest.PORTRAIT_STATUS.str.contains("FALLBACK", na=False).sum())
    missing = int(manifest.PORTRAIT_STATUS.eq("NOT_AVAILABLE").sum())
    return (f"Official NBA headshots · {exact} from the listed season · "
            f"{fallback} nearest-season fallback · {missing} shown as initials")


def plot_headshot_leaderboard(
    ranking: pd.DataFrame,
    output: Path,
    portrait_directory: Path,
    *,
    title: str = "TOP PLAYOFF SECOND OPTIONS SINCE 2000",
    subtitle: str = "One postseason at a time · Conference Finals or better",
    filename: str = "linkedin_top_10_with_headshots.png",
) -> Path:
    """Create a square LinkedIn-ready top-ten run leaderboard."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    top = ranking.head(10).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 12), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(0, 12)
    ax.set_ylim(-.9, 11.8)
    ax.axis("off")
    ax.text(.4, 11.35, title,
            color=INK, fontsize=24, fontweight="bold", va="top")
    ax.text(.4, 10.91, subtitle,
            color=MUTED, fontsize=11, va="top")

    y_positions = np.arange(9.95, -.05, -1.0)
    for idx, (_, row) in enumerate(top.iterrows()):
        y = y_positions[idx]
        accent = TEAM_COLORS.get(str(row.TEAM_ABBREVIATION), PERFORMANCE)
        ax.add_patch(FancyBboxPatch((.35, y - .43), 11.25, .86,
                                    boxstyle="round,pad=0.02,rounding_size=.10",
                                    facecolor=PANEL if idx % 2 == 0 else BACKGROUND,
                                    edgecolor="none", zorder=0))
        ax.text(.65, y, str(idx + 1).zfill(2), color=GOLD, fontsize=15,
                fontweight="bold", va="center", ha="center")
        image = _portrait(portrait_directory, row)
        initials = "".join(part[0] for part in str(row.PLAYER_NAME).split()[:2])
        _place_portrait(ax, image, (1.55, y), zoom=.26, accent=accent,
                        initials=initials)
        ax.text(2.15, y + .14, str(row.PLAYER_NAME), color=INK, fontsize=13,
                fontweight="bold", va="center")
        season_short = f"{int(str(row.SEASON).split('-', maxsplit=1)[0]) + 1} playoffs"
        ax.text(2.15, y - .16,
                f"{season_short} · {row.TEAM_ABBREVIATION} · with {row.PRIMARY_PLAYER_NAME}",
                color=MUTED, fontsize=8.8, va="center")
        components = [
            ("P", float(row.OBSERVED_PERFORMANCE_SCORE), PERFORMANCE),
            ("V", float(row.CUMULATIVE_IMPACT_SCORE), RUN_VALUE),
            ("R", float(row.ROLE_RESPONSIBILITY_SCORE), RESPONSIBILITY),
            ("F", float(row.FIT_EVIDENCE_SCORE), FIT),
            ("S", float(row.SCHEDULE_DIFFICULTY_SCORE), SCHEDULE),
        ]
        start_x = 5.55
        for component_idx, (label, value, color) in enumerate(components):
            x = start_x + component_idx * .93
            ax.text(x, y + .18, label, color=color, fontsize=8,
                    fontweight="bold", va="center")
            ax.plot([x, x + .54], [y - .04, y - .04], color=GRID,
                    linewidth=5, solid_capstyle="round", zorder=1)
            ax.plot([x, x + .54 * value / 100], [y - .04, y - .04], color=color,
                    linewidth=5, solid_capstyle="round", zorder=2)
            ax.text(x + .27, y + .18, f"{value:.0f}", color=INK,
                    fontsize=8.5, ha="center", va="center")
        ax.text(11.18, y + .06, f"{row.SIMPLE_BALANCED_SCORE:.1f}", color=INK,
                fontsize=18, fontweight="bold", ha="right", va="center")
        ax.text(11.18, y - .22, "FINAL SCORE", color=MUTED, fontsize=6.8,
                ha="right", va="center")
        ax.text(10.38, y - .22, f"CONTEXT {row.TOTAL_CONTEXT_ADJUSTMENT:+.1f}",
                color=MUTED, fontsize=6.8, ha="right", va="center")

    ax.text(.4, -.36, "P  production   V  total playoff value   R  role   F  fit with the star   S  opponent strength",
            color=MUTED, fontsize=8.2, va="center")
    ax.text(.4, -.66, _photo_note(portrait_directory, top), color=MUTED,
            fontsize=7.6, va="center")
    path = output / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_context_map(
    ranking: pd.DataFrame,
    output: Path,
    portrait_directory: Path,
) -> Path:
    """Map production against role burden with portraits as the observations."""
    import matplotlib.pyplot as plt

    top = ranking.head(10).copy()
    fig, ax = plt.subplots(figsize=(14, 9), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    x = top.ROLE_RESPONSIBILITY_SCORE.astype(float)
    y = top.OBSERVED_PERFORMANCE_SCORE.astype(float)
    x_mid, y_mid = float(x.median()), float(y.median())
    ax.axvline(x_mid, color=GRID, linewidth=1.2)
    ax.axhline(y_mid, color=GRID, linewidth=1.2)
    label_offsets = {
        ("Anthony Davis", "2019-20"): (-27, 26),
        ("Stephen Curry", "2016-17"): (30, -25),
        ("Shaquille O'Neal", "2003-04"): (0, -27),
        ("Kyrie Irving", "2015-16"): (-34, 26),
        ("Kobe Bryant", "2000-01"): (34, -26),
    }
    for _, row in top.iterrows():
        accent = TEAM_COLORS.get(str(row.TEAM_ABBREVIATION), PERFORMANCE)
        image = _portrait(portrait_directory, row)
        initials = "".join(part[0] for part in str(row.PLAYER_NAME).split()[:2])
        _place_portrait(ax, image,
                        (float(row.ROLE_RESPONSIBILITY_SCORE),
                         float(row.OBSERVED_PERFORMANCE_SCORE)),
                        zoom=.21, accent=accent, initials=initials)
        playoff_year = int(str(row.SEASON).split("-", maxsplit=1)[0]) + 1
        offset = label_offsets.get((str(row.PLAYER_NAME), str(row.SEASON)), (0, -27))
        ax.annotate(f"{row.PLAYER_NAME} · {playoff_year}",
                    (float(row.ROLE_RESPONSIBILITY_SCORE),
                     float(row.OBSERVED_PERFORMANCE_SCORE)),
                    xytext=offset, textcoords="offset points", ha="center",
                    color=INK, fontsize=8.5, fontweight="bold")
    x_pad = max(6.0, float(x.max() - x.min()) * .16)
    y_pad = max(6.0, float(y.max() - y.min()) * .16)
    ax.set_xlim(float(x.min()) - x_pad, float(x.max()) + x_pad)
    ax.set_ylim(float(y.min()) - y_pad, float(y.max()) + y_pad)
    ax.set_title("WHO CARRIED THE MOST—AND PRODUCED THE MOST?", loc="left",
                 color=INK, fontsize=22, fontweight="bold", pad=22)
    ax.text(0, 1.01, "Top 10 runs in the final scorecard · portraits mark single postseasons",
            transform=ax.transAxes, color=MUTED, fontsize=10, va="bottom")
    ax.set_xlabel("Role responsibility →", color=INK, labelpad=12)
    ax.set_ylabel("Observed era-relative performance →", color=INK, labelpad=12)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, alpha=.35, linewidth=.7)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.text(.98, .96, "HEAVY BURDEN + ELITE OUTPUT", transform=ax.transAxes,
            color=GOLD, fontsize=9, fontweight="bold", ha="right", va="top")
    fig.text(.08, .025, "Portrait outline uses team color. Ranking uses four non-duplicative evidence layers.",
             color=MUTED, fontsize=8.5)
    path = output / "second_option_context_map_with_headshots.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_editorial_visuals(
    overall_ranking: pd.DataFrame,
    output: Path,
    portrait_directory: Path,
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    return [
        plot_headshot_leaderboard(overall_ranking, output, portrait_directory),
        plot_context_map(overall_ranking, output, portrait_directory),
    ]
