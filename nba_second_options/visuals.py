"""Publication-ready, rights-safe visuals for distinct analytical questions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BACKGROUND = "#07111f"
INK = "#f8fafc"
MUTED = "#a7b6c9"
GRID = "#2a3b52"
GOLD = "#f8c35c"
BLUE = "#61c6ff"
GREEN = "#5ee1a3"
ORANGE = "#ff9966"


def _playoff_label(row: pd.Series) -> str:
    year = int(str(row.SEASON).split("-", maxsplit=1)[0]) + 1
    return f"{year} {row.PLAYER_NAME}"


def plot_tactical_fit_mix(ranking: pd.DataFrame, output: Path) -> Path:
    """Show which tactical pathway contributes to each run's modeled fit."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    top = ranking.head(10).copy().reset_index(drop=True)
    columns = ["PRESSURE_VALVE_FIT", "GRAVITY_FIT", "DEFENSIVE_COVER_FIT"]
    raw = top[columns].apply(pd.to_numeric, errors="coerce").clip(lower=0)
    shares = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).mul(100).fillna(0)

    fig, ax = plt.subplots(figsize=(13, 11), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    y = np.arange(len(top))
    colors = [ORANGE, BLUE, GREEN]
    labels = ["Pressure valve", "Gravity / scalability", "Defensive cover"]
    left = np.zeros(len(top))

    for column, color in zip(columns, colors, strict=True):
        values = shares[column].to_numpy()
        bars = ax.barh(y, values, left=left, height=.63, color=color, alpha=.9)
        for row_index, (bar, value) in enumerate(zip(bars, values, strict=True)):
            if value >= 11:
                ax.text(left[row_index] + value / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{value:.0f}%", color=BACKGROUND, fontsize=9,
                        fontweight="bold", ha="center", va="center")
        left += values

    ax.set_yticks(y, [_playoff_label(row) for _, row in top.iterrows()])
    ax.invert_yaxis()
    ax.set_xlim(0, 136)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("HOW THE FIT SCORE BREAKS DOWN", color=INK,
                  fontsize=11, labelpad=14)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", colors=INK, labelsize=10.5, length=0, pad=12)
    ax.grid(axis="x", color=GRID, alpha=.35, linewidth=.8)
    ax.grid(axis="y", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row_index, (_, row) in enumerate(top.iterrows()):
        coverage = float(row.COMPLEMENT_FIT_COVERAGE) * 100
        ax.text(102.2, row_index,
                f"with {row.PRIMARY_PLAYER_NAME}\n{coverage:.0f}% of fit data available",
                color=MUTED, fontsize=8.3, ha="left", va="center")

    ax.set_title("WHAT TACTICAL NEED DID EACH SECOND OPTION FILL?", loc="left",
                 color=INK, fontsize=22, fontweight="bold", pad=52)
    ax.text(0, 1.025,
            "This shows how each player fit beside the No. 1—not who was better overall.",
            transform=ax.transAxes, color=MUTED, fontsize=10.5, va="bottom")
    legend = ax.legend(
        handles=[Patch(facecolor=color, label=label)
                 for color, label in zip(colors, labels, strict=True)],
        frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.035),
        fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.text(.08, .035,
             "Pressure valve = secondary creation · gravity = spacing or interior presence · "
             "defense is counted only when it fills a need beside the No. 1.",
             color=MUTED, fontsize=8.7)
    fig.subplots_adjust(left=.20, right=.95, top=.80, bottom=.12)
    path = output / "linkedin_tactical_fit_mix.png"
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_opponent_srs_paths(ranking: pd.DataFrame, output: Path) -> Path:
    """Compare each run's full opponent path with its toughest opponent."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    top = ranking.head(10).copy()
    top["PATH_SRS"] = pd.to_numeric(top.OPPONENT_SRS_WEIGHTED, errors="coerce")
    top["MAX_SRS"] = pd.to_numeric(top.OPPONENT_SRS_MAX, errors="coerce")
    top = top.sort_values(["PATH_SRS", "MAX_SRS"], ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 11), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    y = np.arange(len(top))
    for row_index, (_, row) in enumerate(top.iterrows()):
        path_srs, max_srs = float(row.PATH_SRS), float(row.MAX_SRS)
        ax.hlines(row_index, path_srs, max_srs, color=GRID, linewidth=4, zorder=1)
        ax.scatter(path_srs, row_index, s=115, color=BLUE, edgecolor=BACKGROUND,
                   linewidth=1, zorder=3)
        ax.scatter(max_srs, row_index, s=115, marker="D", color=GOLD,
                   edgecolor=BACKGROUND, linewidth=1, zorder=3)
        ax.text(path_srs - .18, row_index - .24, f"{path_srs:.1f}", color=BLUE,
                fontsize=8.5, fontweight="bold", ha="center", va="center")
        ax.text(max_srs + .18, row_index - .24, f"{max_srs:.1f}", color=GOLD,
                fontsize=8.5, fontweight="bold", ha="center", va="center")
        finish = "Won title" if row.POSTSEASON_FINISH == "Champion" else str(row.POSTSEASON_FINISH)
        ax.text(10.9, row_index, finish, color=INK, fontsize=9, ha="left", va="center")

    ax.set_yticks(y, [_playoff_label(row) for _, row in top.iterrows()])
    ax.invert_yaxis()
    ax.set_xlim(-.8, 12.7)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("OPPONENT REGULAR-SEASON SRS", color=INK, fontsize=11, labelpad=14)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", colors=INK, labelsize=10.5, length=0, pad=12)
    ax.grid(axis="x", color=GRID, alpha=.4, linewidth=.8)
    ax.grid(axis="y", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("WHO FACED THE TOUGHEST PLAYOFF ROAD?", loc="left", color=INK,
                 fontsize=22, fontweight="bold", pad=52)
    ax.text(0, 1.025,
            "Ordered by average opponent strength. Each line ends at the toughest team faced.",
            transform=ax.transAxes, color=MUTED, fontsize=10.5, va="bottom")
    legend = ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
                   markersize=9, label="Games-weighted playoff path"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=GOLD,
                   markersize=8, label="Toughest opponent"),
        ], frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.035),
        fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.text(.08, .035,
             "Average path strength weights games faced and gives later rounds a small bump. "
             "It changes the final score by no more than half a point.",
             color=MUTED, fontsize=8.7)
    fig.subplots_adjust(left=.20, right=.95, top=.80, bottom=.12)
    path = output / "linkedin_opponent_srs_paths.png"
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_linkedin_visuals(ranking: pd.DataFrame, output: Path) -> list[Path]:
    """Create two distinct explanatory graphics to accompany the leaderboard."""
    output.mkdir(parents=True, exist_ok=True)
    return [plot_tactical_fit_mix(ranking, output), plot_opponent_srs_paths(ranking, output)]
