from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import NOT_MODELED, PARTIAL, PipelineConfig, utc_now
from .sensitivity import CHAMPIONSHIP_DOMAINS, championship_weight_sensitivity
from .simple_scorecard import save_simple_scorecard
LOG = logging.getLogger(__name__)


def ranking_pool(frame: pd.DataFrame) -> pd.DataFrame:
    """Return candidates with meaningful terminal-series participation."""
    if "RANKING_ELIGIBLE" not in frame:
        return frame
    return frame[frame.RANKING_ELIGIBLE.fillna(False).astype(bool)]


def plot_outputs(frame: pd.DataFrame, output: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk", font_scale=.9)
    colors = {"Pressure Valve": "#f97316", "Gravity Engine": "#2563eb",
              "Defensive Anchor": "#16a34a", NOT_MODELED: "#64748b"}
    valid = frame.dropna(subset=["ROLE_COMPATIBILITY_SCORE", "IN_ERA_PRODUCTION_SCORE",
                                 "TS_PCT"]).copy()
    from matplotlib.lines import Line2D

    eligible_valid = ranking_pool(valid)
    highlighted = eligible_valid.nlargest(
        min(10, len(eligible_valid)), "BEST_SECOND_OPTION_SCORE")
    context = valid.drop(index=highlighted.index)
    fig = plt.figure(figsize=(16, 9), facecolor="#f8fafc")
    grid = fig.add_gridspec(1, 2, width_ratios=[3.15, 1.25], wspace=.06)
    ax = fig.add_subplot(grid[0, 0])
    key_ax = fig.add_subplot(grid[0, 1])
    ax.set_facecolor("#f8fafc")
    ax.scatter(context.ROLE_COMPATIBILITY_SCORE, context.IN_ERA_PRODUCTION_SCORE,
               s=52, alpha=.22, color="#64748b", edgecolor="none", zorder=1)
    for rank, (_, row) in enumerate(highlighted.iterrows(), 1):
        size = 230 + 500 * (min(.70, max(.48, float(row.TS_PCT))) - .48) / .22
        archetype = row.DOMINANT_ARCHETYPE
        color = colors.get(archetype, "#64748b")
        ax.scatter(row.ROLE_COMPATIBILITY_SCORE, row.IN_ERA_PRODUCTION_SCORE, s=size,
                   alpha=.94, color=color, edgecolor="white", linewidth=2.2, zorder=3)
        ax.text(row.ROLE_COMPATIBILITY_SCORE, row.IN_ERA_PRODUCTION_SCORE, str(rank),
                ha="center", va="center", color="white", fontsize=11,
                fontweight="bold", zorder=4)
    x_mid = valid.ROLE_COMPATIBILITY_SCORE.median()
    y_mid = valid.IN_ERA_PRODUCTION_SCORE.median()
    ax.axvline(x_mid, color="#94a3b8", linewidth=1.1, alpha=.65, zorder=0)
    ax.axhline(y_mid, color="#94a3b8", linewidth=1.1, alpha=.65, zorder=0)
    ax.text(.98, .97, "HIGH PRODUCTION + STRONG FIT", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color="#475569", fontweight="bold")
    ax.set_title("Production vs. Star-Specific Fit", loc="left", fontsize=22,
                 fontweight="bold", color="#0f172a", pad=18)
    ax.text(0, 1.01, f"{len(frame)} Conference Finals second-option runs · 2000–2026",
            transform=ax.transAxes, fontsize=11, color="#64748b", va="bottom")
    ax.set_xlabel("Era-aware role compatibility with the primary star  →", fontsize=13,
                  color="#334155", labelpad=12)
    ax.set_ylabel("In-era postseason production  →", fontsize=13,
                  color="#334155", labelpad=12)
    ax.tick_params(labelsize=11, colors="#475569")
    ax.grid(color="#cbd5e1", alpha=.45, linewidth=.8)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", label=label,
               markerfacecolor=color, markeredgecolor="white", markersize=10)
        for label, color in colors.items() if label != NOT_MODELED
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=False, fontsize=10,
              ncol=3, bbox_to_anchor=(0, -.01))

    key_ax.axis("off")
    key_ax.set_xlim(0, 1)
    key_ax.set_ylim(0, 1)
    key_ax.text(0, .985, "HIGHLIGHTED RUNS", va="top", fontsize=11,
                color="#64748b", fontweight="bold")
    for rank, (_, row) in enumerate(highlighted.iterrows(), 1):
        y = .91 - (rank - 1) * .088
        color = colors.get(row.DOMINANT_ARCHETYPE, "#64748b")
        key_ax.text(.02, y, str(rank), ha="center", va="center", fontsize=10,
                    color="white", fontweight="bold",
                    bbox={"boxstyle": "circle,pad=.30", "facecolor": color,
                          "edgecolor": "none"})
        key_ax.text(.10, y + .012,
                    f"{row.PLAYER_NAME} · {row.SEASON[-2:]}",
                    va="center", fontsize=11.5, color="#0f172a", fontweight="bold")
        key_ax.text(.10, y - .020,
                    f"{row.TEAM_ABBREVIATION}  |  {row.PTS_PER_75:.1f} pts/75  ·  "
                    f"{row.RELATIVE_TS_PCT:+.1%} rTS  ·  production {row.BEST_SECOND_OPTION_SCORE:.1f}",
                    va="center", fontsize=9.3, color="#64748b")
    fig.text(.01, .006,
             "Gray dots provide historical context. Highlight size reflects TS%. Evidence is the "
             "equal-domain geometric mean of scoring, efficiency, BPM and workload. Fit is "
             "descriptive; lineup interaction remains contextual, not causal.",
             fontsize=9.5, color="#64748b")
    fig.subplots_adjust(left=.07, right=.985, top=.90, bottom=.14, wspace=.06)
    scatter = output / "second_option_scatter.png"
    fig.savefig(scatter, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    score_cols = ["CREATION_SCORE", "OFF_BALL_SCORE", "DEFENSIVE_SCORE",
                  "NEED_NORMALIZED_FIT", "STRENGTH_AMPLIFICATION_SCORE"]
    pool = ranking_pool(frame)
    comparison = pool.nlargest(min(8, len(pool)), "BEST_SECOND_OPTION_SCORE")
    long = comparison.melt(id_vars=["PLAYER_NAME", "SEASON"], value_vars=score_cols,
                           var_name="Dimension", value_name="Score")
    long["Run"] = long.PLAYER_NAME + " " + long.SEASON
    labels = {"CREATION_SCORE": "Creation", "OFF_BALL_SCORE": "Off-ball scalability",
              "DEFENSIVE_SCORE": "Defense", "NEED_NORMALIZED_FIT": "Need fulfillment",
              "STRENGTH_AMPLIFICATION_SCORE": "Strength amplification"}
    long["Dimension"] = long.Dimension.map(labels)
    run_order = (comparison.PLAYER_NAME + " " + comparison.SEASON).tolist()
    dimension_order = list(labels.values())
    fig, ax = plt.subplots(figsize=(15, 9))
    sns.barplot(data=long, x="Score", y="Run", hue="Dimension", ax=ax,
                order=run_order, hue_order=dimension_order,
                palette=["#f97316", "#2563eb", "#16a34a", "#7c3aed", "#eab308"])
    offsets = dict(zip(dimension_order, [-.32, -.16, 0, .16, .32]))
    for _, row in long[long.Score.isna()].iterrows():
        ax.text(1.0, run_order.index(row.Run) + offsets[row.Dimension], "NM",
                va="center", ha="left", fontsize=8, color="#64748b", fontweight="bold")
    ax.set(title="What the Best Second Options Actually Supply",
           xlabel="Percentile score", ylabel="")
    ax.legend(title="Dimension", bbox_to_anchor=(1.01, 1), loc="upper left",
              borderaxespad=0)
    fig.tight_layout()
    comparison_png = output / "archetype_comparison.png"
    fig.savefig(comparison_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written = [scatter, comparison_png]
    try:
        import plotly.express as px
        chart = px.scatter(valid, x="ROLE_COMPATIBILITY_SCORE",
            y="IN_ERA_PRODUCTION_SCORE", size="TS_PCT",
            color="DOMINANT_ARCHETYPE", hover_name="PLAYER_NAME",
            hover_data=["SEASON", "TEAM_ABBREVIATION", "PRIMARY_PLAYER_NAME", "BPM",
                        "PPG", "TS_PCT", "RUN_CONFIDENCE"],
            color_discrete_map=colors,
            title="NBA Playoff Second Options: Production vs. Star-Specific Fit")
        interactive = output / "second_option_scatter.html"
        chart.write_html(interactive, include_plotlyjs="cdn")
        bars = output / "archetype_comparison.html"
        px.bar(long, x="Score", y="Run", color="Dimension", barmode="group",
               title="Top Second-Option Runs by Tactical Dimension").write_html(
                   bars, include_plotlyjs="cdn")
        written += [interactive, bars]
    except ImportError:
        LOG.warning("Plotly unavailable; interactive charts skipped")
    top_ten = pool.nlargest(min(10, len(pool)), "BEST_SECOND_OPTION_SCORE").copy()
    top_ten = top_ten.sort_values("BEST_SECOND_OPTION_SCORE")
    labels = top_ten.PLAYER_NAME + "  '" + top_ten.SEASON.str[-2:]
    fig, ax = plt.subplots(figsize=(13, 8), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.barh(labels, top_ten.BEST_SECOND_OPTION_SCORE, color="#334155",
            edgecolor="#94a3b8", label="Observed production score")
    ax.scatter(top_ten.PRODUCTION_SCORE, labels, color="#38bdf8", marker="D", s=74,
               label="In-era production", zorder=3)
    ax.scatter(top_ten.ROLE_COMPATIBILITY_SCORE, labels, color="#fb923c", s=82,
               label="Descriptive role compatibility", zorder=3)
    for index, (_, row) in enumerate(top_ten.iterrows()):
        ax.text(row.BEST_SECOND_OPTION_SCORE + .8, index,
                f"{row.BEST_SECOND_OPTION_SCORE:.1f}", va="center", color="white", fontsize=10)
    ax.set_title("The 10 Best Single-Season Second-Option Runs Since 2000",
                 color="white", pad=16, fontsize=20, fontweight="bold")
    ax.set_xlabel("Percentile-style score", color="#cbd5e1")
    ax.tick_params(colors="#e2e8f0")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="#334155", alpha=.55)
    ax.grid(axis="y", visible=False)
    legend = fig.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, .015),
                        ncol=3)
    for label in legend.get_texts():
        label.set_color("white")
    fig.tight_layout(rect=(0, .08, 1, 1))
    top_png = output / "top_10_second_options_since_2000.png"
    fig.savefig(top_png, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    written.append(top_png)
    try:
        import plotly.express as px
        top_html = output / "top_10_second_options_since_2000.html"
        px.bar(top_ten.sort_values("BEST_SECOND_OPTION_SCORE", ascending=False),
               x="BEST_SECOND_OPTION_SCORE", y="PLAYER_NAME", color="DOMINANT_ARCHETYPE",
               orientation="h", hover_data=["SEASON", "TEAM_ABBREVIATION", "PPG",
                                              "TS_PCT", "BPM", "NEED_NORMALIZED_FIT",
                                              "STRENGTH_AMPLIFICATION_SCORE",
                                              "ROLE_COMPATIBILITY_SCORE"],
               color_discrete_map=colors,
               title="The 10 Best Single-Season Second-Option Runs Since 2000").write_html(
                   top_html, include_plotlyjs="cdn")
        written.append(top_html)
    except ImportError:
        pass
    return written


def generate_summary(frame: pd.DataFrame, config: PipelineConfig) -> str:
    lines = ["# The NBA's Top Single-Season Playoff Second-Option Runs Since 2000", "",
             "> **Demonstration data — not research results.**" if config.demo else
             "> Conference Finals baseline; one row is one complete postseason run.", "",
             "## Executive summary", ""]
    eligible = ranking_pool(frame)
    leader = eligible.dropna(subset=["BEST_SECOND_OPTION_SCORE"]).nlargest(
        1, "BEST_SECOND_OPTION_SCORE")
    if not leader.empty:
        row = leader.iloc[0]
        lines.append(f"**{row.PLAYER_NAME} ({row.SEASON}, {row.TEAM_ABBREVIATION})** has the "
                     "strongest observed complete-run production profile in the reviewed dataset. "
                     "This is not a claim that lineup results are causal: "
                     "creation, spacing, defense, and pairing fit solve different problems.")
    lines += ["", "## Top 10 since 2000", ""]
    top_ten = eligible.nlargest(10, "BEST_SECOND_OPTION_SCORE")
    lines += ["| Rank | Run | Primary star | Pts/75 | rTS | BPM | In-era production | "
              "Need fit | Amplification | Role fit | Ranking score | Coverage |",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rank, (_, row) in enumerate(top_ten.iterrows(), 1):
        lines.append(
            f"| {rank} | {row.PLAYER_NAME}, {row.SEASON} {row.TEAM_ABBREVIATION} | "
            f"{row.PRIMARY_PLAYER_NAME} | {row.PTS_PER_75:.1f} | "
            f"{row.RELATIVE_TS_PCT:+.1%} | {row.BPM:.1f} | {row.PRODUCTION_SCORE:.1f} | "
            f"{row.NEED_NORMALIZED_FIT:.1f} | {row.STRENGTH_AMPLIFICATION_SCORE:.1f} | "
            f"{row.ROLE_COMPATIBILITY_SCORE:.1f} | {row.BEST_SECOND_OPTION_SCORE:.1f} | "
            f"{row.COMPLEMENT_FIT_COVERAGE:.0%} |")
    lines += ["", "## Archetype leaders", ""]
    for score, label in [("CREATION_SCORE", "Pressure Valve"),
                         ("OFF_BALL_SCORE", "Gravity Engine"),
                         ("DEFENSIVE_SCORE", "Defensive Anchor")]:
        lines += [f"### {label}", ""]
        leaders = frame.dropna(subset=[score]).nlargest(5, score)
        if leaders.empty:
            lines.append(f"{NOT_MODELED}: insufficient source coverage.")
        for _, row in leaders.iterrows():
            lines.append(f"- {row.PLAYER_NAME} — {row.SEASON} {row.TEAM_ABBREVIATION}: "
                         f"{row[score]:.1f} ({row.get(score + '_STATUS', PARTIAL)})")
        lines.append("")
    lines += ["## Tactical trade-offs", "",
        "- **Creation versus scalability:** isolation creation rescues broken possessions; shooting gravity preserves space.",
        "- **Output versus pairing fit:** points per 75, relative TS%, and BPM describe "
        "in-era production; fit explains tactical utility but does not determine rank.",
        "- **Defense versus visibility:** rim deterrence is scored only where tracking exists; blocks are not treated as complete rim protection.",
        "", "## Methodology", "",
        f"Players qualify at **{config.min_games}+ games** and **{config.min_mpg:g}+ MPG**. "
        "A documented human-in-the-loop role audit determines #1/#2 labels from an engine-aware proposal. "
        "Season baselines use all playoff rotation players meeting the qualifier. Production is the "
        "equal-domain geometric mean of era-relative points per 75, true shooting, BPM, and "
        "offensive burden. Fit explains the result but cannot override observed production. "
        "Candidates must play at least half the games in the team's terminal series at 15+ MPG.", "",
        "## Limitations", "",
        f"Unavailable tracking, play-type, lineup, and defensive fields remain `{NOT_MODELED}`; "
        "they are never converted to zeros or invented observations. Four-state lineup interaction "
        "is retained as `CONTEXTUAL_NOT_CAUSAL` and does not drive the ranking.", "", "## Sources", "",
        "- NBA Stats via `nba_api`: games, totals, usage, lineups, tracking, and play types.",
        "- Basketball-Reference: canonical playoff BPM 2.0."]
    return "\n".join(lines)


def data_quality(frame: pd.DataFrame) -> str:
    lines = ["# Data Quality Report", "", f"Generated: {utc_now()}", "",
             f"Rows: **{len(frame)}**", "", "| Metric | Modeled | Missing | Coverage |",
             "|---|---:|---:|---:|"]
    for metric in ["BPM", "ERA_PTS75_PERCENTILE", "RELATIVE_TS_PCT",
                   "NEED_NORMALIZED_FIT", "STRENGTH_AMPLIFICATION_SCORE",
                   "PAIR_SYNERGY_DELTA", "ISO_POSS_PCT", "CATCH_SHOOT_FG3A",
                   "RIM_DFG_PCT", "LATE_CLOCK_FGA"]:
        modeled = int(frame[metric].notna().sum()) if metric in frame else 0
        lines.append(f"| {metric} | {modeled} | {len(frame)-modeled} | "
                     f"{modeled/len(frame) if len(frame) else 0:.1%} |")
    return "\n".join(lines + ["", "Missing means `NOT_MODELED`, not zero impact."])


def save_era_context(frame: pd.DataFrame, output: Path) -> list[Path]:
    columns = ["SEASON", "ERA_ENVIRONMENT_SAMPLE_PLAYERS",
               "ERA_ENVIRONMENT_SAMPLE_POSSESSIONS", "ERA_PLAYOFF_PACE",
               "ERA_PLAYOFF_OFF_RATING", "ERA_PLAYOFF_TS_PCT",
               "ERA_PLAYOFF_3PA_RATE", "ERA_PLAYOFF_FTA_RATE"]
    if not set(columns).issubset(frame.columns):
        return []
    environment = frame[columns].drop_duplicates("SEASON").sort_values("SEASON")
    csv_path = output / "era_environment.csv"
    md_path = output / "era_methodology.md"
    environment.to_csv(csv_path, index=False)
    lines = ["# Era Adjustment Methodology", "",
             "Season context is calculated from every playoff rotation player meeting the "
             "8-game and 15-MPG qualifier—not only Conference Finalists.", "",
             "- Scoring volume uses points per 75 possessions.",
             "- Efficiency uses true shooting percentage relative to the season playoff baseline.",
             "- Usage, assist burden, and three-point volume are evaluated within season.",
             "- Tracking and archetype metrics use a centered three-season window with the "
             "current season receiving double weight.",
             "- Missing historical tracking remains `NOT_MODELED`; it is never backfilled as zero.",
             "- The leaderboard uses observed production only; fit remains descriptive.",
             "- Candidates must materially participate in their team's terminal series.", "",
             "| Season | Players | Pace | OffRtg | TS% | 3PA rate | FTA rate |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in environment.iterrows():
        lines.append(
            f"| {row.SEASON} | {int(row.ERA_ENVIRONMENT_SAMPLE_PLAYERS)} | "
            f"{row.ERA_PLAYOFF_PACE:.1f} | {row.ERA_PLAYOFF_OFF_RATING:.1f} | "
            f"{row.ERA_PLAYOFF_TS_PCT:.1%} | {row.ERA_PLAYOFF_3PA_RATE:.1%} | "
            f"{row.ERA_PLAYOFF_FTA_RATE:.1%} |")
    md_path.write_text("\n".join(lines))
    return [csv_path, md_path]


def plot_championship_sensitivity(sensitivity: pd.DataFrame, output: Path) -> Path:
    """Render decision-weight sensitivity intervals for title-winning runs."""
    import matplotlib.pyplot as plt

    ordered = sensitivity.sort_values(
        ["SENSITIVITY_MEDIAN_RANK", "SENSITIVITY_MEAN_SCORE"],
        ascending=[False, True],
    ).copy()
    labels = ordered.PLAYER_NAME + "  '" + ordered.SEASON.str[-2:]
    y = list(range(len(ordered)))
    lower = ordered.SENSITIVITY_MEDIAN_RANK - ordered.SENSITIVITY_P10_RANK
    upper = ordered.SENSITIVITY_P90_RANK - ordered.SENSITIVITY_MEDIAN_RANK
    colors = ["#22c55e" if rank <= 10 else "#38bdf8"
              for rank in ordered.SENSITIVITY_MEDIAN_RANK]
    for index, name in enumerate(ordered.PLAYER_NAME):
        if name == "Jason Terry":
            colors[index] = "#fb923c"
        elif name == "Jaylen Brown":
            colors[index] = "#a78bfa"

    fig, ax = plt.subplots(figsize=(15, 12), facecolor="#0b1220")
    ax.set_facecolor("#0b1220")
    ax.errorbar(ordered.SENSITIVITY_MEDIAN_RANK, y, xerr=[lower, upper], fmt="none",
                ecolor="#475569", elinewidth=3, capsize=4, zorder=1)
    ax.scatter(ordered.SENSITIVITY_MEDIAN_RANK, y, c=colors, s=95,
               edgecolor="#e2e8f0", linewidth=.8, zorder=2)
    for index, (_, row) in enumerate(ordered.iterrows()):
        ax.text(27.55, index,
                f"Top 10 in {row.SENSITIVITY_TOP_10_PROBABILITY:.0%}",
                va="center", fontsize=9, color="#94a3b8")
    ax.axvspan(.5, 10.5, color="#14532d", alpha=.18, zorder=0)
    ax.set_yticks(y, labels, color="#e2e8f0", fontsize=10.5)
    ax.set_xlim(.5, 32)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 27])
    ax.tick_params(axis="x", colors="#cbd5e1")
    ax.set_xlabel("Rank across 50,000 plausible value-weight combinations  →",
                  color="#cbd5e1")
    ax.set_title("Championship Ranking Sensitivity Diagnostic",
                 loc="left", color="white", fontsize=21, fontweight="bold", pad=20)
    ax.text(0, 1.01,
            "Dot = median rank · line = 10th–90th percentile · 2000–2026 champions",
            transform=ax.transAxes, color="#94a3b8", fontsize=11, va="bottom")
    ax.grid(axis="x", color="#334155", alpha=.55)
    ax.grid(axis="y", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.02, .015,
             "Six domains: rate production, role burden, cumulative impact, terminal-series "
             "responsibility, historical defense evidence, and primary-star compatibility. "
             "Intervals measure value-weight sensitivity—not statistical error bars.",
             color="#94a3b8", fontsize=9.5)
    fig.tight_layout(rect=(0, .045, 1, .97))
    path = output / "championship_ranking_sensitivity.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_outputs(frame: pd.DataFrame, config: PipelineConfig) -> list[Path]:
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    csv, parquet = output / "second_option_runs.csv", output / "second_option_runs.parquet"
    summary, quality = output / "executive_summary.md", output / "data_quality_report.md"
    frame.to_csv(csv, index=False)
    frame.to_parquet(parquet, index=False)
    summary.write_text(generate_summary(frame, config))
    quality.write_text(data_quality(frame))
    eligible = ranking_pool(frame)
    top_ten = eligible.nlargest(10, "BEST_SECOND_OPTION_SCORE").copy()
    top_ten.insert(0, "RANK", range(1, len(top_ten) + 1))
    top_csv = output / "top_10_second_options_since_2000.csv"
    top_md = output / "top_10_second_options_since_2000.md"
    top_ten.to_csv(top_csv, index=False)
    top_lines = ["# Production-Only Diagnostic: Top 10", "",
                 "**This is not the final contextual ranking.** It is an intentionally narrow "
                 "production lens: equal-domain geometric mean of era-relative scoring, "
                 "efficiency, BPM, and offensive burden. Fit is descriptive, lineup interaction "
                 "is excluded, and terminal-series participation is required.", "",
                 "| Rank | Run | #1 star | Pts/75 | rTS | BPM | Production | Need fit | "
                 "Amplification | Role fit | Ranking score | Coverage |",
                 "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in top_ten.iterrows():
        top_lines.append(
            f"| {int(row.RANK)} | {row.PLAYER_NAME}, {row.SEASON} {row.TEAM_ABBREVIATION} | "
            f"{row.PRIMARY_PLAYER_NAME} | {row.PTS_PER_75:.1f} | "
            f"{row.RELATIVE_TS_PCT:+.1%} | {row.BPM:.1f} | {row.PRODUCTION_SCORE:.1f} | "
            f"{row.NEED_NORMALIZED_FIT:.1f} | {row.STRENGTH_AMPLIFICATION_SCORE:.1f} | "
            f"{row.ROLE_COMPATIBILITY_SCORE:.1f} | {row.BEST_SECOND_OPTION_SCORE:.1f} | "
            f"{row.COMPLEMENT_FIT_COVERAGE:.0%} |")
    top_md.write_text("\n".join(top_lines))
    sensitivity = championship_weight_sensitivity(frame)
    champions = sensitivity.head(10).copy()
    champions.insert(0, "RANK", range(1, len(champions) + 1))
    champions_csv = output / "top_10_championship_second_options.csv"
    champions_md = output / "top_10_championship_second_options.md"
    champions.to_csv(champions_csv, index=False)
    champion_lines = ["# Provisional Championship Weight-Sensitivity Top 10", "",
        "**Diagnostic only—not the final contextual ranking.** This lens holds the team outcome "
        "constant and uses the median rank across 50,000 weight combinations. It reveals "
        "weight dependence but does not estimate role-conditioned replacement value.", "",
        "| Rank | Run | #1 star | Pts/75 | rTS | BPM | Median rank | Rank range | Top-10 share |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in champions.iterrows():
        champion_lines.append(
            f"| {int(row.RANK)} | {row.PLAYER_NAME}, {row.SEASON} "
            f"{row.TEAM_ABBREVIATION} | {row.PRIMARY_PLAYER_NAME} | "
            f"{row.PTS_PER_75:.1f} | {row.RELATIVE_TS_PCT:+.1%} | {row.BPM:.1f} | "
            f"{row.SENSITIVITY_MEDIAN_RANK:.0f} | "
            f"{row.SENSITIVITY_P10_RANK:.0f}–{row.SENSITIVITY_P90_RANK:.0f} | "
            f"{row.SENSITIVITY_TOP_10_PROBABILITY:.1%} |")
    champions_md.write_text("\n".join(champion_lines))
    completed_2000_2026 = sensitivity.copy()
    completed_2000_2026.insert(0, "RANK", range(1, len(completed_2000_2026) + 1))
    all_champions_csv = output / "all_27_championship_second_options_2000_2026.csv"
    all_champions_md = output / "all_27_championship_second_options_2000_2026.md"
    completed_2000_2026.to_csv(all_champions_csv, index=False)
    all_champion_lines = [
        "# All 27 Championship Second Options: Provisional Sensitivity Order", "",
        "**Diagnostic only—not the final contextual ranking.** All title teams are held to "
        "the same outcome filter. The displayed order is the median rank across 50,000 "
        "six-domain weight combinations and exists to expose definition sensitivity.", "",
        "| Rank | Playoffs | Team | #2 option | #1 star | PPG | Pts/75 | rTS | BPM | "
        "Role burden | Defense evidence | Median | Range |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in completed_2000_2026.iterrows():
        all_champion_lines.append(
            f"| {int(row.RANK)} | {int(row.SEASON[:4]) + 1} | "
            f"{row.TEAM_ABBREVIATION} | {row.PLAYER_NAME} | "
            f"{row.PRIMARY_PLAYER_NAME} | {row.PPG:.1f} | {row.PTS_PER_75:.1f} | "
            f"{row.RELATIVE_TS_PCT:+.1%} | {row.BPM:.1f} | "
            f"{row.ROLE_BURDEN_SCORE:.1f} | {row.HISTORICAL_DEFENSE_EVIDENCE_SCORE:.1f} | "
            f"{row.SENSITIVITY_MEDIAN_RANK:.0f} | "
            f"{row.SENSITIVITY_P10_RANK:.0f}–{row.SENSITIVITY_P90_RANK:.0f} |")
    all_champions_md.write_text("\n".join(all_champion_lines))
    sensitivity_csv = output / "championship_ranking_sensitivity.csv"
    sensitivity_md = output / "championship_ranking_sensitivity.md"
    sensitivity.to_csv(sensitivity_csv, index=False)
    sensitivity_lines = [
        "# Championship Second-Option Weight Sensitivity Diagnostic", "",
        "**This is not the final contextual ranking.** It is a 50,000-scenario robustness "
        "analysis across six evidence domains. "
        "The interval measures sensitivity to the definition of second-option value, "
        "not statistical uncertainty in player performance.", "",
        "Domains: " + ", ".join(CHAMPIONSHIP_DOMAINS), "",
        "| Consensus | Run | Median rank | 10th–90th percentile rank | Top-10 share | "
        "Top-five share | Mean domain score |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for consensus, (_, row) in enumerate(sensitivity.iterrows(), 1):
        sensitivity_lines.append(
            f"| {consensus} | {row.PLAYER_NAME}, {row.SEASON} | "
            f"{row.SENSITIVITY_MEDIAN_RANK:.0f} | "
            f"{row.SENSITIVITY_P10_RANK:.0f}–{row.SENSITIVITY_P90_RANK:.0f} | "
            f"{row.SENSITIVITY_TOP_10_PROBABILITY:.1%} | "
            f"{row.SENSITIVITY_TOP_5_PROBABILITY:.1%} | "
            f"{row.SENSITIVITY_MEAN_SCORE:.1f} |")
    sensitivity_md.write_text("\n".join(sensitivity_lines))
    sensitivity_png = plot_championship_sensitivity(sensitivity, output)
    era_paths = save_era_context(frame, output)
    charts = plot_outputs(frame, output)
    simple_paths = save_simple_scorecard(frame, output)
    manifest = output / "run_manifest.json"
    manifest.write_text(json.dumps({"generated_at": utc_now(), "configuration": {
        **asdict(config), "cache_dir": str(config.cache_dir), "output_dir": str(output)},
        "rows": len(frame), "demo": config.demo,
        "artifacts": [p.name for p in [csv, parquet, summary, quality, top_csv, top_md,
                                        champions_csv, champions_md, all_champions_csv,
                                        all_champions_md, sensitivity_csv, sensitivity_md,
                                        sensitivity_png]
                      + era_paths + charts + simple_paths],
        "python": sys.version}, indent=2, default=str))
    return [csv, parquet, summary, quality, top_csv, top_md, champions_csv,
            champions_md, all_champions_csv, all_champions_md, *era_paths,
            sensitivity_csv, sensitivity_md, sensitivity_png, manifest] + charts + simple_paths
