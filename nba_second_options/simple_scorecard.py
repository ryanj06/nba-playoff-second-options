"""Final scorecard and public-facing ranking outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .era import rolling_era_percentile
from .visuals import save_linkedin_visuals


FINAL_SCORE_WEIGHTS = {
    "OBSERVED_PERFORMANCE_SCORE": .43,
    "CUMULATIVE_IMPACT_SCORE": .22,
    "ROLE_RESPONSIBILITY_SCORE": .15,
    "FIT_EVIDENCE_SCORE": .20,
}

PLAUSIBLE_WEIGHT_RANGES = {
    "OBSERVED_PERFORMANCE_SCORE": (.40, .60),
    "CUMULATIVE_IMPACT_SCORE": (.15, .30),
    "ROLE_RESPONSIBILITY_SCORE": (.10, .25),
    "FIT_EVIDENCE_SCORE": (.10, .25),
}

FINISH_ADJUSTMENT = {"Conference Finals": 0.0, "Finals": 1.5, "Champion": 3.5}
SRS_ADJUSTMENT_CAP = .5
TERMINAL_ADJUSTMENT_CAP = .5


def geometric_mean(frame: pd.DataFrame) -> pd.Series:
    """Return an equal-domain geometric mean only for complete rows."""
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    result = np.exp(np.log(numeric.clip(lower=1e-6)).mean(axis=1))
    return result.where(numeric.notna().all(axis=1))


def weighted_geometric_mean(
    frame: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Return a weighted geometric mean only where every named domain exists."""
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Score weights must sum to one")
    numeric = frame[list(weights)].apply(pd.to_numeric, errors="coerce")
    logged = np.log(numeric.clip(lower=1e-6))
    result = sum(logged[column] * weight for column, weight in weights.items())
    return np.exp(result).where(numeric.notna().all(axis=1))


def attach_simple_scorecard(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate the four-part score and capped postseason adjustments."""
    out = frame.copy()
    out["OBSERVED_PERFORMANCE_SCORE"] = geometric_mean(out[[
        "ERA_PTS75_PERCENTILE",
        "ERA_TS_PERCENTILE",
        "ERA_BPM_PERCENTILE",
    ]])
    scoring_share = rolling_era_percentile(out, "TEAM_PTS_SHARE")
    playmaking_share = rolling_era_percentile(out, "TEAM_AST_SHARE")
    rebounding_share = rolling_era_percentile(out, "TEAM_REB_SHARE")
    rim_protection_share = rolling_era_percentile(out, "TEAM_BLK_SHARE")
    position_group = out.get(
        "POSITION_GROUP", pd.Series("", index=out.index, dtype=object))
    interior_load = geometric_mean(pd.DataFrame({
        "rebounding": rebounding_share,
        "rim_protection": rim_protection_share,
    })).where(position_group.eq("BIG"))
    # Responsibility may be carried through scoring, creation, or a big's
    # interior workload. Taking the strongest route avoids demanding guard-style
    # assists from centers or center-style rim protection from guards.
    out["FULL_RUN_OFFENSIVE_RESPONSIBILITY_SCORE"] = pd.concat(
        [scoring_share, playmaking_share], axis=1
    ).max(axis=1)
    out["ROLE_ROUTE_RESPONSIBILITY_SCORE"] = pd.concat(
        [scoring_share, playmaking_share, interior_load], axis=1
    ).max(axis=1)
    responsibility = pd.DataFrame({
        "usage": out.ERA_USG_PERCENTILE,
        "role_route": out.ROLE_ROUTE_RESPONSIBILITY_SCORE,
        "terminal_minutes": rolling_era_percentile(out, "DEEPEST_ROUND_MPG"),
    })
    out["ROLE_RESPONSIBILITY_SCORE"] = geometric_mean(responsibility)
    # Defense remains visible but does not receive a second independent vote:
    # total BPM already contains DBPM, while box-only defense is least reliable.
    out["DEFENSIVE_EVIDENCE_SCORE"] = pd.to_numeric(
        out.HISTORICAL_DEFENSE_EVIDENCE_SCORE, errors="coerce"
    )
    fit = pd.to_numeric(out.ROLE_COMPATIBILITY_SCORE, errors="coerce")
    fit_coverage = pd.to_numeric(out.COMPLEMENT_FIT_COVERAGE, errors="coerce").clip(0, 1)
    out["FIT_EVIDENCE_SCORE"] = fit_coverage * fit + (1 - fit_coverage) * 50
    out["FIT_EVIDENCE_STATUS"] = np.where(
        fit.notna() & fit_coverage.notna(), "COVERAGE_SHRUNK", "NOT_MODELED"
    )
    path_average = rolling_era_percentile(out, "OPPONENT_SRS_WEIGHTED")
    path_ceiling = rolling_era_percentile(out, "OPPONENT_SRS_MAX")
    out["SCHEDULE_DIFFICULTY_SCORE"] = geometric_mean(pd.DataFrame({
        "full_path": path_average,
        "toughest_opponent": path_ceiling,
    }))
    srs_inputs = pd.DataFrame({
        "path": pd.to_numeric(out.get("OPPONENT_SRS_WEIGHTED"), errors="coerce"),
        "ceiling": pd.to_numeric(out.get("OPPONENT_SRS_MAX"), errors="coerce"),
    }, index=out.index)
    out["SCHEDULE_DIFFICULTY_STATUS"] = np.where(
        srs_inputs.notna().all(axis=1),
        "BASKETBALL_REFERENCE_SRS", "NOT_MODELED",
    )
    out["SIMPLE_BALANCED_CORE_SCORE"] = weighted_geometric_mean(
        out, FINAL_SCORE_WEIGHTS
    )
    out["FINISH_CONTEXT_ADJUSTMENT"] = out.POSTSEASON_FINISH.map(
        FINISH_ADJUSTMENT).fillna(0)
    out["SRS_CONTEXT_ADJUSTMENT"] = (
        (out.SCHEDULE_DIFFICULTY_SCORE - 50) / 50 * SRS_ADJUSTMENT_CAP
    ).clip(-SRS_ADJUSTMENT_CAP, SRS_ADJUSTMENT_CAP).fillna(0)
    terminal_context = rolling_era_percentile(out, "TERMINAL_RESPONSIBILITY_SCORE")
    out["TERMINAL_CONTEXT_ADJUSTMENT"] = (
        (terminal_context - 50) / 50 * TERMINAL_ADJUSTMENT_CAP
    ).clip(-TERMINAL_ADJUSTMENT_CAP, TERMINAL_ADJUSTMENT_CAP).fillna(0)
    out["TOTAL_CONTEXT_ADJUSTMENT"] = out[[
        "FINISH_CONTEXT_ADJUSTMENT", "SRS_CONTEXT_ADJUSTMENT",
        "TERMINAL_CONTEXT_ADJUSTMENT",
    ]].sum(axis=1, min_count=3)
    out["SIMPLE_BALANCED_SCORE"] = (
        out.SIMPLE_BALANCED_CORE_SCORE + out.TOTAL_CONTEXT_ADJUSTMENT
    ).clip(0, 100)
    score_columns = list(FINAL_SCORE_WEIGHTS)
    out["SIMPLE_SCORE_COVERAGE"] = out[score_columns].notna().mean(axis=1)
    out["SIMPLE_RANKING_METHOD"] = (
        "CORE_GEOMEAN_RATE43_RUN22_RESPONSIBILITY15_FIT20_PLUS_BOUNDED_CONTEXT"
    )
    return out


def championship_scorecard(frame: pd.DataFrame) -> pd.DataFrame:
    """Return eligible championship runs in scorecard order."""
    scored = attach_simple_scorecard(frame)
    eligible = scored[
        scored.RANKING_ELIGIBLE.fillna(False).astype(bool)
        & scored.POSTSEASON_FINISH.eq("Champion")
        & scored.SIMPLE_BALANCED_SCORE.notna()
    ].copy()
    eligible = eligible.sort_values(
        ["SIMPLE_BALANCED_SCORE", "OBSERVED_PERFORMANCE_SCORE"],
        ascending=False,
    )
    eligible.insert(0, "RANK", range(1, len(eligible) + 1))
    return eligible


def overall_scorecard(frame: pd.DataFrame) -> pd.DataFrame:
    """Return every eligible Conference Finals run in scorecard order."""
    scored = attach_simple_scorecard(frame)
    eligible = scored[
        scored.RANKING_ELIGIBLE.fillna(False).astype(bool)
        & scored.SIMPLE_BALANCED_SCORE.notna()
    ].copy()
    eligible = eligible.sort_values(
        ["SIMPLE_BALANCED_SCORE", "OBSERVED_PERFORMANCE_SCORE"],
        ascending=False,
    )
    eligible.insert(0, "RANK", range(1, len(eligible) + 1))
    return eligible


def one_run_per_player_scorecard(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the best-scoring eligible run for each distinct player."""
    overall = overall_scorecard(frame).drop(columns="RANK")
    result = overall.drop_duplicates("PLAYER_ID", keep="first").copy()
    result.insert(0, "RANK", range(1, len(result) + 1))
    return result


def ranking_robustness(
    frame: pd.DataFrame,
    *,
    simulations: int = 20_000,
    seed: int = 20260910,
) -> pd.DataFrame:
    """Stress-test the final order across a documented range of value judgments.

    These are sensitivity frequencies, not statistical confidence intervals.
    """
    scored = attach_simple_scorecard(frame)
    eligible = scored[
        scored.RANKING_ELIGIBLE.fillna(False).astype(bool)
        & scored[list(FINAL_SCORE_WEIGHTS)].notna().all(axis=1)
    ].copy().reset_index(drop=True)
    if eligible.empty:
        return eligible
    rng = np.random.default_rng(seed)
    draws = np.column_stack([
        rng.uniform(low, high, simulations)
        for low, high in PLAUSIBLE_WEIGHT_RANGES.values()
    ])
    draws /= draws.sum(axis=1, keepdims=True)
    logged = np.log(eligible[list(FINAL_SCORE_WEIGHTS)].clip(lower=1e-6).to_numpy())
    scores = np.exp(logged @ draws.T)
    scores += eligible.TOTAL_CONTEXT_ADJUSTMENT.to_numpy()[:, None]
    scores = np.clip(scores, 0, 100)
    ranks = (-scores).argsort(axis=0).argsort(axis=0) + 1
    eligible["ROBUSTNESS_FIRST_PROBABILITY"] = (ranks == 1).mean(axis=1)
    eligible["ROBUSTNESS_TOP_5_PROBABILITY"] = (ranks <= 5).mean(axis=1)
    eligible["ROBUSTNESS_TOP_10_PROBABILITY"] = (ranks <= 10).mean(axis=1)
    eligible["ROBUSTNESS_SCORE_P10"] = np.quantile(scores, .10, axis=1)
    eligible["ROBUSTNESS_SCORE_P90"] = np.quantile(scores, .90, axis=1)
    eligible["ROBUSTNESS_SIMULATIONS"] = simulations
    eligible["ROBUSTNESS_SEED"] = seed
    return eligible.sort_values(
        ["ROBUSTNESS_FIRST_PROBABILITY", "SIMPLE_BALANCED_SCORE"],
        ascending=False,
    )


def scorecard_markdown(ranking: pd.DataFrame) -> str:
    lines = [
        "# Championship Second Options Since 2000",
        "",
        "This table holds team result constant: every player here won the title. The "
        "score is 43% era-adjusted production, 22% value across the full run, 15% role "
        "responsibility, and 20% fit beside the primary star. Opponent strength and the "
        "final series can each move the result by no more than half a point.",
        "",
        "| Rank | Run | #1 star | Performance | Run value | Responsibility | Fit | Core | Context | Score |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ranking.iterrows():
        lines.append(
            f"| {int(row.RANK)} | {row.PLAYER_NAME}, {row.SEASON} "
            f"{row.TEAM_ABBREVIATION} | {row.PRIMARY_PLAYER_NAME} | "
            f"{row.OBSERVED_PERFORMANCE_SCORE:.1f} | "
            f"{row.CUMULATIVE_IMPACT_SCORE:.1f} | "
            f"{row.ROLE_RESPONSIBILITY_SCORE:.1f} | "
            f"{row.FIT_EVIDENCE_SCORE:.1f} | "
            f"{row.SIMPLE_BALANCED_CORE_SCORE:.1f} | "
            f"{row.TOTAL_CONTEXT_ADJUSTMENT:+.1f} | "
            f"{row.SIMPLE_BALANCED_SCORE:.1f} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Defense already appears in BPM and can help the fit score when it fills a "
        "specific need beside the primary star.",
        "- Blocks and defensive box-score stats do not stand in for positioning, "
        "matchups, or rim deterrence.",
        "- This compares postseason runs. It does not claim to isolate chemistry.",
    ])
    return "\n".join(lines)


def plot_scorecard(
    ranking: pd.DataFrame,
    output: Path,
    *,
    title: str,
    subtitle: str,
    filename: str,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    top = ranking.head(10).copy().reset_index(drop=True)
    background = "#07111f"
    panel = "#0d1b2d"
    ink = "#f8fafc"
    muted = "#9fb0c5"
    track = "#26384f"
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=background)
    ax.set_facecolor(background)
    ax.set_xlim(-3.65, 9.25)
    ax.set_ylim(-1.25, len(top) + 1.15)
    ax.axis("off")
    layers = [
        ("OBSERVED_PERFORMANCE_SCORE", "PRODUCTION", "#63c5ff"),
        ("CUMULATIVE_IMPACT_SCORE", "RUN VALUE", "#b69cff"),
        ("ROLE_RESPONSIBILITY_SCORE", "ROLE", "#ff9966"),
        ("FIT_EVIDENCE_SCORE", "STAR FIT", "#5ee1a3"),
        ("SCHEDULE_DIFFICULTY_SCORE", "OPPONENTS", "#f472b6"),
    ]
    starts = np.array([0.0, 1.55, 3.10, 4.65, 6.20])
    track_width = 1.02
    score_x = 8.45

    for start, (_, header, color) in zip(starts, layers, strict=True):
        ax.text(start + track_width / 2, len(top) + .34, header, color=color,
                fontsize=9, fontweight="bold", ha="center", va="center")
    ax.text(score_x, len(top) + .34, "FINAL", color="#f8c35c", fontsize=9,
            fontweight="bold", ha="center", va="center")

    for row_index, (_, row) in enumerate(top.iterrows()):
        y = len(top) - 1 - row_index
        if row_index % 2 == 0:
            ax.add_patch(FancyBboxPatch(
                (-3.55, y - .42), 12.58, .84,
                boxstyle="round,pad=0.02,rounding_size=.08",
                facecolor=panel, edgecolor="none", zorder=0,
            ))
        playoff_year = int(str(row.SEASON).split("-", maxsplit=1)[0]) + 1
        ax.text(-3.38, y + .12, str(row.PLAYER_NAME), color=ink, fontsize=11.5,
                fontweight="bold", ha="left", va="center")
        ax.text(-3.38, y - .18,
                f"{playoff_year} · {row.TEAM_ABBREVIATION} · with {row.PRIMARY_PLAYER_NAME}",
                color=muted, fontsize=8.2, ha="left", va="center")

        for start, (column, _, color) in zip(starts, layers, strict=True):
            value = float(row[column])
            ax.plot([start, start + track_width], [y - .10, y - .10], color=track,
                    linewidth=6, solid_capstyle="round", zorder=1)
            endpoint = start + track_width * np.clip(value, 0, 100) / 100
            ax.plot([start, endpoint], [y - .10, y - .10], color=color,
                    linewidth=6, solid_capstyle="round", zorder=2)
            ax.scatter(endpoint, y - .10, s=42, color=color, edgecolor=background,
                       linewidth=.7, zorder=3)
            ax.text(start + track_width / 2, y + .19, f"{value:.0f}", color=ink,
                    fontsize=9.5, fontweight="bold", ha="center", va="center")

        ax.text(score_x, y + .10, f"{row.SIMPLE_BALANCED_SCORE:.1f}", color=ink,
                fontsize=15, fontweight="bold", ha="center", va="center")
        ax.text(score_x, y - .21, f"context {row.TOTAL_CONTEXT_ADJUSTMENT:+.1f}",
                color=muted, fontsize=7.5, ha="center", va="center")

    ax.text(-3.55, len(top) + .88, title, color=ink, fontsize=23,
            fontweight="bold", ha="left", va="center")
    ax.text(-3.55, len(top) + .58, subtitle, color=muted, fontsize=10.5,
            ha="left", va="center")
    ax.text(-3.55, -1.00,
            "Every category uses the same 0–100 scale.",
            color=muted, fontsize=8.5, ha="left", va="center")
    fig.subplots_adjust(left=.035, right=.98, top=.96, bottom=.06)
    path = output / filename
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_simple_scorecard(frame: pd.DataFrame, output: Path) -> list[Path]:
    ranking = championship_scorecard(frame)
    csv_path = output / "championship_second_option_scorecard.csv"
    md_path = output / "championship_second_option_scorecard.md"
    ranking.to_csv(csv_path, index=False)
    md_path.write_text(scorecard_markdown(ranking) + "\n")
    chart_path = plot_scorecard(
        ranking,
        output,
        title="Championship Second Options Since 2000",
        subtitle="Production, full-run value, role, fit, and a small postseason adjustment",
        filename="championship_second_option_scorecard.png",
    )

    one_per_player = one_run_per_player_scorecard(frame).head(10)
    one_player_csv = output / "overall_top_10_one_run_per_player.csv"
    one_player_md = output / "overall_top_10_one_run_per_player.md"
    one_per_player.to_csv(one_player_csv, index=False)
    one_player_lines = [
        "# Top 10 Second Options: One Run Per Player",
        "",
        "Each player gets one spot here, using his highest-rated run. The full dataset "
        "still keeps every qualifying postseason.",
        "",
        "| Rank | Run | Finish | #1 star | Core | Context | Final |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for _, row in one_per_player.iterrows():
        one_player_lines.append(
            f"| {int(row.RANK)} | {row.PLAYER_NAME}, {int(str(row.SEASON)[:4]) + 1} "
            f"{row.TEAM_ABBREVIATION} | {row.POSTSEASON_FINISH} | "
            f"{row.PRIMARY_PLAYER_NAME} | {row.SIMPLE_BALANCED_CORE_SCORE:.1f} | "
            f"{row.TOTAL_CONTEXT_ADJUSTMENT:+.1f} | "
            f"{row.SIMPLE_BALANCED_SCORE:.1f} |"
        )
    one_player_md.write_text("\n".join(one_player_lines) + "\n")
    executive_summary = output / "executive_summary.md"
    leader = one_per_player.iloc[0]
    executive_lines = [
        "# The NBA's Top Single-Season Playoff Second-Option Runs Since 2000",
        "",
        f"**2020 Anthony Davis** comes out first at {leader.SIMPLE_BALANCED_SCORE:.1f}. "
        f"His play accounts for a base score of {leader.SIMPLE_BALANCED_CORE_SCORE:.1f}; "
        "the title, opponent path, and Finals performance add "
        f"{leader.TOTAL_CONTEXT_ADJUSTMENT:+.1f}.",
        "",
        "## Final top 10 — one run per player",
        "",
    ]
    for _, row in one_per_player.iterrows():
        executive_lines.append(
            f"{int(row.RANK)}. {int(str(row.SEASON)[:4]) + 1} {row.PLAYER_NAME} "
            f"({row.TEAM_ABBREVIATION}) — {row.SIMPLE_BALANCED_SCORE:.1f}"
        )
    executive_lines.extend([
        "",
        "Most of the score comes from what the player actually did. It "
        "combines era-adjusted production (43%), total value across the run (22%), "
        "role responsibility (15%), and fit beside the primary star (20%). A title "
        "can add at most 3.5 points, while opponent strength and final-round play "
        "can each change the score by no more than 0.5 point.",
        "",
        "A gap of one or two points is not meaningful enough to declare one player "
        "clearly better. The formulas, sources, and limitations are in "
        "`methodology_research_report.md`.",
    ])
    executive_summary.write_text("\n".join(executive_lines) + "\n")
    one_player_scorecard = plot_scorecard(
        one_per_player,
        output,
        title="Best Playoff Second Options Since 2000",
        subtitle="One run per player · Conference Finals or better",
        filename="overall_top_10_one_run_per_player_scorecard.png",
    )
    linkedin_visuals = save_linkedin_visuals(one_per_player, output)
    robustness = ranking_robustness(frame)
    robustness_csv = output / "final_ranking_robustness.csv"
    robustness.to_csv(robustness_csv, index=False)
    robustness_md = output / "final_ranking_robustness.md"
    robustness_lines = [
        "# How Sensitive Is the Ranking?",
        "",
        "I changed the four main weights and reran the ranking 20,000 times. The table "
        "shows how often each run finished first, in the top five, or in the top ten. "
        "These are stress-test results, not statistical confidence intervals.",
        "",
        "| Run | Finishes #1 | Top five | Top 10 |",
        "|---|---:|---:|---:|",
    ]
    for _, row in robustness.head(15).iterrows():
        robustness_lines.append(
            f"| {row.PLAYER_NAME}, {row.SEASON} | "
            f"{row.ROBUSTNESS_FIRST_PROBABILITY:.1%} | "
            f"{row.ROBUSTNESS_TOP_5_PROBABILITY:.1%} | "
            f"{row.ROBUSTNESS_TOP_10_PROBABILITY:.1%} |"
        )
    robustness_md.write_text("\n".join(robustness_lines) + "\n")
    return [
        csv_path,
        md_path,
        chart_path,
        executive_summary,
        one_player_csv,
        one_player_md,
        one_player_scorecard,
        *linkedin_visuals,
        robustness_csv,
        robustness_md,
    ]
