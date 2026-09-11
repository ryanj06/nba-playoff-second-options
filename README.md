# NBA Playoff Second-Option Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/)
[![CI](https://github.com/ryanj06/nba-playoff-second-options/actions/workflows/ci.yml/badge.svg)](https://github.com/ryanj06/nba-playoff-second-options/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project compares individual playoff runs by NBA second options from 2000
onward. It includes every team that reached at least the Conference Finals.

Instead of assuming that the second-leading scorer was automatically the second
option, the pipeline considers who created the offense, who carried the scoring
load, and how the team actually used each player. Every #1/#2 pairing is saved in
an audit table so the judgment can be reviewed. Players also need at least eight
games, 15 minutes per game, and a meaningful role in their team's final series.

## Headline result

The model ranks **2020 Anthony Davis** as the strongest single-postseason second
option since 2000. In the presentation view, each player appears only once:

1. 2020 Anthony Davis
2. 2017 Stephen Curry
3. 2001 Kobe Bryant
4. 2026 OG Anunoby
5. 2023 Jamal Murray
6. 2005 Manu Ginobili
7. 2011 Dwyane Wade
8. 2016 Kyrie Irving
9. 2004 Shaquille O'Neal
10. 2010 Pau Gasol

The analytical table still retains every player-season observation. The
one-run-per-player rule is a presentation choice, not a hidden scoring change.

![Four-layer top-ten scorecard](analysis/overall_top_10_one_run_per_player_scorecard.png)

The repository checks in only the final, recruiter-facing analysis bundle. A
pipeline run may create additional diagnostics locally, but superseded drafts,
duplicate formats, and intermediate rankings are ignored by Git.

## Project structure

```text
nba_second_options_single_season.py  # thin command-line entry point
nba_second_options/
  config.py       # configuration, statuses, and domain errors
  era.py          # season environments and rolling era normalization
  sources.py      # source adapters, retry behavior, and cache
  contextual_ingestion.py # resumable playoff play-by-play/rotation caching
  stints.py        # exact constant-lineup intervals and observed score changes
  metrics.py      # qualification and feature-engineering functions
  role_audit.py   # model proposals, reviewed decisions, and audit export
  contextual_value.py # validated role-matched replacement-value estimator
  contextual_spec.py  # locked NBA feature contract and missing-data rules
  sensitivity.py  # six-domain championship robustness simulation
  simple_scorecard.py # final four-layer ranking, bounded context, robustness test
  headshots.py    # season-matched NBA portrait cache with dated fallbacks
  editorial_visuals.py # social-ready headshot leaderboard and context map
  pipeline.py     # season and full-history orchestration
  reporting.py    # charts, reports, tables, and run manifest
```

Network access is isolated from analytical calculations, making methodology
decisions independently testable and easy to review.

## Installation

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For an editable development install with tests and linting:

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## Run

From this directory:

```bash
python nba_second_options_single_season.py \
  --start-season 1999-00 \
  --output-dir analysis_outputs
```

Validate rendering without downloading data:

```bash
python nba_second_options_single_season.py --demo --output-dir demo_outputs
```

Replay a completed download without network access:

```bash
python nba_second_options_single_season.py \
  --offline \
  --cache-dir data/cache \
  --output-dir analysis_outputs
```

Cache the official game-level inputs needed for the contextual model. This is a
large, resumable pull (two cached feeds per playoff game), so run a narrow season
first and retain the cache:

```bash
python nba_second_options_single_season.py \
  --start-season 2023-24 \
  --end-season 2023-24 \
  --cache-contextual-games \
  --cache-dir data/cache \
  --output-dir analysis_outputs
```

Use `--strict-metrics` to suppress every composite score that lacks one or more
required components. In standard mode, partial composites remain visible but
are explicitly marked `PARTIAL` and include a coverage fraction.

## How the ranking works

The main ranking is a transparent scorecard. It is separate from the experimental
replacement-value model and from the championship-only sensitivity analysis.

The optional replacement-value model is evaluated chronologically: it learns from
earlier seasons and is tested on later ones. It only produces a ranking if it
beats a simple historical-average baseline. Its comparison holds the primary star
and team needs constant, then replaces the second option with an era- and
role-matched alternative.

The headline result uses the simpler scorecard. It combines playoff production
(43%), total value across the run (22%), role responsibility (15%), and fit with
the primary star (20%). A small adjustment accounts for how far the team went,
the strength of its opponents, and the player's performance in the final series.
The weights are tested across thousands of alternatives; they are not presented
as the only reasonable answer.

- **Qualifier:** A team appearing in playoff round 3, cross-checked by completed
  series victories. Four teams must qualify.
- **Role identification:** The primary proposal emphasizes scoring load,
  creation-engine responsibility, PIE, and minutes. After the primary is
  removed, the secondary proposal emphasizes scoring responsibility. The audit
  preserves `MODEL_PRIMARY`/`MODEL_SECONDARY` beside the reviewed final labels.
- **TS%:** `PTS / (2 × (FGA + 0.44 × FTA))`.
- **BPM:** Canonical postseason BPM 2.0 scraped from Basketball-Reference. NBA
  raw plus-minus is never relabeled as BPM.
- **Stabilized 3PT%:** Beta-binomial empirical-Bayes posterior using a
  season-specific league prior.
- **Era context:** Points per 75, TS%, usage, creation, and three-point volume
  are evaluated against all playoff rotation players from the same season.
  Tracking-based skills use a centered, current-season-weighted three-year window.
- **Complete-run eligibility:** At least half of the terminal-series games at
  15+ MPG. This prevents an injury-truncated aggregate from representing the
  complete Finals or Conference Finals run.
- **In-era production:** Equal-domain geometric mean of full-playoff-population
  BPM, points per 75, TS%, and usage percentiles.
- **Need fulfillment:** Skill supply is weighted by the #1 star's modeled needs
  and normalized by total modeled need; versatile stars no longer mechanically
  suppress every fit score.
- **Strength amplification:** Geometric interactions reward shared creation,
  and gravity strengths. Defensive overlap is not automatically rewarded because
  a second rim protector can be redundant beside an elite defensive big.
- **Role-aware defense:** Secondary defensive supply is compared within broad
  position groups over a nearby-era window before entering need fulfillment.
- **Role compatibility:** Geometric mean of need fulfillment and strength
  amplification, with component-level coverage and `PARTIAL` flags.
- **Final leaderboard core:** 43% rate performance, 22% cumulative run value
  from VORP and Win Shares (which already incorporate playing time),
  15% role responsibility, and 20% coverage-shrunk complementary fit. Production
  remains the largest domain. Missing fit evidence is pulled toward a neutral 50
  in proportion to missing coverage rather than treated as zero.
- **Bounded postseason context:** Conference Finals/Finals/title completion adds
  0/1.5/3.5 points; opponent-SRS path and deepest-round play each move a run by at
  most 0.5 point. Opponent SRS is weighted by games faced and a modest later-round
  multiplier. Context cannot replace the player's core performance.
- **Holistic championship scorecard:** Six separate domains preserve the
  distinction between rate production, role burden, cumulative impact,
  terminal-series responsibility, historical defensive evidence, and
  primary-star compatibility. Championship conclusions use 50,000 uniform
  Dirichlet weight draws instead of presenting one arbitrary 70/30 split.
- **Defensive evidence:** Defense is not a position-blind standalone percentage
  of the final score. Total BPM already contains defensive information, and
  defense also enters complementary fit when the primary-star need and available
  skill evidence support it. DBPM, defensive win shares, and stocks remain visible
  diagnostics—not another independent vote. Modern rim/tracking fields and
  verified matchup assignments remain `NOT_MODELED` when unavailable.
- **Position-aware scalability:** Perimeter spacing is position-relative and is
  shown beside interior gravity, so bigs are not evaluated as if they were guards.
- **Lineup interaction:** Four-state difference-in-differences across both,
  #1-only, #2-only, and neither lineups. It is labeled
  `CONTEXTUAL_NOT_CAUSAL` and is excluded from the ranking.

## Data-source policy

NBA Stats via `nba_api` is primary. Basketball-Reference supplies BPM, VORP,
Win Shares, and regular-season team SRS.
Every response is cached with provenance metadata. The code uses slow requests,
normal retries, and no anti-bot circumvention. If a public page is unavailable,
the metric is recorded as `NOT_MODELED` unless a valid cached response exists.

Historical play-type, shot-clock, catch-and-shoot, and rim-defense coverage is
uneven. Missing tracking is never filled with zero or an invented estimate.
The pipeline begins tracking requests at 2013-14, the NBA optical-tracking era;
earlier runs are explicitly marked `NOT_MODELED` for those fields.

## Responsible use and limitations

This is an explanatory multi-criteria scorecard, not a causal player-impact
estimate. Close scores should be interpreted as tiers. Role decisions are
published in `analysis/role_pairing_audit.csv`; missing tracking evidence is
marked `NOT_MODELED`; raw lineup on/off is never presented as isolated player
value. Source terms and image rights remain with their respective owners.
The `--demo` command uses synthetic data only and must not be presented as NBA
research.
