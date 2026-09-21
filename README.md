# NBA Playoff Second-Option Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/)
[![CI](https://github.com/ryanj06/nba-playoff-second-options/actions/workflows/ci.yml/badge.svg)](https://github.com/ryanj06/nba-playoff-second-options/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

I built this project because most conversations about great second options end up
being lists of points per game. That misses the interesting part. The best No. 2s
did not all have the same job: some created shots when the star was trapped, some
stretched the floor, and others covered the biggest holes on defense.

The dataset includes every team that reached at least the Conference Finals since
2000. I assigned the roles by looking at how each offense actually worked—not by
automatically calling the second-leading scorer the No. 2. Every pairing is in
the role audit so the judgment calls are easy to check. Players also had to appear
in at least eight playoff games, average 15 minutes, and play a meaningful role in
their team's final series.

## Headline result

**2020 Anthony Davis** finishes first. I limited the public list to one run per
player so it does not become three versions of the same star:

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

The full data still keeps every qualifying run. The one-run limit only applies to
the graphic and top-ten list.

![Four-layer top-ten scorecard](analysis/overall_top_10_one_run_per_player_scorecard.png)

## Key visuals

The tactical-fit chart shows what kind of problem each player solved beside the
primary star. It separates secondary creation, scalable gravity, and
need-specific defensive cover. It is a breakdown of playing style, not another
ranking.

![Tactical fit mix](analysis/linkedin_tactical_fit_mix.png)

The opponent chart compares each player's full playoff path with the strongest
team he faced. Competition is part of the story, but it only has a small effect
on the final score.

![Opponent SRS paths](analysis/linkedin_opponent_srs_paths.png)

The repo keeps the final analysis and leaves out draft charts, duplicate exports,
and local photo files.

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
  simple_scorecard.py # final four-part ranking, small context adjustments, weight tests
  visuals.py      # rights-safe LinkedIn and portfolio graphics
  pipeline.py     # season and full-history orchestration
  reporting.py    # charts, reports, tables, and run manifest
```

Data collection is kept separate from the calculations, which makes the model
easier to test, review, and reproduce.

## Installation

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
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

The published ranking uses the four-part score below. The experimental
replacement-value model and championship sensitivity test are kept separate.

The optional replacement-value model trains on earlier seasons and is tested on
later ones. I only use its ranking if it beats a historical-average baseline. In
each comparison, the primary star and team needs stay fixed while the No. 2 is
replaced by a similar player from the same general era and role.

The main result comes from the simpler scorecard. It combines playoff production
(43%), total value across the run (22%), role responsibility (15%), and fit with
the primary star (20%). A small adjustment accounts for how far the team went,
the strength of its opponents, and the player's performance in the final series.
I also rerun the model across thousands of alternative weights to see which
results hold up and which ones depend on a specific choice.

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
- **Strength amplification:** Geometric interactions reward shared creation
  and gravity strengths. Defensive overlap is not automatically rewarded because
  a second rim protector can be redundant beside an elite defensive big.
- **Role-aware defense:** Secondary defensive supply is compared within broad
  position groups over a nearby-era window before entering need fulfillment.
- **Role compatibility:** Geometric mean of need fulfillment and strength
  amplification, with component-level coverage and `PARTIAL` flags.
- **Final leaderboard core:** 43% rate performance, 22% cumulative run value
  from VORP and Win Shares (which already incorporate playing time),
  15% role responsibility, and 20% fit beside the primary star. Production
  remains the largest part. When fit data is missing, that portion moves toward
  a neutral 50 instead of being treated as zero.
- **Bounded postseason context:** Conference Finals/Finals/title completion adds
  0/1.5/3.5 points; opponent-SRS path and deepest-round play each move a run by at
  most 0.5 point. Opponent SRS is weighted by games faced and a modest later-round
  multiplier. Context cannot replace the player's core performance.
- **Championship sensitivity check:** Six separate domains preserve the
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

Most of the data comes from NBA Stats through `nba_api`. I use
Basketball-Reference for BPM, VORP, Win Shares, and team SRS. The downloads are
cached, along with the source and retrieval details, so the analysis can be
reproduced without hitting the same pages every time.

Some of the more detailed stats simply do not exist for older playoff runs. The
NBA's tracking era began in 2013-14, and even the early tracking seasons have
gaps in play-type, shot-clock, catch-and-shoot, and rim-defense data. When a
number is unavailable, I label it `NOT_MODELED`. I do not replace it with zero or
make up an estimate from an unrelated box-score stat.

## Responsible use and limitations

This ranking compares evidence; it does not isolate a player's causal impact.
Treat close scores as ties or tiers. The role calls are in
`analysis/role_pairing_audit.csv`, missing tracking is labeled `NOT_MODELED`, and
raw lineup on/off never stands in for individual value. Player photos stay out
of the public repo because I am not claiming redistribution rights. The `--demo`
command uses fake data and is only for testing the pipeline.
