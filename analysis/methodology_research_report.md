# Research-backed methodology for playoff second-option runs

## Executive conclusion

There is no published statistic—and no defensible training label—that yields an
objectively correct set of weights for “best second option.” Championship outcome
is team-dependent, expert rankings are subjective, and single-postseason lineup
samples are too sparse to fit a stable causal model across 2000–2026. The sound
approach is therefore a transparent multi-criteria scorecard: define distinct
constructs, prevent double counting, keep uncertain context bounded, publish the
formula, and stress-test the result across plausible weights.

The final model gives most of the score to what the player actually did, then
adds a small postseason-context adjustment. It produces a result that is both
data-led and basketball-coherent: 2020 Anthony Davis ranks first; 2023 Jamal
Murray ranks above 2020 Murray; 2016 Kyrie Irving and 2004 Shaquille O'Neal rank
above 2020 Murray; and 2010 Pau Gasol appears in the presentation top 10 when
each player is represented by only his best run.

## What the research supports

### Use several evidence types, not one all-in-one statistic

Basketball-Reference defines BPM as a box-score-based estimate of points above
league average per 100 possessions. It explicitly warns that box-score defense
cannot capture positioning, communication, deterrence, and other important work.
BPM is a rate statistic; VORP adds playing time. This supports using BPM as one
rate signal—not the whole answer—and separating rate performance from cumulative
run value. [BPM methodology](https://www.basketball-reference.com/about/bpm2.html)

Win Shares allocates team success to players and is constructed from player,
team, and league inputs. That makes it useful as one cumulative signal but not an
independent truth or a pure individual measure. It is paired with VORP rather
than used alone. Minutes are not added separately because both cumulative
statistics already incorporate playing time. [Win Shares methodology](https://www.basketball-reference.com/about/ws.html)

### Adjust for role and teammate fit, but do not claim causal isolation

Research on regularized adjusted plus-minus finds that even opponent- and
teammate-adjusted metrics retain complementarity effects; basketball players are
not randomly assigned to teammates, so the necessary counterfactuals are largely
unobserved. The model therefore calls its fit layer **compatibility evidence**,
not “chemistry caused by the second option.” [PLOS One study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237920)

Recent lineup-RAPM work also shows why raw lineup ratings are dangerous in short
samples: an average lineup may have only a few dozen possessions, and opponent
quality materially affects the result. That supports excluding raw on/off delta
from the rank and using coverage shrinkage wherever tracking or lineup evidence
is incomplete. [L-RAPM paper](https://arxiv.org/abs/2601.15000)

### Include opponent quality, but only as a modifier

Basketball-Reference defines SRS as a team rating based on average point
differential and strength of schedule, denominated in points above or below
average. It is a reasonable description of the quality of opponents on a playoff
path, but it is still a regular-season team estimate—not a matchup-specific
player statistic. The model therefore limits SRS to one half-point of adjustment
in either direction. [Basketball-Reference glossary](https://www.basketball-reference.com/about/glossary.html)

### Use official possession-aware definitions

The NBA defines TS% as points divided by `2 × (FGA + 0.44 × FTA)` and usage as the
share of team plays a player uses while on court. Usage records possession
endings, not who organizes the offense, which is why it cannot identify the #1
option by itself. [NBA Stats glossary](https://www.nba.com/stats/help/glossary)

### Do not pretend the weights were “discovered” without a target

Penalized basketball models can select tuning parameters by cross-validation
when they have a predictive outcome. This project has no comparable ground-truth
label for historical second-option quality. Training weights to predict titles
would mostly teach the model team strength and then circularly reward players for
winning. The weights below are therefore declared decision weights, and their
uncertainty is tested rather than hidden. [Penalized regression research](https://arxiv.org/abs/1301.3523)

## Population and role assignment

One observation is one player-team-postseason run. The population includes every
team reaching at least the Conference Finals from the 2000 playoffs through the
latest completed season. A player must have at least eight postseason games,
15 minutes per game, and meaningful participation in at least half of the team's
deepest-round games.

The #1 is the structural offensive engine, not automatically the leading scorer
or usage leader. The proposal model combines scoring load, creation, impact, and
minutes. Every proposed #1/#2 pairing is preserved in the audit, and reviewed
historical overrides remain explicit. This is why Nash can be the #1 beside
Stoudemire, Billups beside Hamilton, and Jokić beside Murray.

## Era normalization

All percentile-style inputs use the same postseason comparison population and a
centered nearby-era window. For run `i`, metric `x`, and reference set `R_i`:

```text
p_i(x) = 100 × [count(x_j < x_i) + 0.5 × count(x_j = x_i)] / |R_i|
```

The reference set uses the current season twice and the adjacent seasons once,
which preserves local era context without relying on a single tiny season. The
raw data remain in the output so the percentile can always be audited.

## Component mathematics

All component scores are on a 0–100 percentile-style scale. Geometric means are
used because they reward multi-dimensional strength and prevent one extreme
number from fully compensating for a weak dimension:

```text
G(x1, …, xk) = exp[(ln x1 + … + ln xk) / k]
```

### P — observed rate performance

```text
P = G(percentile(points per 75), percentile(TS%), percentile(BPM))
```

Points per 75 reduces pace distortion. TS% values two-pointers, three-pointers,
and free throws in one efficiency measure. BPM supplies an empirically estimated
overall rate signal, but its defensive portion is not treated as definitive.

### V — cumulative run value

```text
V = G(percentile(VORP), percentile(Win Shares))
```

Both inputs already convert rate production into cumulative postseason value,
including playing time. MPG is intentionally **not** added here because doing so
would count minutes twice and unfairly depress high-impact sixth men such as 2005
Manu Ginóbili. This layer distinguishes a spectacular rate over a short interval
from value sustained through a long postseason. It is why 2010 Gasol's
playoff-leading 4.3 Win Shares matters without making Win Shares the ranking by itself.
[2010 playoff leaders](https://www.basketball-reference.com/playoffs/NBA_2010_leaders.html)

### R — role responsibility

First define the player's strongest legitimate responsibility route:

```text
offensive route = max(percentile(team scoring share),
                      percentile(team assist share))

interior route = G(percentile(team rebound share),
                   percentile(team block share))     # bigs only

role route = max(offensive route, interior route)

R = G(percentile(usage), role route, percentile(deepest-round MPG))
```

The maximum is intentional. A guard does not need to protect the rim, and a big
does not need point-guard assist volume, to shoulder real second-star
responsibility. Interior responsibility is available only to the broad `BIG`
position group; it is not a universal defense bonus.

### F — complementary-fit evidence

Secondary skill supply has three channels:

```text
creation = usage + assist load + decision quality + available iso/late-clock data
scalable offense = max(perimeter off-ball gravity, interior gravity)
defense = role- and nearby-era-adjusted defensive evidence
```

The primary star's needs are estimated from creation burden, shooting/off-ball
capability, and defensive capability. Need fulfillment is:

```text
need_fit = Σ(available need_k × skill_k) / Σ(available need_k)
```

Shared offensive strengths can also amplify each other:

```text
amplification = mean(sqrt(secondary creation × primary creation),
                     sqrt(secondary scalable offense × primary off-ball))

raw_fit = sqrt(need_fit × amplification)
```

Defense is included only where the primary's modeled need and the secondary's
relevant supply support it. Shared defense is not automatically rewarded because
a second rim protector may be less necessary beside an elite defensive big.

Missing historical evidence is shrunk toward neutral rather than set to zero:

```text
F = coverage × raw_fit + (1 − coverage) × 50
```

Tracking fields that do not exist for an era remain `NOT_MODELED`.

## Core score

The final declared core weights are:

| Component | Weight | Reason for scale |
|---|---:|---|
| Observed rate performance | 43% | Largest share; the player's actual postseason remains primary |
| Cumulative run value | 22% | Enough to check short hot streaks and reward full-run availability |
| Role responsibility | 15% | Distinguishes true second-star burden from low-load efficiency |
| Complementary fit | 20% | Large enough for the project's tactical question, below production |

```text
Core = exp(0.43 ln P + 0.22 ln V + 0.15 ln R + 0.20 ln F)
```

These numbers are not fitted coefficients. They satisfy four predeclared
guardrails: production is the plurality; production plus cumulative value is
65%; fit cannot outweigh observed play; and no single contextual judgment can
dominate the result.

## Bounded postseason context

### Opponent SRS path

For each series `s`, the opponent's regular-season SRS is weighted by games faced
and a modest later-round multiplier:

```text
w_s = games_s × [1 + 0.15 × (series_order_s − 1)]

path_SRS = Σ(w_s × opponent_SRS_s) / Σw_s

S = G(percentile(path_SRS), percentile(max opponent SRS)))

A_srs = clip[0.5 × (S − 50) / 50, −0.5, +0.5]
```

This gives credit for a difficult path but cannot move a run by more than half a
point.

### Deepest-round performance

```text
T = G(percentile(deepest-round PPG), percentile(deepest-round TS%),
      percentile(deepest-round MPG), percentile(team scoring share))

A_terminal = clip[0.5 × (percentile(T) − 50) / 50, −0.5, +0.5]
```

This is deliberately small because one series is a noisy sample.

### Run completion

```text
A_finish = 0.0  if Conference Finals exit
           1.5  if Finals exit
           3.5  if champion
```

The championship adjustment is meaningful but bounded: a conference-finalist
with a core score more than 3.5 points better still ranks ahead of a champion.
It rewards completion of the run without making the score a ring count.

### Final score

```text
Final = clip(Core + A_finish + A_srs + A_terminal, 0, 100)
```

The total contextual movement ranges from `−1.0` to `+4.5` points. At least 95.5%
of the possible 100-point scale therefore comes from the player's production,
cumulative value, responsibility, and fit.

## What is intentionally not in the final score

- Raw on/off or pair net rating: too sensitive to teammates, opponents, and
  deployment; retained only as descriptive evidence.
- Awards, reputation, or manually assigned “clutch” points.
- A universal two-way bonus that would demand rim protection from guards.
- Invented pre-tracking iso, late-clock, matchup, or rim-deterrence data.
- A trained title-prediction model, because that would answer “which team won?”
  rather than “how good was this second option?”

## How to read the controversial cases

- **2020 Anthony Davis:** No. 1. His 27.7 PPG, 66.5% TS, 8.7 BPM, 4.5 Win Shares,
  elite finishing, interior defense, and fit beside LeBron create the strongest
  core score. His easier SRS path costs only 0.38 points, not an entire domain.
- **2023 vs. 2020 Murray:** 2020 has the better shooting/scoring rate; 2023 has
  the stronger responsibility/fit profile and completed a title run. The final
  contextual model ranks 2023 higher.
- **2016 Kyrie vs. 2020 Murray:** Their cores are close enough that the title,
  deepest-round performance, and stronger SRS path move Kyrie ahead.
- **2004 Shaq:** BPM does not directly punish him for making no threes. His rate
  and cumulative scores are elite; his Finals run and difficult path move him
  above 2020 Murray.
- **2010 Pau Gasol:** His 4.3 Win Shares, interior responsibility, title, and
  cumulative workload place him 10th in the one-run-per-player presentation.
- **2011 Wade:** His rate performance remains elite, but the bounded system no
  longer lets a losing Finals run rank second solely from box-score dominance.

## Uncertainty and reproducibility

The pipeline draws 20,000 alternative core-weight combinations from documented
ranges and re-ranks every eligible run with the same bounded context rules. These
are **sensitivity frequencies**, not statistical confidence intervals. Close
scores should be described as tiers, not as proof that No. 7 is meaningfully
better than No. 8.

Every SRS table, NBA Stats response, and Basketball-Reference advanced table is
cached with provenance. Missing fields are tagged `NOT_MODELED`; failures are not
silently converted to zeros.
