# How I Built the Second-Option Ranking

## The basic problem

There is no official “second-option value” stat, and there is no clean answer to
train a model against. Titles depend on the whole team, expert lists reflect the
author's preferences, and one playoff run does not provide enough lineup data to
isolate chemistry. I ended up using a scorecard: keep the major ideas separate,
avoid counting the same evidence twice, publish the formula, and check how much
the result changes when the weights move.

Most of the score comes from the player's own postseason. Team result, opponent
strength, and play in the final series make small adjustments. With that setup,
2020 Anthony Davis finishes first. The model also puts 2023 Murray over 2020
Murray, moves 2016 Kyrie and 2004 Shaq ahead of the bubble run, and places 2010
Pau Gasol in the one-run-per-player top ten.

## What I took from the research

### No single metric can do the whole job

Basketball-Reference defines BPM as a box-score estimate of points above league
average per 100 possessions. Its documentation notes that box-score defense
misses positioning, communication, and deterrence. I use BPM as one rate signal,
not the final verdict. VORP goes in a separate full-run bucket because it adds
playing time. [BPM methodology](https://www.basketball-reference.com/about/bpm2.html)

Win Shares uses player, team, and league inputs to divide team success among the
roster. I pair it with VORP rather than treating it as an independent truth.
Minutes are not added again because both statistics already account for playing
time. [Win Shares methodology](https://www.basketball-reference.com/about/ws.html)

### Fit matters, but this does not isolate chemistry

Even adjusted plus-minus retains teammate and role effects because players are
not randomly assigned to lineups. We never observe the clean counterfactual—what
the same team would have done with a different No. 2 in the same possessions. I
therefore call this part **fit evidence**, not isolated chemistry.
[PLOS One study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237920)

Lineup RAPM research points to the same problem: many playoff lineups only play a
few dozen possessions, and opponent quality changes the result. Raw on/off stays
in the dataset as context, but it does not enter the ranking. Missing fit evidence
is pulled toward neutral instead of being treated as fact.
[L-RAPM paper](https://arxiv.org/abs/2601.15000)

### Opponent quality belongs in the margins

SRS measures team quality using point differential and schedule strength. It is
useful for describing a playoff path, but it is still a regular-season team
number—not a player matchup grade. I cap its effect at half a point in either
direction. [Basketball-Reference glossary](https://www.basketball-reference.com/about/glossary.html)

### Usage does not tell us who ran the team

The NBA defines TS% as points divided by `2 × (FGA + 0.44 × FTA)`. Usage measures
how many possessions a player finishes while on the floor; it does not tell us
who organized the offense. That is why the role audit uses more than usage alone.
[NBA Stats glossary](https://www.nba.com/stats/help/glossary)

### The weights are choices, not discoveries

Predictive models can tune parameters when they have a real target. This project
does not. Training on titles would mostly teach team strength and then reward
players for the outcome used as the label. I chose the weights directly and test
them across a wide range instead of presenting them as fitted coefficients.
[Penalized regression research](https://arxiv.org/abs/1301.3523)

## Who qualifies and how roles are assigned

Each row is one player, one team, and one postseason. The dataset includes every
team reaching at least the Conference Finals from the 2000 playoffs through the
latest completed season. A player must have at least eight postseason games,
15 minutes per game, and meaningful participation in at least half of the team's
deepest-round games.

The No. 1 is the player the offense is built around, not automatically the top
scorer or usage leader. The first pass combines scoring load, creation, impact,
and minutes. Every proposed pairing stays in the audit, including manual
corrections. That is why Nash can be the No. 1 beside Stoudemire, Billups beside
Hamilton, and Jokić beside Murray.

## Comparing different eras

All percentile inputs use the same postseason comparison group and a
centered nearby-era window. For run `i`, metric `x`, and reference set `R_i`:

```text
p_i(x) = 100 × [count(x_j < x_i) + 0.5 × count(x_j = x_i)] / |R_i|
```

The reference set uses the current season twice and the adjacent seasons once.
That keeps the comparison close to the player's era without relying on one small
playoff sample. The raw numbers remain in the output.

## The math

Every component uses a 0–100 percentile scale. I use geometric means so one huge
number cannot completely cover for a weak part of the profile:

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

Both inputs already account for playing time, so I do not add MPG again. That
would count minutes twice and work against high-impact bench players such as 2005
Manu Ginóbili. This section separates a short hot streak from value sustained
over a full run. It also gives 2010 Gasol credit for leading the playoffs with
4.3 Win Shares without turning the list into a Win Shares ranking.
[2010 playoff leaders](https://www.basketball-reference.com/playoffs/NBA_2010_leaders.html)

### R — role responsibility

Responsibility can come through different jobs:

```text
offensive route = max(percentile(team scoring share),
                      percentile(team assist share))

interior route = G(percentile(team rebound share),
                   percentile(team block share))     # bigs only

role route = max(offensive route, interior route)

R = G(percentile(usage), role route, percentile(deepest-round MPG))
```

I take the strongest valid route. A guard should not lose points for failing to
protect the rim, and a big should not need point-guard assist numbers to carry a
major role. Only the broad `BIG` group can use the interior route; this is not a
blanket defense bonus.

### F — fit beside the primary star

The No. 2 can fill three types of need:

```text
creation = usage + assist load + decision quality + available iso/late-clock data
scalable offense = max(perimeter off-ball gravity, interior gravity)
defense = role- and nearby-era-adjusted defensive evidence
```

I estimate the primary star's needs from creation burden, shooting and off-ball
value, and defense. The fit calculation is:

```text
need_fit = Σ(available need_k × skill_k) / Σ(available need_k)
```

Two offensive strengths can also work together:

```text
amplification = mean(sqrt(secondary creation × primary creation),
                     sqrt(secondary scalable offense × primary off-ball))

raw_fit = sqrt(need_fit × amplification)
```

Defense only helps this section when it fills a modeled need. Two good defenders
are not automatically treated as a perfect fit; another rim protector may add
less beside an elite defensive big than he would beside a weak one.

When evidence is missing, the fit score moves toward neutral rather than zero:

```text
F = coverage × raw_fit + (1 − coverage) × 50
```

Tracking fields that do not exist for an era remain `NOT_MODELED`.

## The base score

These are the weights I use:

| Component | Weight | Why |
|---|---:|---|
| Observed rate performance | 43% | The player's actual postseason should matter most |
| Cumulative run value | 22% | Rewards value sustained across the run |
| Role responsibility | 15% | Separates major second-star work from low-load efficiency |
| Fit beside the primary star | 20% | Enough to matter without outweighing production |

```text
Core = exp(0.43 ln P + 0.22 ln V + 0.15 ln R + 0.20 ln F)
```

These are judgment weights, not fitted coefficients. Production is the largest
piece, production plus full-run value equals 65%, fit cannot outweigh observed
play, and no one context adjustment can take over the ranking.

## Small postseason adjustments

### Opponent SRS path

For each series, I weight the opponent's regular-season SRS by games faced and
give later rounds a small bump:

```text
w_s = games_s × [1 + 0.15 × (series_order_s − 1)]

path_SRS = Σ(w_s × opponent_SRS_s) / Σw_s

S = G(percentile(path_SRS), percentile(max opponent SRS)))

A_srs = clip[0.5 × (S − 50) / 50, −0.5, +0.5]
```

This gives some credit for a hard path, with a half-point cap.

### Deepest-round performance

```text
T = G(percentile(deepest-round PPG), percentile(deepest-round TS%),
      percentile(deepest-round MPG), percentile(team scoring share))

A_terminal = clip[0.5 × (percentile(T) − 50) / 50, −0.5, +0.5]
```

One series is noisy, so this adjustment is also capped at half a point.

### Run completion

```text
A_finish = 0.0  if Conference Finals exit
           1.5  if Finals exit
           3.5  if champion
```

A title matters, but it does not erase the base score. A Conference Finalist with
a base score more than 3.5 points higher still finishes ahead of a champion.

### Final score

```text
Final = clip(Core + A_finish + A_srs + A_terminal, 0, 100)
```

Context can move the score from `−1.0` to `+4.5` points. At least 95.5 points of
the 100-point scale still come from production, full-run value, role, and fit.

## What I chose not to score

- **Raw on/off and pair net rating:** These numbers swing with bench units,
  matchups, and coaching decisions. I keep them for context, but they do not
  affect the ranking.
- **Awards, reputation, and a made-up clutch bonus:** The score is based on what
  happened during that playoff run, not the player's résumé or a subjective
  label.
- **A generic two-way bonus:** Defense matters when it fits the player's role and
  fills a need beside the No. 1. The model does not expect a small guard to
  provide the same defensive value as a rim-protecting big.
- **Estimated tracking stats for older seasons:** If iso, late-clock, matchup, or
  rim-deterrence data was never tracked, I leave it missing instead of trying to
  recreate it from the box score.
- **A model trained to predict champions:** That would mostly learn which teams
  were strongest. This project is trying to evaluate the second option's run,
  not predict the series winner.

## The cases people will probably ask about

- **2020 Anthony Davis:** He is No. 1 because 27.7 PPG, 66.5% TS, an 8.7 BPM, 4.5
  Win Shares, elite finishing, and interior defense add up to the best base
  profile. His easier opponent path costs 0.38 points.
- **2023 Jamal Murray:** His 2020 run had the hotter shooting numbers, but 2023
  came with more playmaking responsibility, a better fit score, and a title. I
  use 2023 as his representative run.
- **2016 Kyrie Irving:** His scoring and shot creation held up through the
  Finals, and he faced the toughest opponent path in the public top ten. That
  combination puts the run eighth overall.
- **2004 Shaquille O'Neal:** BPM does not punish him for making no threes. His
  scoring rate, efficiency, cumulative value, and difficult Finals path still
  produce a top-ten profile.
- **2010 Pau Gasol:** His 4.3 Win Shares, interior responsibility, title, and
  cumulative workload place him 10th in the one-run-per-player presentation.
- **2011 Dwyane Wade:** His individual numbers remain elite, but the smaller
  team-context bonus keeps a Finals loss from jumping to No. 2 on box-score
  dominance alone.

## How stable is the list?

I rerun the list 20,000 times using different weight combinations from the ranges
in the methodology. The percentages are **sensitivity frequencies**, not
confidence intervals. If two players are close, I treat them as the same tier.

The SRS tables, NBA Stats responses, and Basketball-Reference tables are cached
with source metadata. Missing fields stay `NOT_MODELED`; they never become zero.
