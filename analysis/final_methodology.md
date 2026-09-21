# How the Ranking Works

## The score

Four parts make up the base score. Postseason context is added afterward and kept
small on purpose.

| Part | Weight |
|---|---:|
| Era-relative rate performance | 43% |
| Cumulative run value | 22% |
| Role-specific responsibility | 15% |
| Fit beside the primary star | 20% |

```text
Core = exp(0.43 ln P + 0.22 ln V + 0.15 ln R + 0.20 ln F)

Final = clip(Core + finish adjustment + SRS adjustment
                   + deepest-round adjustment, 0, 100)
```

The postseason adjustments are:

- finish: `0.0` for a Conference Finals exit, `+1.5` for a Finals exit,
  `+3.5` for a title;
- opponent path: `−0.5` to `+0.5`, based on opponents' regular-season SRS;
- deepest-round play: `−0.5` to `+0.5`.

Put together, context can add no more than 4.5 points or subtract no more than one point.
The player's own production still drives the result.

## What goes into each part

```text
P = geomean(era percentile of points/75, TS%, BPM)

V = geomean(era percentile of VORP, Win Shares)

role route = max(scoring share, assist share,
                 geomean(rebound share, block share) for bigs)

R = geomean(era percentile of usage, role route,
            deepest-round MPG)

F = coverage × raw compatibility + (1 − coverage) × 50
```

The fit score compares the second option's creation, scalable offense
(perimeter gravity or interior gravity), and role-adjusted defense with the #1
star's needs. Shared offensive strengths may amplify each other. Defense helps
only when the primary-star need and secondary supply are modeled; guards do not
receive a universal rim-protection penalty.

## Opponent strength

Each opponent's regular-season SRS is weighted by games faced and a modest 15%
increase per later series:

```text
series weight = games × [1 + 0.15 × (series order − 1)]
```

I combine the full path with the strongest opponent, then compare that number
with nearby seasons. SRS can only move the final score by half a point because it
describes the path, not the player's performance.

## Rules for role and defense

- The #1 is the structural offensive engine, not automatically the top scorer or
  usage leader; all pairings are preserved in the role audit.
- Bigs can carry responsibility through rebounding/rim protection instead of
  guard-style assist volume.
- BPM defense is treated cautiously; historical positioning, deterrence, and
  matchup information are not invented.
- Missing tracking evidence is `NOT_MODELED` and fit is shrunk toward neutral in
  proportion to coverage.
- Raw lineup on/off is descriptive only and excluded from the rank.
- VORP and Win Shares already include playing time, so MPG is not counted again
  inside cumulative value. Deepest-round MPG appears only in role responsibility.

## Why I chose the weights this way

There is no accepted historical answer I can train against. If I trained the
model on titles, it would mostly learn which teams won. If I trained it on media
lists, it would inherit those opinions. I chose the weights openly and reran the
ranking 20,000 times across reasonable alternatives. Those results show weight
sensitivity, not statistical confidence. Close scores belong in the same tier.

The longer version, with sources and case-by-case notes, is in
`methodology_research_report.md`.
