# Final Methodology: Contextual Playoff Second-Option Value

## Final scoring architecture

The ranking separates four core evidence layers from a small postseason-context
adjustment:

| Core layer | Weight |
|---|---:|
| Era-relative rate performance | 43% |
| Cumulative run value | 22% |
| Role-specific responsibility | 15% |
| Coverage-shrunk complementary fit | 20% |

```text
Core = exp(0.43 ln P + 0.22 ln V + 0.15 ln R + 0.20 ln F)

Final = clip(Core + finish adjustment + SRS adjustment
                   + deepest-round adjustment, 0, 100)
```

The context terms are bounded:

- finish: `0.0` for a Conference Finals exit, `+1.5` for a Finals exit,
  `+3.5` for a title;
- opponent path: `−0.5` to `+0.5`, based on opponents' regular-season SRS;
- deepest-round play: `−0.5` to `+0.5`.

Thus context can move a run by at most 4.5 points upward or 1.0 point downward;
it cannot replace what the player actually produced.

## Core components

```text
P = geomean(era percentile of points/75, TS%, BPM)

V = geomean(era percentile of VORP, Win Shares)

role route = max(scoring share, assist share,
                 geomean(rebound share, block share) for bigs)

R = geomean(era percentile of usage, role route,
            deepest-round MPG)

F = coverage × raw compatibility + (1 − coverage) × 50
```

Complementary fit compares the second option's creation, scalable offense
(perimeter gravity or interior gravity), and role-adjusted defense with the #1
star's needs. Shared offensive strengths may amplify each other. Defense helps
only when the primary-star need and secondary supply are modeled; guards do not
receive a universal rim-protection penalty.

## Opponent SRS

Each opponent's regular-season SRS is weighted by games faced and a modest 15%
increase per later series:

```text
series weight = games × [1 + 0.15 × (series order − 1)]
```

The SRS score combines the weighted full path and the strongest opponent, then
converts that result to a nearby-era percentile. Its final contribution is capped
at half a point because SRS is team context, not player production.

## Role and defense safeguards

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

## Why the weights are declared

No ground-truth historical label exists for “best second option.” Training on
titles would produce a circular team-success model; training on expert lists
would reproduce subjective labels. The weights therefore encode explicit
guardrails and are stress-tested with 20,000 plausible alternatives. Sensitivity
frequencies are not confidence intervals, and close runs should be treated as a
tier.

For the complete research basis, formulas, source links, limitations, and case
interpretations, see `methodology_research_report.md`.
