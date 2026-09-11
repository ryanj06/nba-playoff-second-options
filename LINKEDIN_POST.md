# LinkedIn Post

Who had the best playoff run as an NBA second option since 2000?

I built a Python project to look beyond points per game. The dataset covers 108
second-option runs by teams that reached at least the Conference Finals.

The main question was not simply, “Who scored the most?” It was, “How much did
the player produce, how much responsibility did he carry, and how well did his
game complement the team’s primary star?”

The model combines four parts:

- playoff production adjusted for era: 43%;
- total value across the full run: 22%;
- scoring, playmaking, or interior responsibility: 15%;
- fit with the primary star: 20%.

Championship progress, opponent strength, and performance in the final series
matter, but their influence is capped so team success cannot overwhelm individual
performance. Missing historical tracking data is marked `NOT_MODELED` instead of
being filled with made-up estimates.

For the final graphic, I used one run per player:

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

Anthony Davis finished first with 27.7 points per game on 66.5% true shooting,
an 8.7 BPM, and 4.5 Win Shares across 21 games. He supplied elite scoring and
rim protection without taking primary creation duties away from LeBron James.

One interesting result: 2020 Jamal Murray still grades as an outstanding run in
the full player-season table, but 2023 is his stronger overall run and represents
him in the one-run-per-player list. That keeps the public ranking focused on ten
different players without changing the underlying data.

The biggest lesson from the project was that basketball context cannot be reduced
to one advanced statistic. A useful model should make its assumptions visible,
show what is missing, and let readers disagree with the weights.

Code, methodology, role audit, data, and reproducible charts:
https://github.com/ryanj06/nba-playoff-second-options

#NBA #BasketballAnalytics #SportsAnalytics #Python #DataScience

## Recommended image

`analysis/linkedin_top_10_one_run_per_player.png`
