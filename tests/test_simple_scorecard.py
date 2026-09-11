import numpy as np
import pandas as pd

from nba_second_options.simple_scorecard import (
    championship_scorecard,
    geometric_mean,
    weighted_geometric_mean,
)


def test_geometric_mean_requires_complete_layers():
    frame = pd.DataFrame({"a": [100.0, 50.0], "b": [25.0, np.nan]})
    result = geometric_mean(frame)
    assert np.isclose(result.iloc[0], 50)
    assert np.isnan(result.iloc[1])


def test_weighted_geometric_mean_uses_declared_domain_weights():
    frame = pd.DataFrame({"a": [100.0], "b": [25.0]})
    result = weighted_geometric_mean(frame, {"a": .75, "b": .25})
    assert np.isclose(result.iloc[0], 100 ** .75 * 25 ** .25)


def test_championship_scorecard_filters_ineligible_nonchampions(monkeypatch):
    frame = pd.DataFrame({
        "PLAYER_NAME": ["A", "B", "C"],
        "SEASON": ["2020-21"] * 3,
        "TEAM_ABBREVIATION": ["AAA", "BBB", "CCC"],
        "PRIMARY_PLAYER_NAME": ["P1", "P2", "P3"],
        "RANKING_ELIGIBLE": [True, True, False],
        "POSTSEASON_FINISH": ["Champion", "Finals", "Champion"],
        "ERA_PTS75_PERCENTILE": [90, 95, 100],
        "ERA_TS_PERCENTILE": [90, 95, 100],
        "ERA_BPM_PERCENTILE": [90, 95, 100],
        "ERA_USG_PERCENTILE": [90, 95, 100],
        "TEAM_PTS_SHARE": [.25, .26, .27],
        "DEEPEST_ROUND_TEAM_PTS_SHARE": [.25, .26, .27],
        "HISTORICAL_DEFENSE_EVIDENCE_SCORE": [90, 95, 100],
        "CUMULATIVE_IMPACT_SCORE": [90, 95, 100],
        "ROLE_COMPATIBILITY_SCORE": [90, 95, 100],
        "COMPLEMENT_FIT_COVERAGE": [1.0, 1.0, 1.0],
    })
    monkeypatch.setattr(
        "nba_second_options.simple_scorecard.rolling_era_percentile",
        lambda data, column: pd.Series([90.0, 95.0, 100.0], index=data.index),
    )
    result = championship_scorecard(frame)
    assert result.PLAYER_NAME.tolist() == ["A"]
    assert result.RANK.tolist() == [1]
