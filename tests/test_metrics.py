import numpy as np
import pandas as pd

from nba_second_options.config import COMPLETE, NOT_MODELED, PipelineConfig, seasons_between
from nba_second_options.era import attach_season_environment, rolling_era_percentile
from nba_second_options.metrics import (attach_optional_metrics, beta_binomial_shrinkage,
                                        confidence_label, deepest_round_availability,
                                        match_bpm, normalized_name, opponent_srs_context,
                                        pairing_synergy,
                                        role_adjusted_era_percentile, select_options,
                                        true_shooting)
from nba_second_options.sensitivity import championship_weight_sensitivity


def test_season_generation():
    assert seasons_between("1999-00", "2001-02") == ["1999-00", "2000-01", "2001-02"]


def test_true_shooting_from_totals():
    result = true_shooting(pd.Series([30]), pd.Series([20]), pd.Series([10]))
    assert np.isclose(result.iloc[0], 30 / (2 * (20 + 4.4)))


def test_opponent_srs_context_weights_later_series_modestly_more():
    games = pd.DataFrame([
        {"TEAM_ID": 1, "GAME_ID": "1", "GAME_DATE": "2020-08-01",
         "MATCHUP": "AAA vs. BBB"},
        {"TEAM_ID": 1, "GAME_ID": "2", "GAME_DATE": "2020-08-03",
         "MATCHUP": "AAA @ BBB"},
        {"TEAM_ID": 1, "GAME_ID": "3", "GAME_DATE": "2020-08-10",
         "MATCHUP": "AAA vs. CCC"},
        {"TEAM_ID": 1, "GAME_ID": "4", "GAME_DATE": "2020-08-12",
         "MATCHUP": "AAA @ CCC"},
        {"TEAM_ID": 1, "GAME_ID": "5", "GAME_DATE": "2020-08-14",
         "MATCHUP": "AAA vs. CCC"},
    ])
    srs = pd.DataFrame({
        "TEAM_ABBREVIATION": ["BBB", "CCC"],
        "SRS": [2.0, 8.0],
    })
    result = opponent_srs_context(games, srs, pd.DataFrame([{"TEAM_ID": 1}])).iloc[0]
    expected = (2 * 2.0 + 3 * 1.15 * 8.0) / (2 + 3 * 1.15)
    assert np.isclose(result.OPPONENT_SRS_WEIGHTED, expected)
    assert result.OPPONENT_SRS_MAX == 8.0
    assert result.OPPONENT_SERIES_COUNT == 2
    assert result.OPPONENT_SRS_STATUS == COMPLETE


def test_empirical_bayes_shrinks_tiny_sample_more():
    made = pd.Series([1, 40, 35, 20, 25, 30, 28, 32])
    attempts = pd.Series([2, 100, 100, 60, 75, 90, 80, 95])
    posterior, alpha, beta = beta_binomial_shrinkage(made, attempts)
    prior = alpha / (alpha + beta)
    raw = made / attempts
    assert abs(posterior.iloc[0] - prior) < abs(raw.iloc[0] - prior)
    assert abs(posterior.iloc[1] - raw.iloc[1]) < abs(posterior.iloc[0] - raw.iloc[0])


def test_engine_aware_roles_define_second_option():
    players = pd.DataFrame([
        {"TEAM_ID": 1, "TEAM_ABBREVIATION": "TST", "PLAYER_ID": 11,
         "PLAYER_NAME": "One", "GP": 10, "MIN": 350, "USG_PCT": .30,
         "PTS": 200, "FGA": 150, "FTA": 40, "FG3M": 20, "FG3A": 55,
         "AST": 80, "TOV": 30, "AST_PCT": .35, "PIE": .19},
        {"TEAM_ID": 1, "TEAM_ABBREVIATION": "TST", "PLAYER_ID": 12,
         "PLAYER_NAME": "Two", "GP": 10, "MIN": 340, "USG_PCT": .25,
         "PTS": 180, "FGA": 140, "FTA": 35, "FG3M": 18, "FG3A": 50,
         "AST": 40, "TOV": 20, "AST_PCT": .20, "PIE": .15},
        {"TEAM_ID": 1, "TEAM_ABBREVIATION": "TST", "PLAYER_ID": 13,
         "PLAYER_NAME": "Three", "GP": 10, "MIN": 330, "USG_PCT": .20,
         "PTS": 130, "FGA": 100, "FTA": 20, "FG3M": 15, "FG3A": 42,
         "AST": 20, "TOV": 15, "AST_PCT": .10, "PIE": .10}])
    qualifiers = pd.DataFrame([{
        "TEAM_ID": 1, "TEAM_ABBREVIATION": "TST", "SERIES_WINS": 2,
        "POSTSEASON_FINISH": "Conference Finals", "QUALIFIER_METHOD": "TEST",
        "QUALIFIER_VALIDATION": "AGREES"}])
    result = select_options(players, qualifiers, PipelineConfig())
    assert result.iloc[0].PLAYER_NAME == "Two"
    assert result.iloc[0].PRIMARY_PLAYER_NAME == "One"


def test_pairing_delta_uses_primary_without_second():
    lineups = pd.DataFrame([
        {"GROUP_ID": "-11-12-13-14-15-", "NET_RATING": 12, "POSS": 100},
        {"GROUP_ID": "-11-16-17-18-19-", "NET_RATING": 2, "POSS": 50},
        {"GROUP_ID": "-12-20-21-22-23-", "NET_RATING": 5, "POSS": 40},
        {"GROUP_ID": "-24-25-26-27-28-", "NET_RATING": 1, "POSS": 60}])
    result = pairing_synergy(lineups, 11, 12)
    assert result["PAIR_SYNERGY_DELTA"] == 10
    assert result["CONTEXTUAL_INTERACTION"] == 6
    assert result["SYNERGY_STATUS"] == COMPLETE


def test_missing_pair_comparator_is_not_modeled():
    lineups = pd.DataFrame([
        {"GROUP_ID": "-11-12-13-14-15-", "NET_RATING": 12, "POSS": 100}])
    result = pairing_synergy(lineups, 11, 12)
    assert np.isnan(result["PAIR_SYNERGY_DELTA"])
    assert result["SYNERGY_STATUS"] == NOT_MODELED
    assert confidence_label(result["PRIMARY_WITHOUT_SECOND_SAMPLE"]) == NOT_MODELED


def test_current_rim_schema_is_mapped():
    runs = pd.DataFrame([{"PLAYER_ID": 11}])
    rim = pd.DataFrame([{"CLOSE_DEF_PERSON_ID": 11, "LT_06_PCT": .51,
                         "PLUSMINUS": -.08}])
    result = attach_optional_metrics(runs, pd.DataFrame(), pd.DataFrame(), rim, pd.DataFrame())
    assert result.iloc[0].RIM_DFG_PCT == .51
    assert result.iloc[0].RIM_DFG_DIFF == -.08


def test_name_matching_handles_initials_suffixes_and_mojibake():
    assert normalized_name("JR Smith") == normalized_name("J.R. Smith")
    assert normalized_name("Manu Ginobili") == normalized_name("Manu GinÃ³bili*")
    assert normalized_name("Gary Payton II") == "garypayton"


def test_era_environment_uses_full_playoff_rotation_population():
    players = pd.DataFrame([
        {"PLAYER_ID": 1, "GP": 10, "MIN": 300, "PTS": 200, "FGA": 150,
         "FTA": 40, "FG3A": 40, "POSS": 400, "USG_PCT": .20, "AST_PCT": .10,
         "PACE": 90, "OFF_RATING": 105},
        {"PLAYER_ID": 2, "GP": 10, "MIN": 300, "PTS": 300, "FGA": 200,
         "FTA": 50, "FG3A": 80, "POSS": 400, "USG_PCT": .30, "AST_PCT": .20,
         "PACE": 92, "OFF_RATING": 110},
        {"PLAYER_ID": 3, "GP": 4, "MIN": 20, "PTS": 1000, "FGA": 1,
         "FTA": 0, "FG3A": 1, "POSS": 5, "USG_PCT": .90, "AST_PCT": .90,
         "PACE": 120, "OFF_RATING": 200},
    ])
    runs = players.iloc[[1]].copy()
    runs["PPG"] = 30.0
    runs["TS_PCT"] = 300 / (2 * (200 + .44 * 50))
    result = attach_season_environment(players, runs, PipelineConfig())
    assert result.iloc[0].ERA_ENVIRONMENT_SAMPLE_PLAYERS == 2
    assert result.iloc[0].ERA_PTS75_PERCENTILE == 75
    assert result.iloc[0].RELATIVE_TS_PCT > 0


def test_rolling_era_percentile_uses_adjacent_seasons():
    frame = pd.DataFrame({
        "SEASON": ["1999-00", "2000-01", "2001-02", "2004-05"],
        "VALUE": [10, 20, 30, 100],
    })
    scores = rolling_era_percentile(frame, "VALUE")
    assert scores.iloc[1] > scores.iloc[0]
    assert scores.iloc[1] < scores.iloc[2]
    assert scores.iloc[3] == 50


def test_role_adjusted_defense_compares_guards_with_guards():
    frame = pd.DataFrame({
        "SEASON": ["2021-22"] * 8,
        "POSITION_GROUP": ["GUARD"] * 4 + ["BIG"] * 4,
        "DEFENSE": [10, 20, 30, 40, 70, 80, 90, 100],
    })
    scores = role_adjusted_era_percentile(frame, "DEFENSE")
    assert scores.iloc[3] > 80
    assert scores.iloc[3] > scores.iloc[4]


def test_terminal_series_availability_excludes_partial_finalist():
    team_games = pd.DataFrame([
        {"TEAM_ID": 1, "GAME_ID": str(game), "GAME_DATE": f"2020-10-{game:02d}",
         "MATCHUP": "AAA vs. BBB"}
        for game in range(1, 7)
    ])
    player_games = pd.DataFrame([
        {"PLAYER_ID": 11, "GAME_ID": "1", "MIN": 44},
        *[{"PLAYER_ID": 12, "GAME_ID": str(game), "MIN": 40}
          for game in range(1, 7)],
    ])
    runs = pd.DataFrame([{"TEAM_ID": 1, "PLAYER_ID": 11,
                          "PRIMARY_PLAYER_ID": 12}])
    result = deepest_round_availability(team_games, player_games, runs).iloc[0]
    assert result.DEEPEST_ROUND_AVAILABILITY == 1 / 6
    assert not result.RANKING_ELIGIBLE


def test_bpm_reference_is_full_rotation_and_position_aware():
    runs = pd.DataFrame([{
        "PLAYER_NAME": "Big Shooter", "TEAM_ABBREVIATION": "AAA",
    }])
    source = pd.DataFrame([
        {"Player": "Big Shooter", "Tm": "AAA", "Pos": "PF", "G": 10,
         "MP": 300, "BPM": 8, "OBPM": 5, "DBPM": 3, "PER": 25,
         "WS/48": .25, "3PAr": .25, "TS%": .66},
        {"Player": "Other Big", "Tm": "BBB", "Pos": "C", "G": 10,
         "MP": 300, "BPM": 2, "OBPM": 1, "DBPM": 1, "PER": 18,
         "WS/48": .15, "3PAr": .05, "TS%": .58},
    ])
    result = match_bpm(runs, source).iloc[0]
    assert result.POSITION_GROUP == "BIG"
    assert result.BBR_BPM_PERCENTILE == 100
    assert result.POSITION_3PAR_PERCENTILE == 100


def test_weight_sensitivity_is_reproducible_and_reports_rank_ranges():
    domains = ["PRODUCTION_SCORE", "ROLE_BURDEN_SCORE", "CUMULATIVE_IMPACT_SCORE",
               "TERMINAL_RESPONSIBILITY_SCORE", "HISTORICAL_DEFENSE_EVIDENCE_SCORE",
               "ROLE_COMPATIBILITY_SCORE"]
    rows = []
    for player, value in [("Complete", 90), ("Middle", 60), ("Lower", 30)]:
        row = {"PLAYER_NAME": player, "POSTSEASON_FINISH": "Champion",
               "RANKING_ELIGIBLE": True}
        row.update({domain: value for domain in domains})
        rows.append(row)
    first = championship_weight_sensitivity(pd.DataFrame(rows), simulations=500, seed=7)
    second = championship_weight_sensitivity(pd.DataFrame(rows), simulations=500, seed=7)
    assert first.PLAYER_NAME.tolist() == ["Complete", "Middle", "Lower"]
    assert first.SENSITIVITY_MEDIAN_RANK.tolist() == [1, 2, 3]
    assert first.SENSITIVITY_MEAN_SCORE.tolist() == second.SENSITIVITY_MEAN_SCORE.tolist()
