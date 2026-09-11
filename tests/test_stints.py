import pandas as pd

from nba_second_options.stints import build_constant_lineup_stints, elapsed_deciseconds


def test_clock_conversion_handles_regulation_and_overtime():
    assert elapsed_deciseconds(1, "PT12M00.00S") == 0
    assert elapsed_deciseconds(2, "PT12M00.00S") == 7200
    assert elapsed_deciseconds(4, "PT00M00.00S") == 28800
    assert elapsed_deciseconds(5, "PT05M00.00S") == 28800
    assert elapsed_deciseconds(5, "PT00M00.00S") == 31800


def test_constant_lineup_stints_preserve_score_change_and_players():
    rows = []
    for is_home, team_id, players in (
        (True, 1, range(1, 6)),
        (False, 2, range(6, 11)),
    ):
        for player_id in players:
            rows.append({
                "GAME_ID": "g1",
                "TEAM_ID": team_id,
                "PERSON_ID": player_id,
                "IN_TIME_REAL": 0,
                "OUT_TIME_REAL": 7200,
                "IS_HOME": is_home,
            })
    rotation = pd.DataFrame(rows)
    play_by_play = pd.DataFrame([
        {
            "period": 1,
            "clock": "PT12M00.00S",
            "actionNumber": 1,
            "scoreHome": "0",
            "scoreAway": "0",
        },
        {
            "period": 1,
            "clock": "PT06M00.00S",
            "actionNumber": 2,
            "scoreHome": "10",
            "scoreAway": "8",
        },
        {
            "period": 1,
            "clock": "PT00M00.00S",
            "actionNumber": 3,
            "scoreHome": "20",
            "scoreAway": "18",
        },
    ])
    result = build_constant_lineup_stints(rotation, play_by_play)
    assert len(result) == 1
    assert result.iloc[0].HOME_POINT_MARGIN == 2
    assert result.iloc[0].HOME_PLAYER_IDS == "1|2|3|4|5"
    assert result.iloc[0].OBSERVATION_STATUS == "OBSERVED_NOT_CAUSAL"
