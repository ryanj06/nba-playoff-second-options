import pandas as pd

from nba_second_options.headshots import (
    ARCHIVED_SEASON_PORTRAIT_URLS,
    portrait_path,
    season_portrait_url,
)


def test_season_specific_url_uses_team_season_and_player_id(tmp_path):
    row = pd.Series({
        "SEASON": "2019-20",
        "TEAM_ID": 1610612747,
        "PLAYER_ID": 203076,
    })
    assert season_portrait_url(row).endswith(
        "/nba/1610612747/2019/260x190/203076.png"
    )
    assert portrait_path(tmp_path, "2019-20", 203076).name == "2019_20_203076.png"


def test_archived_portraits_are_keyed_to_the_exact_run() -> None:
    assert ("2000-01", 977) in ARCHIVED_SEASON_PORTRAIT_URLS
    assert "20001003" in ARCHIVED_SEASON_PORTRAIT_URLS[("2000-01", 977)]
    assert ("2004-05", 1938) in ARCHIVED_SEASON_PORTRAIT_URLS
