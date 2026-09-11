import numpy as np
import pandas as pd
import pytest

from nba_second_options.contextual_value import (
    ContextualReplacementModel,
    ContextualValueConfig,
    Interaction,
    unique_feature_votes,
)
from nba_second_options.contextual_spec import validate_not_modeled_disclosures


def synthetic_frame(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season_start in range(2000, 2012):
        for player in range(10):
            need = rng.uniform(0.05, 1.0)
            supply = rng.uniform(0.05, 1.0)
            star_quality = rng.normal()
            rate_value = rng.normal()
            role_creator = rng.uniform(0.0, 1.0)
            outcome = (
                1.5 * star_quality
                + 2.0 * rate_value
                + 3.0 * supply
                + 7.0 * need * supply
                + rng.normal(scale=0.15)
            )
            rows.append({
                "RUN_ID": f"{season_start}-{player}",
                "PLAYER_ID": season_start * 100 + player,
                "SEASON": f"{season_start}-{str(season_start + 1)[-2:]}",
                "OUTCOME": outcome,
                "STAR_QUALITY": star_quality,
                "TEAM_NEED": need,
                "RATE_VALUE": rate_value,
                "SUPPLY": supply,
                "ROLE_CREATOR": role_creator,
            })
    return pd.DataFrame(rows)


def config(**overrides) -> ContextualValueConfig:
    values = {
        "target": "OUTCOME",
        "base_features": ("STAR_QUALITY", "TEAM_NEED"),
        "secondary_features": ("RATE_VALUE", "SUPPLY"),
        "role_features": ("ROLE_CREATOR",),
        "interactions": (Interaction("TEAM_NEED", "SUPPLY", "NEED_X_SUPPLY"),),
        "minimum_rows": 80,
        "minimum_training_rows": 30,
        "replacement_neighbors": 8,
        "minimum_relative_rmse_improvement": 0.01,
    }
    values.update(overrides)
    return ContextualValueConfig(**values)


def test_nested_chronological_validation_and_contextual_scoring():
    frame = synthetic_frame()
    model = ContextualReplacementModel(config()).fit(frame)

    assert model.validation_ is not None
    assert model.validation_.status == "VALIDATED"
    assert model.validation_.folds == 3
    assert model.validation_.relative_improvement > 0.5

    target = frame.iloc[[frame.SUPPLY.argmax()]].copy()
    result = model.score(target).iloc[0]
    assert result.MODEL_VALIDATION_STATUS == "VALIDATED"
    assert result.CONTEXTUAL_REPLACEMENT_VALUE > 0
    assert result.REPLACEMENT_NEIGHBORS == 8


def test_interaction_supply_must_be_counterfactually_replaceable():
    bad = config(secondary_features=("RATE_VALUE",))
    with pytest.raises(ValueError, match="interaction supply"):
        ContextualReplacementModel(bad).fit(synthetic_frame())


def test_duplicate_feature_votes_are_rejected():
    bad = config(base_features=("STAR_QUALITY", "RATE_VALUE", "TEAM_NEED"))
    with pytest.raises(ValueError, match="both base and secondary"):
        ContextualReplacementModel(bad).fit(synthetic_frame())


def test_failed_validation_blocks_ranking():
    model = ContextualReplacementModel(
        config(minimum_relative_rmse_improvement=0.9999)
    ).fit(synthetic_frame())
    assert model.validation_ is not None
    assert model.validation_.status == "FAILED_OUT_OF_SAMPLE_VALIDATION"
    with pytest.raises(RuntimeError, match="ranking blocked"):
        model.score(synthetic_frame().iloc[[0]])


def test_feature_vote_helper():
    assert unique_feature_votes((("a", "b"), ("c",)))
    assert not unique_feature_votes((("a", "b"), ("b", "c")))


def test_missing_skill_requires_not_modeled_disclosure():
    frame = pd.DataFrame({
        "SUPPLY_POA_DEFENSE": [np.nan, 0.8],
        "SUPPLY_POA_DEFENSE_STATUS": ["NOT_MODELED", "COMPLETE"],
    })
    validate_not_modeled_disclosures(frame)

    frame.loc[0, "SUPPLY_POA_DEFENSE_STATUS"] = "COMPLETE"
    with pytest.raises(ValueError, match="must be tagged NOT_MODELED"):
        validate_not_modeled_disclosures(frame)
