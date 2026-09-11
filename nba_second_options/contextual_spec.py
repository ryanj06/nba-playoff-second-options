"""NBA-specific feature contract for the contextual replacement-value model.

The names in this module describe independent basketball concepts, not whichever
provider column happens to be convenient. Source adapters are responsible for
constructing these fields without using the selected second option in a team-need
calculation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .contextual_value import ContextualValueConfig, Interaction


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    meaning: str
    historical_source: str
    modern_source: str | None = None


TEAM_NEEDS = (
    "NEED_SELF_CREATION",
    "NEED_OFF_BALL_GRAVITY",
    "NEED_INTERIOR_PRESSURE",
    "NEED_POA_DEFENSE",
    "NEED_RIM_DEFENSE",
)

SECOND_OPTION_SUPPLIES = (
    "SUPPLY_SCORING_EFFICIENCY",
    "SUPPLY_SELF_CREATION",
    "SUPPLY_OFF_BALL_GRAVITY",
    "SUPPLY_INTERIOR_PRESSURE",
    "SUPPLY_POA_DEFENSE",
    "SUPPLY_RIM_DEFENSE",
)

ALL_ERA_TEAM_NEEDS = TEAM_NEEDS[:3]
ALL_ERA_SECOND_OPTION_SUPPLIES = (
    *SECOND_OPTION_SUPPLIES[:4],
    "SUPPLY_DEFENSIVE_BOX_EVIDENCE",
)

CONTEXT_CONTROLS = (
    "PRIMARY_IMPACT_PRIOR",
    "REST_OF_ROSTER_IMPACT_PRIOR",
    "OPPONENT_STRENGTH",
    "HOME_COURT",
    "SCORE_STATE",
    "ROUND_NUMBER",
    "REST_DAYS",
)

ROLE_MATCHING_FEATURES = (
    "ROLE_SCORING_BURDEN",
    "ROLE_CREATION_BURDEN",
    "ROLE_OFF_BALL_SHARE",
    "ROLE_INTERIOR_SHARE",
    "ROLE_DEFENSIVE_ASSIGNMENT",
    "ROLE_MINUTES_BAND",
    "ROLE_SIZE_FAMILY",
)

INTERACTIONS = (
    Interaction("NEED_SELF_CREATION", "SUPPLY_SELF_CREATION", "FIT_SELF_CREATION"),
    Interaction(
        "NEED_OFF_BALL_GRAVITY",
        "SUPPLY_OFF_BALL_GRAVITY",
        "FIT_OFF_BALL_GRAVITY",
    ),
    Interaction(
        "NEED_INTERIOR_PRESSURE",
        "SUPPLY_INTERIOR_PRESSURE",
        "FIT_INTERIOR_PRESSURE",
    ),
    Interaction("NEED_POA_DEFENSE", "SUPPLY_POA_DEFENSE", "FIT_POA_DEFENSE"),
    Interaction("NEED_RIM_DEFENSE", "SUPPLY_RIM_DEFENSE", "FIT_RIM_DEFENSE"),
)

ALL_ERA_INTERACTIONS = INTERACTIONS[:3]

FEATURE_CATALOG = (
    FeatureDefinition(
        "SUPPLY_SCORING_EFFICIENCY",
        "Era-relative scoring efficiency at the player's observed burden.",
        "Playoff points/75, relative TS%, and empirical-Bayes shooting components.",
    ),
    FeatureDefinition(
        "SUPPLY_SELF_CREATION",
        "Ability to create a viable attempt when the primary cannot initiate.",
        "AST%, turnover rate, shot burden, and documented unassisted estimates.",
        "Isolation, drives, pull-ups, time of possession, and late-clock attempts.",
    ),
    FeatureDefinition(
        "SUPPLY_OFF_BALL_GRAVITY",
        "Spacing and movement value without monopolizing initiation.",
        "Position-relative 3PA rate and stabilized 3P%.",
        "Catch-and-shoot, movement, defender distance, and optical gravity.",
    ),
    FeatureDefinition(
        "SUPPLY_INTERIOR_PRESSURE",
        "Rim pressure supplied by finishing, rolling, cutting, and free throws.",
        "Two-point attempt share, two-point efficiency, and free-throw rate.",
        "Drives, paint touches, rolls, cuts, and rim-shot frequency.",
    ),
    FeatureDefinition(
        "SUPPLY_POA_DEFENSE",
        "Capacity to absorb difficult perimeter assignments.",
        "NOT_MODELED before reliable assignment data; box evidence is separate.",
        "Matchup share, matchup difficulty, screen navigation, and pressure.",
    ),
    FeatureDefinition(
        "SUPPLY_RIM_DEFENSE",
        "Interior deterrence and shot suppression.",
        "Blocks and defensive rebounds are evidence, not a rim-defense substitute.",
        "Rim contests, opponent rim frequency, and defended FG% differential.",
    ),
    FeatureDefinition(
        "SUPPLY_DEFENSIVE_BOX_EVIDENCE",
        "Limited all-era defensive evidence without claiming assignment or deterrence.",
        "Era-relative DBPM, defensive win shares, steals, blocks, and rebounds.",
    ),
)


def all_era_core_config(target: str = "OPP_ADJ_MARGIN_PER_100") -> ContextualValueConfig:
    """Return the locked all-era CRV specification.

    The core does not pretend that box-score defense measures point-of-attack
    coverage or rim deterrence. It includes a separately labeled defensive-box
    evidence field and omits the two defensive-fit interactions.
    """

    return ContextualValueConfig(
        target=target,
        sample_weight="ESTIMATED_POSSESSIONS",
        base_features=(*CONTEXT_CONTROLS, *ALL_ERA_TEAM_NEEDS),
        secondary_features=ALL_ERA_SECOND_OPTION_SUPPLIES,
        role_features=ROLE_MATCHING_FEATURES,
        interactions=ALL_ERA_INTERACTIONS,
    )


def modern_tracking_config(target: str = "OPP_ADJ_MARGIN_PER_100") -> ContextualValueConfig:
    """Return the extension that can estimate defensive-fit interactions."""

    return ContextualValueConfig(
        target=target,
        sample_weight="ESTIMATED_POSSESSIONS",
        base_features=(*CONTEXT_CONTROLS, *TEAM_NEEDS),
        secondary_features=SECOND_OPTION_SUPPLIES,
        role_features=ROLE_MATCHING_FEATURES,
        interactions=INTERACTIONS,
    )


def validate_not_modeled_disclosures(frame: pd.DataFrame) -> None:
    """Require a status column whenever a modeled concept is unavailable."""

    errors: list[str] = []
    disclosed_features = set(SECOND_OPTION_SUPPLIES) | set(
        ALL_ERA_SECOND_OPTION_SUPPLIES
    )
    for feature in sorted(disclosed_features):
        if feature not in frame:
            continue
        status = f"{feature}_STATUS"
        if status not in frame:
            errors.append(f"{feature} requires {status}")
            continue
        missing = frame[feature].isna()
        disclosed = frame[status].eq("NOT_MODELED")
        if (missing & ~disclosed).any():
            errors.append(f"missing {feature} values must be tagged NOT_MODELED")
        if ((~missing) & disclosed).any():
            errors.append(f"observed {feature} values cannot be tagged NOT_MODELED")
    if errors:
        raise ValueError("; ".join(errors))
