from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Interaction:
    """One team-need × second-option-supply interaction."""

    need: str
    supply: str
    output: str


@dataclass(frozen=True)
class ContextualValueConfig:
    """Configuration for a role-conditioned replacement-value model."""

    target: str
    season: str = "SEASON"
    row_id: str = "RUN_ID"
    player_id: str = "PLAYER_ID"
    sample_weight: str | None = None
    base_features: tuple[str, ...] = ()
    secondary_features: tuple[str, ...] = ()
    role_features: tuple[str, ...] = ()
    interactions: tuple[Interaction, ...] = ()
    alphas: tuple[float, ...] = (.1, 1.0, 10.0, 100.0, 1000.0)
    replacement_neighbors: int = 12
    era_radius: int = 2
    minimum_rows: int = 80
    minimum_training_rows: int = 40
    validation_holdout_seasons: int = 3
    minimum_relative_rmse_improvement: float = .01


@dataclass(frozen=True)
class ValidationResult:
    selected_alpha: float
    model_rmse: float
    baseline_rmse: float
    relative_improvement: float
    folds: int
    observations: int
    status: str


@dataclass
class ContextualReplacementModel:
    """Estimate value above a role- and era-matched second option.

    The class deliberately separates model features from replacement-matching
    features. A role measurement may determine who constitutes a plausible
    replacement without receiving another independent vote in the prediction.
    """

    config: ContextualValueConfig
    estimator_: Pipeline | None = field(default=None, init=False)
    training_: pd.DataFrame | None = field(default=None, init=False)
    validation_: ValidationResult | None = field(default=None, init=False)
    feature_columns_: list[str] = field(default_factory=list, init=False)

    @staticmethod
    def _season_end_year(values: pd.Series) -> pd.Series:
        return pd.to_numeric(values.astype(str).str[:4], errors="coerce") + 1

    def _required_columns(self) -> set[str]:
        interaction_columns = {
            column
            for interaction in self.config.interactions
            for column in (interaction.need, interaction.supply)
        }
        return {
            self.config.target,
            self.config.season,
            self.config.row_id,
            self.config.player_id,
            *self.config.base_features,
            *self.config.secondary_features,
            *self.config.role_features,
            *interaction_columns,
            *([self.config.sample_weight] if self.config.sample_weight else []),
        }

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        if self.config.validation_holdout_seasons < 1:
            raise ValueError("validation_holdout_seasons must be at least one")
        if not self.config.alphas or any(alpha <= 0 for alpha in self.config.alphas):
            raise ValueError("alphas must contain positive values")
        if self.config.replacement_neighbors < 1:
            raise ValueError("replacement_neighbors must be at least one")
        if not self.config.role_features:
            raise ValueError("role_features are required for replacement matching")
        missing = self._required_columns() - set(frame.columns)
        if missing:
            raise ValueError(f"Contextual model missing columns: {sorted(missing)}")
        if frame[self.config.row_id].duplicated().any():
            raise ValueError(f"{self.config.row_id} must uniquely identify training rows")
        if len(frame) < self.config.minimum_rows:
            raise ValueError(
                f"Contextual model requires at least {self.config.minimum_rows} rows; "
                f"received {len(frame)}"
            )
        if self.config.sample_weight:
            weights = pd.to_numeric(frame[self.config.sample_weight], errors="coerce")
            if weights.notna().sum() < self.config.minimum_rows or (weights.dropna() <= 0).any():
                raise ValueError("sample weights must be positive for enough training rows")
        minimum_seasons = self.config.validation_holdout_seasons + 3
        if frame[self.config.season].nunique() < minimum_seasons:
            raise ValueError(
                f"At least {minimum_seasons} seasons are required for nested chronological "
                "validation"
            )
        duplicated_votes = set(self.config.base_features) & set(
            self.config.secondary_features)
        if duplicated_votes:
            raise ValueError(
                "Features cannot vote in both base and secondary groups: "
                f"{sorted(duplicated_votes)}"
            )
        supplies = {interaction.supply for interaction in self.config.interactions}
        unreplaceable_supplies = supplies - set(self.config.secondary_features)
        if unreplaceable_supplies:
            raise ValueError(
                "Every interaction supply must be replaced in the counterfactual: "
                f"{sorted(unreplaceable_supplies)}"
            )
        needs = {interaction.need for interaction in self.config.interactions}
        mutable_needs = needs & set(self.config.secondary_features)
        if mutable_needs:
            raise ValueError(
                "Team-need features must remain fixed in the counterfactual: "
                f"{sorted(mutable_needs)}"
            )

    def _with_interactions(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for interaction in self.config.interactions:
            out[interaction.output] = (
                pd.to_numeric(out[interaction.need], errors="coerce")
                * pd.to_numeric(out[interaction.supply], errors="coerce")
            )
        return out

    def _model_features(self) -> list[str]:
        return [
            *self.config.base_features,
            *self.config.secondary_features,
            *(interaction.output for interaction in self.config.interactions),
        ]

    @staticmethod
    def _pipeline(alpha: float) -> Pipeline:
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])

    def _weights(self, frame: pd.DataFrame) -> pd.Series:
        if self.config.sample_weight is None:
            return pd.Series(1.0, index=frame.index)
        return pd.to_numeric(frame[self.config.sample_weight], errors="coerce")

    @staticmethod
    def _fit_estimator(
        estimator: Pipeline,
        features: pd.DataFrame,
        target: pd.Series,
        weights: pd.Series,
    ) -> None:
        estimator.fit(features, target, ridge__sample_weight=weights)

    @staticmethod
    def _weighted_rmse(
        observed: np.ndarray,
        predicted: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        return float(np.sqrt(np.average((observed - predicted) ** 2, weights=weights)))

    def _chronological_predictions(
        self,
        frame: pd.DataFrame,
        alpha: float,
        held_out_seasons: Iterable[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        predictions = np.full(len(frame), np.nan)
        baselines = np.full(len(frame), np.nan)
        season_years = self._season_end_year(frame[self.config.season])
        features = frame[self.feature_columns_]
        target = pd.to_numeric(frame[self.config.target], errors="coerce")
        weights = self._weights(frame)
        if held_out_seasons is None:
            held_out_years = sorted(season_years.dropna().unique())
        else:
            held_out_years = sorted(
                self._season_end_year(pd.Series(list(held_out_seasons))).dropna().unique()
            )
        minimum_training_rows = max(
            self.config.minimum_training_rows,
            len(self.feature_columns_) * 3,
        )
        for held_out_year in held_out_years:
            valid_weight = weights.notna() & weights.gt(0)
            train = (season_years < held_out_year) & target.notna() & valid_weight
            test = (season_years == held_out_year) & target.notna() & valid_weight
            # Never backfill an early fold with future seasons. If there is not
            # enough historical data, that fold is ineligible for validation.
            if train.sum() < minimum_training_rows or not test.any():
                continue
            estimator = self._pipeline(alpha)
            self._fit_estimator(
                estimator,
                features.loc[train],
                target.loc[train],
                weights.loc[train],
            )
            predictions[test.to_numpy()] = estimator.predict(features.loc[test])
            baselines[test.to_numpy()] = float(
                np.average(target.loc[train], weights=weights.loc[train])
            )
        return predictions, baselines

    def fit(self, frame: pd.DataFrame) -> ContextualReplacementModel:
        self._validate_frame(frame)
        work = self._with_interactions(frame).reset_index(drop=True)
        self.feature_columns_ = self._model_features()
        target = pd.to_numeric(work[self.config.target], errors="coerce")
        weights = self._weights(work)
        valid_target = (target.notna() & weights.notna() & weights.gt(0)).to_numpy()
        if valid_target.sum() < self.config.minimum_rows:
            raise ValueError("Too few non-missing target observations")

        seasons = sorted(work[self.config.season].astype(str).unique())
        holdout_count = self.config.validation_holdout_seasons
        development_seasons = seasons[:-holdout_count]
        holdout_seasons = seasons[-holdout_count:]

        candidates: list[tuple[float, float]] = []
        for alpha in self.config.alphas:
            development = work[work[self.config.season].astype(str).isin(development_seasons)]
            predictions, baselines = self._chronological_predictions(development, alpha)
            development_target = pd.to_numeric(
                development[self.config.target], errors="coerce"
            ).to_numpy()
            development_weights = self._weights(development).to_numpy()
            valid = np.isfinite(predictions) & np.isfinite(baselines) & np.isfinite(
                development_target
            ) & np.isfinite(development_weights) & (development_weights > 0)
            if valid.any():
                model_rmse = self._weighted_rmse(
                    development_target[valid],
                    predictions[valid],
                    development_weights[valid],
                )
                candidates.append((model_rmse, alpha))
        if not candidates:
            raise ValueError(
                "No leakage-free development folds had enough historical training rows"
            )
        _, alpha = min(candidates)

        predictions, baselines = self._chronological_predictions(
            work, alpha, held_out_seasons=holdout_seasons
        )
        valid = valid_target & np.isfinite(predictions) & np.isfinite(baselines)
        if not valid.any():
            raise ValueError("No eligible observations in the final chronological holdout")
        target_values = target.to_numpy()
        weight_values = weights.to_numpy()
        model_rmse = self._weighted_rmse(
            target_values[valid], predictions[valid], weight_values[valid]
        )
        baseline_rmse = self._weighted_rmse(
            target_values[valid], baselines[valid], weight_values[valid]
        )
        improvement = 1 - model_rmse / baseline_rmse if baseline_rmse else 0.0
        observations = int(valid.sum())
        status = (
            "VALIDATED"
            if improvement >= self.config.minimum_relative_rmse_improvement
            else "FAILED_OUT_OF_SAMPLE_VALIDATION"
        )
        self.validation_ = ValidationResult(
            selected_alpha=float(alpha),
            model_rmse=model_rmse,
            baseline_rmse=baseline_rmse,
            relative_improvement=improvement,
            folds=int(work.loc[valid, self.config.season].nunique()),
            observations=observations,
            status=status,
        )
        self.estimator_ = self._pipeline(float(alpha))
        self._fit_estimator(
            self.estimator_,
            work.loc[valid_target, self.feature_columns_],
            target.loc[valid_target],
            weights.loc[valid_target],
        )
        work["_SEASON_END_YEAR"] = self._season_end_year(work[self.config.season])
        self.training_ = work
        return self

    def _replacement_candidates(self, row: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
        if self.training_ is None:
            raise RuntimeError("Fit the contextual model before requesting replacements")
        pool = self.training_.copy()
        target_year = float(row["_SEASON_END_YEAR"])
        pool = pool[
            (pool["_SEASON_END_YEAR"] - target_year).abs() <= self.config.era_radius
        ]
        pool = pool[
            (pool[self.config.row_id] != row[self.config.row_id])
            & (pool[self.config.player_id] != row[self.config.player_id])
        ]
        if pool.empty:
            raise ValueError(f"No role-matched replacements for {row[self.config.row_id]}")

        role_pool = pool[list(self.config.role_features)].apply(
            pd.to_numeric, errors="coerce")
        role_target = pd.to_numeric(row[list(self.config.role_features)], errors="coerce")
        medians = role_pool.median()
        role_pool = role_pool.fillna(medians)
        role_target = role_target.fillna(medians)
        scale = role_pool.std(ddof=0).replace(0, 1).fillna(1)
        distance = np.sqrt((((role_pool - role_target) / scale) ** 2).mean(axis=1))
        count = min(self.config.replacement_neighbors, len(pool))
        selected_index = distance.nsmallest(count).index
        selected = pool.loc[selected_index].copy()
        selected_distance = distance.loc[selected_index].to_numpy()
        positive = selected_distance[selected_distance > 0]
        bandwidth = float(np.median(positive)) if len(positive) else 1.0
        weights = np.exp(-(selected_distance ** 2) / (2 * bandwidth ** 2))
        weights = weights / weights.sum()
        return selected, weights

    def score(
        self, rows: pd.DataFrame, *, require_validated: bool = True
    ) -> pd.DataFrame:
        if self.estimator_ is None or self.training_ is None or self.validation_ is None:
            raise RuntimeError("Fit the contextual model before scoring")
        if require_validated and self.validation_.status != "VALIDATED":
            raise RuntimeError(
                "Contextual ranking blocked because the model failed out-of-sample validation"
            )
        missing = {
            self.config.row_id,
            self.config.player_id,
            self.config.season,
            *self.config.base_features,
            *self.config.secondary_features,
            *self.config.role_features,
            *(column for interaction in self.config.interactions
              for column in (interaction.need, interaction.supply)),
        } - set(rows.columns)
        if missing:
            raise ValueError(f"Scoring rows missing columns: {sorted(missing)}")

        work = self._with_interactions(rows).copy()
        work["_SEASON_END_YEAR"] = self._season_end_year(work[self.config.season])
        records: list[dict[str, object]] = []
        for _, row in work.iterrows():
            missing_secondary = [
                column
                for column in self.config.secondary_features
                if pd.isna(row[column])
            ]
            actual = float(self.estimator_.predict(
                pd.DataFrame([row[self.feature_columns_]]))[0])
            replacements, weights = self._replacement_candidates(row)
            counterfactual_rows = pd.DataFrame(
                [row.copy() for _ in range(len(replacements))],
                index=replacements.index,
            )
            for column in self.config.secondary_features:
                counterfactual_rows[column] = replacements[column]
            counterfactual_rows = self._with_interactions(counterfactual_rows)
            counterfactual_predictions = self.estimator_.predict(
                counterfactual_rows[self.feature_columns_])
            expected_replacement = float(np.average(counterfactual_predictions, weights=weights))
            records.append({
                self.config.row_id: row[self.config.row_id],
                "PREDICTED_CONTEXT_OUTCOME": actual,
                "ROLE_MATCHED_REPLACEMENT_OUTCOME": expected_replacement,
                "CONTEXTUAL_REPLACEMENT_VALUE": actual - expected_replacement,
                "REPLACEMENT_NEIGHBORS": len(replacements),
                "MODEL_VALIDATION_STATUS": self.validation_.status,
                "MODEL_ALPHA": self.validation_.selected_alpha,
                "MODEL_RELATIVE_RMSE_IMPROVEMENT": self.validation_.relative_improvement,
                "SECONDARY_FEATURE_COVERAGE": (
                    1 - len(missing_secondary) / len(self.config.secondary_features)
                    if self.config.secondary_features
                    else 1.0
                ),
                "NOT_MODELED_FEATURES": "|".join(missing_secondary),
                "DATA_COVERAGE_STATUS": "PARTIAL" if missing_secondary else "COMPLETE",
            })
        return pd.DataFrame(records)


def unique_feature_votes(groups: Iterable[Iterable[str]]) -> bool:
    """Return True when no model input is repeated across conceptual groups."""
    flattened = [feature for group in groups for feature in group]
    return len(flattened) == len(set(flattened))
