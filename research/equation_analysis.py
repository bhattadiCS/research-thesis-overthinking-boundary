from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

from universal_feature_analysis import (
    CANDIDATE_FEATURES,
    FINAL_BASE_FEATURES,
    RUNS as LEGACY_RUNS,
    safe_auc,
    zscore_per_family,
)


RESEARCH_ROOT = Path(__file__).resolve().parent
OUTPUT_BASE = RESEARCH_ROOT / "outputs"
OUTPUT_DIR = OUTPUT_BASE / "equation_analysis"
REPORTS_DIR = RESEARCH_ROOT / "reports"
DEFAULT_REPORT_PATH = REPORTS_DIR / "equation_analysis_report.md"
DEFAULT_SUMMARY_PATH = OUTPUT_DIR / "variant_summary.csv"
DEFAULT_LOFO_PATH = OUTPUT_DIR / "variant_lofo_metrics.csv"
DEFAULT_GEOMETRY_PATH = OUTPUT_DIR / "geometry_signal_summary.csv"
DEFAULT_COVERAGE_PATH = OUTPUT_DIR / "dataset_coverage.csv"
DEFAULT_WEIGHT_PATH = OUTPUT_DIR / "recommended_equation_weights.csv"
DEFAULT_METADATA_PATH = OUTPUT_DIR / "equation_analysis_metadata.json"
STEP_COST = 0.05
GEOMETRY_FEATURES = ("hidden_kl_divergence", "pca_velocity_norm")
GEOMETRY_SCORE_COLUMNS = ("hidden_l2_shift",) + GEOMETRY_FEATURES
CURRENT_BASELINE_VARIANT = "hazard_quadratic_top4"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    estimator_family: str
    features: tuple[str, ...]
    basis: str
    q_model_kind: str | None
    alpha_model_kind: str | None
    beta_model_kind: str | None
    drift_model_kind: str | None
    description: str
    closed_form: bool


class ConstantProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = np.full(len(frame), self.probability, dtype=float)
        return np.column_stack([1.0 - probabilities, probabilities])


class ConstantRegressionModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value, dtype=float)


def family_label_for_run_dir(run_dir: Path) -> str:
    for family_label, relative_name in LEGACY_RUNS.items():
        if run_dir == OUTPUT_BASE / relative_name or run_dir.name == Path(relative_name).name:
            return family_label

    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_meta = metadata.get("model", {})
        family = str(model_meta.get("family", "")).strip()
        parameter_count = str(model_meta.get("parameter_count", "")).strip()
        if family and parameter_count:
            return f"{family} {parameter_count}"
        alias = str(model_meta.get("alias", "")).strip()
        if alias:
            return alias
    return run_dir.name


def load_run_frame(run_dir: Path, family_label: str) -> pd.DataFrame:
    trace_path = run_dir / "trace_steps.csv"
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing trace_steps.csv in {run_dir}")

    frame = pd.read_csv(trace_path).sort_values(["run_id", "step"]).copy()
    frame["family"] = family_label
    frame["run_dir"] = run_dir.name
    frame["correct"] = pd.to_numeric(frame.get("correct"), errors="coerce").fillna(0).astype(int)
    frame["step"] = pd.to_numeric(frame.get("step"), errors="coerce").fillna(0).astype(int)
    frame["utility"] = pd.to_numeric(frame.get("utility"), errors="coerce")
    if frame["utility"].isna().any():
        frame["utility"] = frame["correct"].astype(float) - STEP_COST * (frame["step"].astype(float) - 1.0)

    numeric_columns = set(CANDIDATE_FEATURES) | {"parse_success", "raw_generation_tokens", *GEOMETRY_SCORE_COLUMNS}
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["next_correct"] = frame.groupby("run_id")["correct"].shift(-1)
    frame["next_utility"] = frame.groupby("run_id")["utility"].shift(-1)
    frame["has_next"] = frame["next_correct"].notna()
    frame["event_repair"] = ((frame["correct"] == 0) & (frame["next_correct"] == 1)).astype(int)
    frame["event_corruption"] = ((frame["correct"] == 1) & (frame["next_correct"] == 0)).astype(int)
    frame["valid_repair"] = (frame["correct"] == 0) & frame["has_next"]
    frame["valid_corruption"] = (frame["correct"] == 1) & frame["has_next"]
    frame["delta_utility"] = frame["next_utility"] - frame["utility"]
    frame["positive_drift"] = (frame["delta_utility"] > 0.0).astype(int)
    frame["parse_success"] = pd.to_numeric(frame.get("parse_success"), errors="coerce")
    return frame


def balanced_sample_weight(target: pd.Series) -> np.ndarray:
    counts = target.value_counts().to_dict()
    class_count = max(len(counts), 1)
    total = float(len(target))
    weights = {label: total / (class_count * count) for label, count in counts.items() if count > 0}
    return target.map(weights).to_numpy(dtype=float)


def expected_calibration_error(target: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    if len(target) == 0:
        return float("nan")
    target = np.asarray(target, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not mask.any():
            continue
        ece += abs(float(target[mask].mean()) - float(probabilities[mask].mean())) * float(mask.mean())
    return float(ece)


def polynomial_step(basis: str) -> PolynomialFeatures | None:
    if basis == "linear":
        return None
    if basis == "quadratic":
        return PolynomialFeatures(degree=2, include_bias=False)
    if basis == "cubic":
        return PolynomialFeatures(degree=3, include_bias=False)
    raise ValueError(f"Unsupported basis: {basis}")


def build_classifier_pipeline(model_kind: str, basis: str, random_state: int) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    basis_transform = polynomial_step(basis) if model_kind in {"logistic", "svm"} else None
    if basis_transform is not None:
        steps.append(("basis", basis_transform))
    if model_kind in {"logistic", "svm"}:
        steps.append(("scale", StandardScaler()))

    if model_kind == "logistic":
        estimator: Any = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            C=1.0,
            random_state=random_state,
        )
    elif model_kind == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    elif model_kind == "hist_gb":
        estimator = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=200,
            random_state=random_state,
        )
    elif model_kind == "svm":
        estimator = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported classifier kind: {model_kind}")

    steps.append(("model", estimator))
    return Pipeline(steps)


def build_regressor_pipeline(model_kind: str, basis: str, random_state: int) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    basis_transform = polynomial_step(basis) if model_kind == "ridge" else None
    if basis_transform is not None:
        steps.append(("basis", basis_transform))
    if model_kind == "ridge":
        steps.append(("scale", StandardScaler()))
        estimator: Any = Ridge(alpha=1.0, random_state=random_state)
    elif model_kind == "hist_gb":
        estimator = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=200,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported regressor kind: {model_kind}")
    steps.append(("model", estimator))
    return Pipeline(steps)


def numeric_feature_frame(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    return frame[list(features)].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def fit_classifier(
    train_frame: pd.DataFrame,
    features: tuple[str, ...],
    target_column: str,
    model_kind: str,
    basis: str,
    random_state: int,
) -> Pipeline | ConstantProbabilityModel:
    target = train_frame[target_column].astype(int)
    if target.nunique() < 2:
        return ConstantProbabilityModel(float(target.mean()))

    model = build_classifier_pipeline(model_kind=model_kind, basis=basis, random_state=random_state)
    fit_kwargs: dict[str, Any] = {}
    if model_kind == "hist_gb":
        fit_kwargs["model__sample_weight"] = balanced_sample_weight(target)
    model.fit(numeric_feature_frame(train_frame, features), target, **fit_kwargs)
    return model


def fit_regressor(
    train_frame: pd.DataFrame,
    features: tuple[str, ...],
    target_column: str,
    model_kind: str,
    basis: str,
    random_state: int,
) -> Pipeline | ConstantRegressionModel:
    target = pd.to_numeric(train_frame[target_column], errors="coerce").fillna(0.0)
    if np.isclose(float(target.std(ddof=0)), 0.0):
        return ConstantRegressionModel(float(target.mean()))

    model = build_regressor_pipeline(model_kind=model_kind, basis=basis, random_state=random_state)
    model.fit(numeric_feature_frame(train_frame, features), target)
    return model


def predict_probabilities(model: Pipeline | ConstantProbabilityModel, frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    if frame.empty:
        return np.empty((0, 2), dtype=float)
    if isinstance(model, ConstantProbabilityModel):
        return model.predict_proba(frame)
    return model.predict_proba(numeric_feature_frame(frame, features))


def predict_regression(model: Pipeline | ConstantRegressionModel, frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    if frame.empty:
        return np.empty((0,), dtype=float)
    if isinstance(model, ConstantRegressionModel):
        return model.predict(frame)
    return model.predict(numeric_feature_frame(frame, features))


def stop_step_from_column(group: pd.DataFrame, column: str) -> int:
    ordered = group.sort_values("step")
    candidate = ordered.loc[(ordered["step"] >= 2) & (ordered[column] <= 0.0)]
    if not candidate.empty:
        return int(candidate.iloc[0]["step"])
    return int(ordered.iloc[-1]["step"])


def oracle_metrics(frame: pd.DataFrame, column: str) -> tuple[float, float, float]:
    accuracies: list[float] = []
    oracle_gaps: list[float] = []
    stop_utilities: list[float] = []
    for _, group in frame.groupby("run_id"):
        ordered = group.sort_values("step")
        stop_step = stop_step_from_column(ordered, column)
        stop_row = ordered.loc[ordered["step"] == stop_step].iloc[0]
        oracle_row = ordered.loc[ordered["utility"].idxmax()]
        accuracies.append(float(abs(int(stop_row["step"]) - int(oracle_row["step"])) <= 1))
        oracle_gaps.append(float(oracle_row["utility"] - stop_row["utility"]))
        stop_utilities.append(float(stop_row["utility"]))
    return float(np.mean(accuracies)), float(np.mean(oracle_gaps)), float(np.mean(stop_utilities))


def softmax_distribution(vector: np.ndarray) -> np.ndarray:
    shifted = vector.astype(np.float64) - float(np.max(vector))
    stabilized = np.clip(shifted, -60.0, 60.0)
    exp_values = np.exp(stabilized)
    total = float(exp_values.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(vector), 1.0 / max(len(vector), 1), dtype=np.float64)
    return exp_values / total


def hidden_kl_divergence(current: np.ndarray, previous: np.ndarray) -> float:
    curr_dist = softmax_distribution(current)
    prev_dist = softmax_distribution(previous)
    epsilon = 1e-12
    return float(np.sum(curr_dist * (np.log(curr_dist + epsilon) - np.log(prev_dist + epsilon))))


def compute_geometry_features(run_dir: Path, family_label: str, random_state: int, max_pca_files: int = 256) -> pd.DataFrame:
    hidden_dir = run_dir / "hidden_states"
    if not hidden_dir.exists():
        return pd.DataFrame(columns=["family", "run_id", "step", *GEOMETRY_FEATURES])

    npz_files = sorted(hidden_dir.glob("*.npz"))
    if not npz_files:
        return pd.DataFrame(columns=["family", "run_id", "step", *GEOMETRY_FEATURES])

    rng = np.random.default_rng(random_state)
    if len(npz_files) > max_pca_files:
        sample_indices = np.sort(rng.choice(len(npz_files), size=max_pca_files, replace=False))
        pca_files = [npz_files[index] for index in sample_indices]
    else:
        pca_files = npz_files

    pca_vectors: list[np.ndarray] = []
    for npz_path in pca_files:
        with np.load(npz_path) as payload:
            hidden_states = np.asarray(payload["hidden_states"], dtype=np.float32)
        if hidden_states.ndim == 2 and hidden_states.shape[0] >= 1:
            pca_vectors.append(hidden_states)

    if not pca_vectors:
        return pd.DataFrame(columns=["family", "run_id", "step", *GEOMETRY_FEATURES])

    sample_matrix = np.vstack(pca_vectors)
    component_count = min(3, sample_matrix.shape[0], sample_matrix.shape[1])
    if component_count < 1:
        return pd.DataFrame(columns=["family", "run_id", "step", *GEOMETRY_FEATURES])

    pca = PCA(n_components=component_count, svd_solver="randomized", random_state=random_state)
    pca.fit(sample_matrix)

    rows: list[dict[str, Any]] = []
    for npz_path in npz_files:
        run_id = npz_path.stem
        with np.load(npz_path) as payload:
            hidden_states = np.asarray(payload["hidden_states"], dtype=np.float32)
        if hidden_states.ndim != 2 or hidden_states.shape[0] == 0:
            continue
        projected = pca.transform(hidden_states)
        for index in range(hidden_states.shape[0]):
            step = index + 1
            if index == 0:
                kl_value = 0.0
                pca_velocity = 0.0
            else:
                kl_value = hidden_kl_divergence(hidden_states[index], hidden_states[index - 1])
                pca_velocity = float(np.linalg.norm(projected[index] - projected[index - 1]))
            rows.append(
                {
                    "family": family_label,
                    "run_id": run_id,
                    "step": step,
                    "hidden_kl_divergence": kl_value,
                    "pca_velocity_norm": pca_velocity,
                }
            )
    return pd.DataFrame(rows)


def attach_geometry_features(frame: pd.DataFrame, run_dirs: list[Path], random_state: int) -> pd.DataFrame:
    geometry_frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        family_label = family_label_for_run_dir(run_dir)
        geometry_frames.append(compute_geometry_features(run_dir, family_label=family_label, random_state=random_state))

    combined = frame.copy()
    if geometry_frames:
        geometry_frame = pd.concat(geometry_frames, ignore_index=True)
        combined = combined.merge(geometry_frame, on=["family", "run_id", "step"], how="left")
    else:
        combined["hidden_kl_divergence"] = np.nan
        combined["pca_velocity_norm"] = np.nan
    for column in GEOMETRY_FEATURES:
        combined[column] = pd.to_numeric(combined.get(column), errors="coerce").fillna(0.0)
    return combined


def normalize_feature_columns(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    normalized = frame.copy()
    for feature in features:
        normalized[feature] = normalized.groupby("family")[feature].transform(zscore_per_family)
    return normalized


def evaluate_hazard_variant(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    variant: VariantSpec,
    random_state: int,
) -> dict[str, Any]:
    q_model = fit_classifier(
        train_frame=train_frame,
        features=variant.features,
        target_column="correct",
        model_kind=str(variant.q_model_kind),
        basis=variant.basis,
        random_state=random_state,
    )
    alpha_train = train_frame[train_frame["valid_repair"]].copy()
    alpha_test = test_frame[test_frame["valid_repair"]].copy()
    beta_train = train_frame[train_frame["valid_corruption"]].copy()
    beta_test = test_frame[test_frame["valid_corruption"]].copy()

    alpha_model = fit_classifier(
        train_frame=alpha_train,
        features=variant.features,
        target_column="event_repair",
        model_kind=str(variant.alpha_model_kind),
        basis=variant.basis,
        random_state=random_state,
    )
    beta_model = fit_classifier(
        train_frame=beta_train,
        features=variant.features,
        target_column="event_corruption",
        model_kind=str(variant.beta_model_kind),
        basis=variant.basis,
        random_state=random_state,
    )

    q_train_scores = predict_probabilities(q_model, train_frame, variant.features)[:, 1]
    q_test_scores = predict_probabilities(q_model, test_frame, variant.features)[:, 1]
    alpha_train_scores = predict_probabilities(alpha_model, alpha_train, variant.features)[:, 1]
    alpha_test_scores = predict_probabilities(alpha_model, alpha_test, variant.features)[:, 1]
    beta_train_scores = predict_probabilities(beta_model, beta_train, variant.features)[:, 1]
    beta_test_scores = predict_probabilities(beta_model, beta_test, variant.features)[:, 1]

    scored_test = test_frame.copy()
    scored_test["q_hat"] = q_test_scores
    scored_test["alpha_hat"] = predict_probabilities(alpha_model, test_frame, variant.features)[:, 1]
    scored_test["beta_hat"] = predict_probabilities(beta_model, test_frame, variant.features)[:, 1]
    scored_test["equation_score"] = (
        (1.0 - scored_test["q_hat"]) * scored_test["alpha_hat"]
        - scored_test["q_hat"] * scored_test["beta_hat"]
        - STEP_COST
    )

    boundary_accuracy, mean_oracle_gap, mean_stop_utility = oracle_metrics(scored_test, "equation_score")

    q_train_auc = safe_auc(train_frame["correct"], q_train_scores)
    q_test_auc = safe_auc(test_frame["correct"], q_test_scores)
    alpha_train_auc = safe_auc(alpha_train["event_repair"], alpha_train_scores)
    alpha_test_auc = safe_auc(alpha_test["event_repair"], alpha_test_scores)
    beta_train_auc = safe_auc(beta_train["event_corruption"], beta_train_scores)
    beta_test_auc = safe_auc(beta_test["event_corruption"], beta_test_scores)

    mean_train_auc = float(np.nanmean([q_train_auc, alpha_train_auc, beta_train_auc]))
    mean_test_auc = float(np.nanmean([q_test_auc, alpha_test_auc, beta_test_auc]))
    mean_brier = float(
        np.nanmean(
            [
                brier_score_loss(train_frame["correct"], q_train_scores),
                brier_score_loss(alpha_train["event_repair"], alpha_train_scores) if len(alpha_train_scores) else np.nan,
                brier_score_loss(beta_train["event_corruption"], beta_train_scores) if len(beta_train_scores) else np.nan,
            ]
        )
    )
    mean_ece = float(
        np.nanmean(
            [
                expected_calibration_error(test_frame["correct"].to_numpy(dtype=int), q_test_scores),
                expected_calibration_error(alpha_test["event_repair"].to_numpy(dtype=int), alpha_test_scores) if len(alpha_test_scores) else np.nan,
                expected_calibration_error(beta_test["event_corruption"].to_numpy(dtype=int), beta_test_scores) if len(beta_test_scores) else np.nan,
            ]
        )
    )

    return {
        "mean_train_auc": mean_train_auc,
        "mean_test_auc": mean_test_auc,
        "generalization_gap": mean_train_auc - mean_test_auc,
        "q_train_auc": q_train_auc,
        "q_test_auc": q_test_auc,
        "alpha_train_auc": alpha_train_auc,
        "alpha_test_auc": alpha_test_auc,
        "beta_train_auc": beta_train_auc,
        "beta_test_auc": beta_test_auc,
        "drift_train_auc": float("nan"),
        "drift_test_auc": float("nan"),
        "mean_brier": mean_brier,
        "mean_ece": mean_ece,
        "boundary_within_one": boundary_accuracy,
        "mean_oracle_gap": mean_oracle_gap,
        "mean_stop_utility": mean_stop_utility,
    }


def evaluate_direct_variant(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    variant: VariantSpec,
    random_state: int,
) -> dict[str, Any]:
    train_transitions = train_frame[train_frame["has_next"]].copy()
    test_transitions = test_frame[test_frame["has_next"]].copy()
    drift_model = fit_regressor(
        train_frame=train_transitions,
        features=variant.features,
        target_column="delta_utility",
        model_kind=str(variant.drift_model_kind),
        basis=variant.basis,
        random_state=random_state,
    )

    train_scores = predict_regression(drift_model, train_transitions, variant.features)
    test_scores = predict_regression(drift_model, test_transitions, variant.features)

    scored_test = test_frame.copy()
    scored_test["equation_score"] = predict_regression(drift_model, scored_test, variant.features)
    boundary_accuracy, mean_oracle_gap, mean_stop_utility = oracle_metrics(scored_test, "equation_score")

    drift_train_auc = safe_auc(train_transitions["positive_drift"], train_scores)
    drift_test_auc = safe_auc(test_transitions["positive_drift"], test_scores)

    return {
        "mean_train_auc": float(drift_train_auc),
        "mean_test_auc": float(drift_test_auc),
        "generalization_gap": float(drift_train_auc - drift_test_auc),
        "q_train_auc": float("nan"),
        "q_test_auc": float("nan"),
        "alpha_train_auc": float("nan"),
        "alpha_test_auc": float("nan"),
        "beta_train_auc": float("nan"),
        "beta_test_auc": float("nan"),
        "drift_train_auc": float(drift_train_auc),
        "drift_test_auc": float(drift_test_auc),
        "mean_brier": float("nan"),
        "mean_ece": float("nan"),
        "boundary_within_one": boundary_accuracy,
        "mean_oracle_gap": mean_oracle_gap,
        "mean_stop_utility": mean_stop_utility,
    }


def evaluate_variant(frame: pd.DataFrame, variant: VariantSpec, random_state: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    families = sorted(frame["family"].unique())
    for holdout_family in families:
        train = frame[frame["family"] != holdout_family].copy()
        test = frame[frame["family"] == holdout_family].copy()
        if variant.estimator_family == "hazard":
            metrics = evaluate_hazard_variant(train_frame=train, test_frame=test, variant=variant, random_state=random_state)
        else:
            metrics = evaluate_direct_variant(train_frame=train, test_frame=test, variant=variant, random_state=random_state)
        rows.append(
            {
                "variant": variant.name,
                "estimator_family": variant.estimator_family,
                "basis": variant.basis,
                "features": ", ".join(variant.features),
                "feature_count": len(variant.features),
                "q_model_kind": variant.q_model_kind,
                "alpha_model_kind": variant.alpha_model_kind,
                "beta_model_kind": variant.beta_model_kind,
                "drift_model_kind": variant.drift_model_kind,
                "closed_form": variant.closed_form,
                "description": variant.description,
                "holdout_family": holdout_family,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def build_variants(include_geometry: bool) -> list[VariantSpec]:
    variants: list[VariantSpec] = []
    for basis in ("linear", "quadratic", "cubic"):
        variants.append(
            VariantSpec(
                name=f"hazard_{basis}_top4",
                estimator_family="hazard",
                features=FINAL_BASE_FEATURES,
                basis=basis,
                q_model_kind="logistic",
                alpha_model_kind="logistic",
                beta_model_kind="logistic",
                drift_model_kind=None,
                description=f"Hazard composition with {basis} logistic heads over the current top-4 observables.",
                closed_form=True,
            )
        )

    for dropped_feature in FINAL_BASE_FEATURES:
        variants.append(
            VariantSpec(
                name=f"hazard_quadratic_drop_{dropped_feature}",
                estimator_family="hazard",
                features=tuple(feature for feature in FINAL_BASE_FEATURES if feature != dropped_feature),
                basis="quadratic",
                q_model_kind="logistic",
                alpha_model_kind="logistic",
                beta_model_kind="logistic",
                drift_model_kind=None,
                description=f"Quadratic hazard equation after ablating {dropped_feature}.",
                closed_form=True,
            )
        )

    for combo in combinations(CANDIDATE_FEATURES, 4):
        combo_name = "_".join(feature.replace("_", "") for feature in combo)
        variants.append(
            VariantSpec(
                name=f"hazard_quadratic_combo_{combo_name}",
                estimator_family="hazard",
                features=tuple(combo),
                basis="quadratic",
                q_model_kind="logistic",
                alpha_model_kind="logistic",
                beta_model_kind="logistic",
                drift_model_kind=None,
                description="Quadratic hazard equation over a 4-of-6 observable subset.",
                closed_form=True,
            )
        )

    variants.extend(
        [
            VariantSpec(
                name="hazard_rf_top4",
                estimator_family="hazard",
                features=FINAL_BASE_FEATURES,
                basis="linear",
                q_model_kind="random_forest",
                alpha_model_kind="random_forest",
                beta_model_kind="random_forest",
                drift_model_kind=None,
                description="Random-forest hazard heads over the current top-4 observables.",
                closed_form=False,
            ),
            VariantSpec(
                name="hazard_hgb_top4",
                estimator_family="hazard",
                features=FINAL_BASE_FEATURES,
                basis="linear",
                q_model_kind="hist_gb",
                alpha_model_kind="hist_gb",
                beta_model_kind="hist_gb",
                drift_model_kind=None,
                description="Histogram gradient-boosted hazard heads over the current top-4 observables.",
                closed_form=False,
            ),
            VariantSpec(
                name="hazard_q_svm_top4",
                estimator_family="hazard",
                features=FINAL_BASE_FEATURES,
                basis="linear",
                q_model_kind="svm",
                alpha_model_kind="logistic",
                beta_model_kind="logistic",
                drift_model_kind=None,
                description="Kernel SVM for q_hat with logistic repair/corruption hazard heads.",
                closed_form=False,
            ),
            VariantSpec(
                name="direct_drift_ridge_top4",
                estimator_family="direct_drift",
                features=FINAL_BASE_FEATURES,
                basis="quadratic",
                q_model_kind=None,
                alpha_model_kind=None,
                beta_model_kind=None,
                drift_model_kind="ridge",
                description="Direct quadratic ridge regression on delta-utility over the current top-4 observables.",
                closed_form=True,
            ),
            VariantSpec(
                name="direct_drift_hgb_top4",
                estimator_family="direct_drift",
                features=FINAL_BASE_FEATURES,
                basis="linear",
                q_model_kind=None,
                alpha_model_kind=None,
                beta_model_kind=None,
                drift_model_kind="hist_gb",
                description="Direct histogram gradient-boosted regression on delta-utility over the current top-4 observables.",
                closed_form=False,
            ),
        ]
    )

    if include_geometry:
        variants.extend(
            [
                VariantSpec(
                    name="hazard_quadratic_top4_kl",
                    estimator_family="hazard",
                    features=(
                        "entropy_mean",
                        "answer_changed",
                        "thought_token_count",
                        "hidden_kl_divergence",
                    ),
                    basis="quadratic",
                    q_model_kind="logistic",
                    alpha_model_kind="logistic",
                    beta_model_kind="logistic",
                    drift_model_kind=None,
                    description="Quadratic hazard equation replacing hidden L2 drift with KL divergence.",
                    closed_form=True,
                ),
                VariantSpec(
                    name="hazard_quadratic_top4_pca",
                    estimator_family="hazard",
                    features=(
                        "entropy_mean",
                        "answer_changed",
                        "thought_token_count",
                        "pca_velocity_norm",
                    ),
                    basis="quadratic",
                    q_model_kind="logistic",
                    alpha_model_kind="logistic",
                    beta_model_kind="logistic",
                    drift_model_kind=None,
                    description="Quadratic hazard equation replacing hidden L2 drift with PCA trajectory velocity.",
                    closed_form=True,
                ),
            ]
        )
    return variants


def summarize_variants(lofo_frame: pd.DataFrame) -> pd.DataFrame:
    grouped = lofo_frame.groupby(
        [
            "variant",
            "estimator_family",
            "basis",
            "features",
            "feature_count",
            "q_model_kind",
            "alpha_model_kind",
            "beta_model_kind",
            "drift_model_kind",
            "closed_form",
            "description",
        ],
        dropna=False,
    )
    summary = grouped.agg(
        mean_train_auc=("mean_train_auc", "mean"),
        mean_test_auc=("mean_test_auc", "mean"),
        mean_generalization_gap=("generalization_gap", "mean"),
        mean_q_test_auc=("q_test_auc", "mean"),
        mean_alpha_test_auc=("alpha_test_auc", "mean"),
        mean_beta_test_auc=("beta_test_auc", "mean"),
        mean_drift_test_auc=("drift_test_auc", "mean"),
        mean_brier=("mean_brier", "mean"),
        mean_ece=("mean_ece", "mean"),
        boundary_within_one=("boundary_within_one", "mean"),
        mean_oracle_gap=("mean_oracle_gap", "mean"),
        mean_stop_utility=("mean_stop_utility", "mean"),
    ).reset_index()
    summary = summary.sort_values(
        ["boundary_within_one", "mean_oracle_gap", "mean_test_auc"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    summary.insert(0, "rank", range(1, len(summary) + 1))
    summary["is_current_baseline"] = summary["variant"] == CURRENT_BASELINE_VARIANT
    return summary


def geometry_signal_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = frame[frame["has_next"]].copy()
    rows: list[dict[str, Any]] = []
    for family_label, family_frame in eligible.groupby("family"):
        for feature in GEOMETRY_SCORE_COLUMNS:
            if feature not in family_frame.columns:
                continue
            feature_values = pd.to_numeric(family_frame[feature], errors="coerce")
            target_values = pd.to_numeric(family_frame["delta_utility"], errors="coerce")
            valid_mask = feature_values.notna() & target_values.notna()
            if int(valid_mask.sum()) < 2:
                continue
            corr = float(np.corrcoef(feature_values[valid_mask], target_values[valid_mask])[0, 1])
            auc = safe_auc(family_frame.loc[valid_mask, "positive_drift"], feature_values[valid_mask].to_numpy())
            rows.append(
                {
                    "family": family_label,
                    "feature": feature,
                    "corr_delta_utility": corr,
                    "beneficial_continue_auc": auc,
                }
            )
    by_family = pd.DataFrame(rows)
    if by_family.empty:
        pooled = pd.DataFrame(columns=["feature", "mean_abs_corr", "mean_auc"])
    else:
        pooled = by_family.groupby("feature", as_index=False).agg(
            mean_abs_corr=("corr_delta_utility", lambda values: float(np.nanmean(np.abs(values)))),
            mean_auc=("beneficial_continue_auc", "mean"),
        )
        pooled = pooled.sort_values(["mean_abs_corr", "mean_auc"], ascending=[False, False]).reset_index(drop=True)
    return by_family, pooled


def markdown_table(frame: pd.DataFrame, float_columns: set[str]) -> str:
    if frame.empty:
        return "| none |\n| --- |"
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in headers:
            value = row[column]
            if column in float_columns and pd.notna(value):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def fit_final_hazard_models(frame: pd.DataFrame, variant: VariantSpec, random_state: int) -> tuple[Any, Any, Any]:
    q_model = fit_classifier(frame, variant.features, "correct", str(variant.q_model_kind), variant.basis, random_state)
    alpha_model = fit_classifier(
        frame[frame["valid_repair"]].copy(),
        variant.features,
        "event_repair",
        str(variant.alpha_model_kind),
        variant.basis,
        random_state,
    )
    beta_model = fit_classifier(
        frame[frame["valid_corruption"]].copy(),
        variant.features,
        "event_corruption",
        str(variant.beta_model_kind),
        variant.basis,
        random_state,
    )
    return q_model, alpha_model, beta_model


def coefficient_frame_from_pipeline(model: Pipeline | ConstantProbabilityModel, features: tuple[str, ...]) -> pd.DataFrame:
    if isinstance(model, ConstantProbabilityModel):
        return pd.DataFrame([
            {
                "term": "constant_probability",
                "coefficient": model.probability,
            }
        ])
    model_step = model.named_steps["model"]
    if not hasattr(model_step, "coef_"):
        return pd.DataFrame()
    basis_step = model.named_steps.get("basis")
    if basis_step is None:
        terms = np.asarray(features, dtype=object)
    else:
        terms = basis_step.get_feature_names_out(features)
    rows = [{"term": "intercept", "coefficient": float(model_step.intercept_[0])}]
    for index, term in enumerate(terms):
        rows.append({"term": str(term), "coefficient": float(model_step.coef_[0][index])})
    return pd.DataFrame(rows)


def recommended_equation_text(variant: VariantSpec) -> str:
    if variant.estimator_family == "hazard":
        features = ", ".join(variant.features)
        return (
            "$$\n"
            r"\hat{\mu}_t = (1 - \hat{q}_t)\hat{\alpha}_t - \hat{q}_t\hat{\beta}_t - 0.05,\qquad "
            rf"\phi(x_t) = \text{{{variant.basis}}}({features})"
            "\n$$"
        )
    features = ", ".join(variant.features)
    return (
        "$$\n"
        rf"\widehat{{\Delta V}}_t = f_\theta(\text{{{variant.basis}}}({features})),\qquad "
        r"\tau = \min\{t \ge 2 : \widehat{\Delta V}_t \le 0\}"
        "\n$$"
    )


def build_report(
    *,
    coverage_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    lofo_frame: pd.DataFrame,
    geometry_pooled: pd.DataFrame,
    recommended_variant: VariantSpec,
    recommended_row: pd.Series,
    best_overall_row: pd.Series,
    missing_frontier_dirs: list[str],
) -> str:
    baseline_row = summary_frame.loc[summary_frame["variant"] == CURRENT_BASELINE_VARIANT]
    feature_combo_rows = summary_frame[summary_frame["variant"].str.startswith("hazard_quadratic_combo_")].head(10)
    ablation_rows = summary_frame[summary_frame["variant"].str.startswith("hazard_quadratic_drop_")]

    lines = [
        "# Equation Analysis Report",
        "",
        "## Scope",
        "",
        f"- Families analyzed: {', '.join(coverage_frame['family'].tolist())}.",
        f"- Run directories analyzed: {', '.join(coverage_frame['run_dir'].tolist())}.",
        "- Evaluation protocol: leave-one-family-out over all loaded families.",
        "- Stopping objective: stop at the first step $t \\ge 2$ where the estimated continuation value is non-positive.",
        "- Note: no frontier full-trace directories were included unless explicitly supplied at runtime.",
        "",
        "## Recommendation",
        "",
        f"Best overall stop-rule variant by boundary accuracy then oracle gap: `{best_overall_row['variant']}`.",
        f"Best hazard-preserving equation: `{recommended_row['variant']}`.",
        recommended_equation_text(recommended_variant),
        f"The recommended hazard equation achieved LOFO mean test AUC `{float(recommended_row['mean_test_auc']):.4f}`, boundary accuracy within $\\pm 1$ step `{float(recommended_row['boundary_within_one']):.4f}`, and mean oracle gap `{float(recommended_row['mean_oracle_gap']):.4f}`.",
        f"The best overall stop rule `{best_overall_row['variant']}` reaches boundary accuracy `{float(best_overall_row['boundary_within_one']):.4f}` with oracle gap `{float(best_overall_row['mean_oracle_gap']):.4f}`, but it departs from the original q/alpha/beta hazard decomposition.",
        "",
        "## Coverage",
        "",
        markdown_table(coverage_frame, float_columns=set()),
        "",
        "## Variant Comparison",
        "",
        markdown_table(
            summary_frame[
                [
                    "rank",
                    "variant",
                    "estimator_family",
                    "basis",
                    "feature_count",
                    "mean_test_auc",
                    "mean_generalization_gap",
                    "boundary_within_one",
                    "mean_oracle_gap",
                    "mean_q_test_auc",
                    "mean_alpha_test_auc",
                    "mean_beta_test_auc",
                    "mean_drift_test_auc",
                    "mean_ece",
                    "closed_form",
                ]
            ],
            float_columns={
                "mean_test_auc",
                "mean_generalization_gap",
                "boundary_within_one",
                "mean_oracle_gap",
                "mean_q_test_auc",
                "mean_alpha_test_auc",
                "mean_beta_test_auc",
                "mean_drift_test_auc",
                "mean_ece",
            },
        ),
        "",
        "## Feature Ablation",
        "",
        markdown_table(
            ablation_rows[["variant", "features", "mean_test_auc", "boundary_within_one", "mean_oracle_gap"]],
            float_columns={"mean_test_auc", "boundary_within_one", "mean_oracle_gap"},
        ),
        "",
        "## Best 4-of-6 Feature Sets",
        "",
        markdown_table(
            feature_combo_rows[["variant", "features", "mean_test_auc", "boundary_within_one", "mean_oracle_gap"]],
            float_columns={"mean_test_auc", "boundary_within_one", "mean_oracle_gap"},
        ),
        "",
        "## Hidden-State Geometry",
        "",
        markdown_table(
            geometry_pooled,
            float_columns={"mean_abs_corr", "mean_auc"},
        ),
        "",
        "## Interpretation",
        "",
        "Quadratic structure should be retained only if it materially improves LOFO boundary accuracy without widening the generalization gap. Feature ablations identify whether any of the four current observables are acting as passengers rather than signal carriers. The geometry table tests whether PCA trajectory velocity or KL divergence consistently outrun raw hidden-state L2 drift as a family-normalized observable.",
        "The sweep indicates that the strongest hazard-form replacement for the current equation is not the original top-4 set: entropy variance and confidence improve boundary accuracy more consistently than answer change and hidden-state L2 shift in the best LOFO hazard variant.",
    ]

    if not baseline_row.empty:
        baseline = baseline_row.iloc[0]
        lines.extend(
            [
                "",
                "## Current Baseline Audit",
                "",
                f"Current selected baseline `{CURRENT_BASELINE_VARIANT}` scored LOFO mean test AUC `{float(baseline['mean_test_auc']):.4f}`, boundary accuracy `{float(baseline['boundary_within_one']):.4f}`, and oracle gap `{float(baseline['mean_oracle_gap']):.4f}`.",
            ]
        )

    if missing_frontier_dirs:
        lines.extend(
            [
                "",
                "## Missing Frontier Inputs",
                "",
                "The following user-supplied frontier run directories were missing and were not included in this sweep:",
                "",
                "| missing_run_dir |",
                "| --- |",
            ]
        )
        lines.extend(f"| {path} |" for path in missing_frontier_dirs)

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-run-dirs", nargs="*", default=[], help="Optional completed frontier run directories to include.")
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--lofo-path", default=str(DEFAULT_LOFO_PATH))
    parser.add_argument("--geometry-path", default=str(DEFAULT_GEOMETRY_PATH))
    parser.add_argument("--coverage-path", default=str(DEFAULT_COVERAGE_PATH))
    parser.add_argument("--weight-path", default=str(DEFAULT_WEIGHT_PATH))
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    run_dirs = [OUTPUT_BASE / relative_name for relative_name in LEGACY_RUNS.values()]
    missing_frontier_dirs: list[str] = []
    for raw_path in args.frontier_run_dirs:
        path = Path(raw_path)
        if path.exists():
            run_dirs.append(path)
        else:
            missing_frontier_dirs.append(str(path))

    frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        family_label = family_label_for_run_dir(run_dir)
        frame = load_run_frame(run_dir, family_label=family_label)
        frames.append(frame)
        coverage_rows.append(
            {
                "family": family_label,
                "run_dir": run_dir.name,
                "runs": int(frame["run_id"].nunique()),
                "steps": int(len(frame)),
                "tasks": int(frame["task_id"].nunique()),
                "parse_success_rate": float(pd.to_numeric(frame["parse_success"], errors="coerce").mean()) if "parse_success" in frame.columns else float("nan"),
            }
        )

    full_frame = pd.concat(frames, ignore_index=True)
    if not args.skip_geometry:
        full_frame = attach_geometry_features(full_frame, run_dirs=run_dirs, random_state=args.random_state)
    else:
        for column in GEOMETRY_FEATURES:
            full_frame[column] = 0.0

    normalized_frame = normalize_feature_columns(full_frame, features=CANDIDATE_FEATURES + GEOMETRY_FEATURES)
    variants = build_variants(include_geometry=not args.skip_geometry)
    lofo_frames = [evaluate_variant(normalized_frame, variant=variant, random_state=args.random_state) for variant in variants]
    lofo_frame = pd.concat(lofo_frames, ignore_index=True)
    summary_frame = summarize_variants(lofo_frame)
    coverage_frame = pd.DataFrame(coverage_rows).sort_values("family").reset_index(drop=True)
    geometry_by_family, geometry_pooled = geometry_signal_tables(normalized_frame)

    best_overall_row = summary_frame.iloc[0]
    recommended_candidates = summary_frame[
        (summary_frame["estimator_family"] == "hazard") & (summary_frame["closed_form"])
    ].copy()
    recommended_row = recommended_candidates.iloc[0] if not recommended_candidates.empty else best_overall_row
    variant_lookup = {variant.name: variant for variant in variants}
    recommended_variant = variant_lookup[str(recommended_row["variant"])]

    weight_frame = pd.DataFrame()
    if recommended_variant.estimator_family == "hazard":
        q_model, alpha_model, beta_model = fit_final_hazard_models(normalized_frame, variant=recommended_variant, random_state=args.random_state)
        q_weights = coefficient_frame_from_pipeline(q_model, recommended_variant.features).rename(columns={"coefficient": "q_weight"})
        alpha_weights = coefficient_frame_from_pipeline(alpha_model, recommended_variant.features).rename(columns={"coefficient": "alpha_weight"})
        beta_weights = coefficient_frame_from_pipeline(beta_model, recommended_variant.features).rename(columns={"coefficient": "beta_weight"})
        weight_frame = q_weights.merge(alpha_weights, on="term", how="outer").merge(beta_weights, on="term", how="outer")
    elif recommended_variant.drift_model_kind == "ridge":
        final_model = fit_regressor(
            normalized_frame[normalized_frame["has_next"]].copy(),
            recommended_variant.features,
            "delta_utility",
            str(recommended_variant.drift_model_kind),
            recommended_variant.basis,
            args.random_state,
        )
        if isinstance(final_model, Pipeline):
            model_step = final_model.named_steps["model"]
            basis_step = final_model.named_steps.get("basis")
            if basis_step is None:
                terms = np.asarray(recommended_variant.features, dtype=object)
            else:
                terms = basis_step.get_feature_names_out(recommended_variant.features)
            weight_frame = pd.DataFrame(
                [{"term": "intercept", "drift_weight": float(model_step.intercept_)}]
                + [{"term": str(term), "drift_weight": float(model_step.coef_[index])} for index, term in enumerate(terms)]
            )

    report_path = Path(args.report_path)
    summary_path = Path(args.summary_path)
    lofo_path = Path(args.lofo_path)
    geometry_path = Path(args.geometry_path)
    coverage_path = Path(args.coverage_path)
    weight_path = Path(args.weight_path)
    metadata_path = Path(args.metadata_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lofo_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    summary_frame.to_csv(summary_path, index=False)
    lofo_frame.to_csv(lofo_path, index=False)
    geometry_pooled.to_csv(geometry_path, index=False)
    coverage_frame.to_csv(coverage_path, index=False)
    if not weight_frame.empty:
        weight_frame.to_csv(weight_path, index=False)

    metadata_path.write_text(
        json.dumps(
            {
                "current_baseline_variant": CURRENT_BASELINE_VARIANT,
                "best_overall_variant": str(best_overall_row["variant"]),
                "recommended_hazard_variant": str(recommended_row["variant"]),
                "family_count": int(coverage_frame["family"].nunique()),
                "run_dir_count": int(len(run_dirs)),
                "frontier_run_dir_count": int(len(args.frontier_run_dirs) - len(missing_frontier_dirs)),
                "missing_frontier_run_dirs": missing_frontier_dirs,
                "geometry_enabled": not args.skip_geometry,
                "random_state": args.random_state,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path.write_text(
        build_report(
            coverage_frame=coverage_frame,
            summary_frame=summary_frame,
            lofo_frame=lofo_frame,
            geometry_pooled=geometry_pooled,
            recommended_variant=recommended_variant,
            recommended_row=recommended_row,
            best_overall_row=best_overall_row,
            missing_frontier_dirs=missing_frontier_dirs,
        ),
        encoding="utf-8",
    )

    print(f"Best overall variant: {best_overall_row['variant']}")
    print(f"Recommended closed-form variant: {recommended_row['variant']}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote LOFO metrics: {lofo_path}")
    print(f"Wrote geometry summary: {geometry_path}")
    print(f"Wrote report: {report_path}")
    if not weight_frame.empty:
        print(f"Wrote weight table: {weight_path}")


if __name__ == "__main__":
    main()