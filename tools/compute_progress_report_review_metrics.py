"""Recompute the statistical tables added after advisor review of the progress report.

The script uses only committed experiment artifacts. It produces:

* task-cluster bootstrap intervals for empirical correctness, repair, corruption,
  and net-drift curves;
* cell-cluster bootstrap intervals and explicit denominators for N1-N6; and
* task-macro, domain-macro, per-domain, and outer-fold tournament metrics.

Run from the repository root:

    python tools/compute_progress_report_review_metrics.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
MATRIX = RESEARCH / "outputs" / "experiment_matrix"
V2 = RESEARCH / "outputs" / "experiments_v2"
OUTPUT = RESEARCH / "outputs" / "progress_report_review_metrics"
STEP_COST = 0.05
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 804

SELECTED_BOUNDARY_CELLS = {
    "qwen2p5_7b__gsm8k": range(2, 7),
    "qwen2p5_32b__math": range(2, 8),
    "qwen2p5_7b__arc": (2, 4, 6),
    "qwen2p5_7b__gpqa": (2, 4, 6),
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def label_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
    return float(low), float(high)


def cluster_ratio_bootstrap(
    aggregates: np.ndarray,
    numerator_column: int,
    denominator_column: int,
    label: str,
) -> tuple[float, float]:
    rng = np.random.default_rng(label_seed(label))
    cluster_count = len(aggregates)
    draws: list[np.ndarray] = []
    for start in range(0, BOOTSTRAP_DRAWS, 1_000):
        size = min(1_000, BOOTSTRAP_DRAWS - start)
        indices = rng.integers(0, cluster_count, size=(size, cluster_count))
        sample = aggregates[indices].sum(axis=1)
        denominator = sample[:, denominator_column]
        ratio = np.divide(
            sample[:, numerator_column],
            denominator,
            out=np.full(size, np.nan, dtype=float),
            where=denominator != 0,
        )
        draws.append(ratio)
    values = np.concatenate(draws)
    values = values[np.isfinite(values)]
    return percentile_interval(values)


def load_boundary_frame() -> pd.DataFrame:
    trace_analysis = load_module("progress_trace_analysis", RESEARCH / "trace_analysis.py")
    frames: list[pd.DataFrame] = []
    for path in sorted(MATRIX.glob("*/trace_steps.csv")):
        if not (path.parent / "detector_comparison_by_run.csv").exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame, _ = trace_analysis._sanitize_step_frame(frame)
        model, domain = path.parent.name.rsplit("__", 1)
        frame = frame[["run_id", "task_id", "step", "correct"]].copy()
        frame["model"] = model
        frame["domain"] = domain
        frame["cell"] = path.parent.name
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["correct"] = pd.to_numeric(result["correct"], errors="coerce").fillna(0).astype(int)
    result["step"] = pd.to_numeric(result["step"], errors="raise").astype(int)
    result = result.sort_values(["cell", "run_id", "step"], kind="stable")
    result["next_correct"] = result.groupby(["cell", "run_id"], sort=False)["correct"].shift(-1)
    result = result[result["next_correct"].notna()].copy()
    result["wrong_at_t"] = (result["correct"] == 0).astype(int)
    result["correct_at_t"] = (result["correct"] == 1).astype(int)
    result["repair"] = ((result["correct"] == 0) & (result["next_correct"] == 1)).astype(int)
    result["corruption"] = ((result["correct"] == 1) & (result["next_correct"] == 0)).astype(int)
    result["net_drift"] = result["next_correct"] - result["correct"] - STEP_COST
    return result


def boundary_statistics(group: pd.DataFrame, label: str) -> dict[str, Any]:
    clustered = (
        group.groupby("task_id", sort=False)
        .agg(
            n=("correct", "size"),
            correct=("correct", "sum"),
            wrong_at_t=("wrong_at_t", "sum"),
            correct_at_t=("correct_at_t", "sum"),
            repair=("repair", "sum"),
            corruption=("corruption", "sum"),
            net_drift=("net_drift", "sum"),
        )
        .to_numpy(dtype=float)
    )
    # Columns: n, correct, wrong, currently-correct, repair, corruption, drift.
    q_low, q_high = cluster_ratio_bootstrap(clustered, 1, 0, f"{label}|q")
    repair_low, repair_high = cluster_ratio_bootstrap(clustered, 4, 2, f"{label}|repair")
    corruption_low, corruption_high = cluster_ratio_bootstrap(clustered, 5, 3, f"{label}|corruption")
    drift_low, drift_high = cluster_ratio_bootstrap(clustered, 6, 0, f"{label}|drift")
    wrong_count = int(group["wrong_at_t"].sum())
    correct_count = int(group["correct_at_t"].sum())
    return {
        "trajectories": int(len(group)),
        "task_clusters": int(group["task_id"].nunique()),
        "accuracy": float(group["correct"].mean()),
        "accuracy_ci_low": q_low,
        "accuracy_ci_high": q_high,
        "repair_events": int(group["repair"].sum()),
        "repair_denominator": wrong_count,
        "repair_hazard": float(group["repair"].sum() / wrong_count),
        "repair_ci_low": repair_low,
        "repair_ci_high": repair_high,
        "corruption_events": int(group["corruption"].sum()),
        "corruption_denominator": correct_count,
        "corruption_hazard": float(group["corruption"].sum() / correct_count),
        "corruption_ci_low": corruption_low,
        "corruption_ci_high": corruption_high,
        "net_drift": float(group["net_drift"].mean()),
        "net_drift_ci_low": drift_low,
        "net_drift_ci_high": drift_high,
    }


def write_boundary_outputs(frame: pd.DataFrame) -> None:
    domain_rows: list[dict[str, Any]] = []
    for (domain, step), group in frame.groupby(["domain", "step"], sort=True):
        row = {"domain": domain, "step": int(step)}
        row.update(boundary_statistics(group, f"domain|{domain}|{step}"))
        domain_rows.append(row)
    pd.DataFrame(domain_rows).to_csv(OUTPUT / "boundary_domain_step_metrics.csv", index=False)

    selected_rows: list[dict[str, Any]] = []
    for cell, steps in SELECTED_BOUNDARY_CELLS.items():
        for step in steps:
            group = frame[(frame["cell"] == cell) & (frame["step"] == step)]
            if group.empty:
                raise ValueError(f"Missing selected boundary stratum {cell}, step {step}")
            model, domain = cell.rsplit("__", 1)
            row = {"cell": cell, "model": model, "domain": domain, "step": int(step)}
            row.update(boundary_statistics(group, f"cell|{cell}|{step}"))
            selected_rows.append(row)
    pd.DataFrame(selected_rows).to_csv(OUTPUT / "boundary_selected_cell_step_metrics.csv", index=False)


def weighted_cell_bootstrap(
    effects: np.ndarray,
    counts: np.ndarray,
    label: str,
) -> tuple[float, float]:
    rng = np.random.default_rng(label_seed(label))
    indices = rng.integers(0, len(effects), size=(BOOTSTRAP_DRAWS, len(effects)))
    means = effects[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return percentile_interval(means)


def n1_contributions(cache_rows: list[dict[str, Any]], groups: list[Any]) -> np.ndarray:
    taus = np.round(np.arange(-0.21, 0.2101, 0.005), 3)
    deltas = np.round(np.arange(-0.15, 0.1501, 0.005), 3)

    def tau_index(value: float) -> int:
        return int(round((value - taus[0]) / 0.005))

    feature_names = ["step1_acc", "step2_acc", "churn_rate", "mean_entropy", "mean_len"]
    domains = sorted({row["cell"].rsplit("__", 1)[1] for row in cache_rows})
    features = np.asarray(
        [
            [row["features"][name] for name in feature_names]
            + [float(row["cell"].rsplit("__", 1)[1] == domain) for domain in domains]
            for row in cache_rows
        ],
        dtype=float,
    )
    targets = np.asarray([row["delta_full"] for row in cache_rows], dtype=float)
    profiles = np.asarray([row["dU_profile"] for row in cache_rows], dtype=float)
    result = np.zeros(len(cache_rows), dtype=float)
    for held_group in sorted(set(groups), key=str):
        test = np.asarray([index for index, group in enumerate(groups) if group == held_group])
        test_set = set(test.tolist())
        train = np.asarray([index for index in range(len(cache_rows)) if index not in test_set])
        scaler = StandardScaler().fit(features[train])
        model = Ridge(alpha=1.0).fit(scaler.transform(features[train]), targets[train])
        prediction = np.clip(
            model.predict(scaler.transform(features[test])),
            deltas[0],
            deltas[-1],
        )
        for local_index, cell_index in enumerate(test):
            rounded = round(round(prediction[local_index] / 0.005) * 0.005, 3)
            result[cell_index] = profiles[cell_index, tau_index(rounded)]
    return result


def task_cluster_effect_interval(frame: pd.DataFrame, effect_column: str, label: str) -> tuple[float, float]:
    aggregates = (
        frame.groupby("task_id", sort=False)[effect_column]
        .agg(["sum", "count"])
        .to_numpy(dtype=float)
    )
    # Reorder to denominator, numerator for the generic ratio helper.
    matrix = aggregates[:, [1, 0]]
    return cluster_ratio_bootstrap(matrix, 1, 0, label)


def load_policy_comparison(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path / "detector_comparison_by_run.csv")
    pivot = frame[frame["detector"].isin(["hazard_drift", "never_stop"])].pivot(
        index="run_id",
        columns="detector",
        values="stop_utility",
    )
    pivot["loss"] = (pivot["hazard_drift"] < pivot["never_stop"]).astype(int)
    pivot["delta_utility"] = pivot["hazard_drift"] - pivot["never_stop"]
    return pivot


def algorithm_row(
    *,
    experiment: str,
    design: str,
    corpus: str,
    experimental_unit: str,
    n_units: int,
    task_clusters: int,
    folds: str,
    replications: str,
    comparison: str,
    raw_aggregate: str,
    effect_total: float | None,
    mean_effect: float,
    ci_low: float,
    ci_high: float,
    effect_unit: str,
    ci_method: str,
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "design": design,
        "corpus": corpus,
        "experimental_unit": experimental_unit,
        "n_units": n_units,
        "task_clusters": task_clusters,
        "folds_or_holdouts": folds,
        "generation_replications": replications,
        "comparison": comparison,
        "raw_aggregate": raw_aggregate,
        "controlled_effect_total": effect_total,
        "mean_controlled_effect": mean_effect,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "effect_unit": effect_unit,
        "ci_method": ci_method,
    }


def write_algorithm_outputs() -> None:
    n1_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((V2 / "algov2_cache").glob("*.json"))
    ]
    n1_rows.sort(key=lambda row: row["cell"])
    n23_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((V2 / "algov2_cache_n2_n3").glob("*.json"))
    ]
    n23_rows.sort(key=lambda row: row["cell"])
    if [row["cell"] for row in n1_rows] != [row["cell"] for row in n23_rows]:
        raise ValueError("N1/N4 and N2/N3 cache cells do not align")
    counts = np.asarray([row["n"] for row in n1_rows], dtype=float)
    if int(counts.sum()) != 75_965:
        raise ValueError(f"Unexpected canonical trajectory count {counts.sum()}")
    models = [row["cell"].rsplit("__", 1)[0] for row in n1_rows]
    loco = n1_contributions(n1_rows, list(range(len(n1_rows))))
    lomo = n1_contributions(n1_rows, models)
    baseline = np.asarray([row["baseline"]["du_1param"] for row in n23_rows], dtype=float)
    n2a = np.asarray([row["n2a"]["du_1param"] for row in n23_rows], dtype=float)
    n2b = np.asarray([row["n2b"]["du_1param"] for row in n23_rows], dtype=float)
    n2c = np.asarray([row["n2c"]["du_1param"] for row in n23_rows], dtype=float)
    n3 = np.asarray([row["n3"]["du_1param"] for row in n23_rows], dtype=float)
    n4 = np.asarray(
        [row["n4"]["du_2param"] - row["n4"]["du_1param"] for row in n1_rows],
        dtype=float,
    )

    rows: list[dict[str, Any]] = []
    common = {
        "corpus": "52 model-domain cells; 75,965 trajectories; 1,948 unique tasks",
        "experimental_unit": "trajectory; model-domain cell is the uncertainty cluster",
        "n_units": 75_965,
        "task_clusters": 1_948,
        "replications": "seed 7 at temperatures 0.1, 0.6, and 1.0; one deterministic analysis pass",
        "ci_method": "10,000-draw percentile bootstrap over 52 model-domain cells",
        "effect_unit": "step utility per trajectory",
    }

    def add_cell_effect(
        experiment: str,
        design: str,
        folds: str,
        comparison: str,
        raw_aggregate: str,
        effect: np.ndarray,
    ) -> None:
        low, high = weighted_cell_bootstrap(effect, counts, experiment)
        rows.append(
            algorithm_row(
                experiment=experiment,
                design=design,
                folds=folds,
                comparison=comparison,
                raw_aggregate=raw_aggregate,
                effect_total=float(effect.sum()),
                mean_effect=float(effect.sum() / counts.sum()),
                ci_low=low,
                ci_high=high,
                **common,
            )
        )

    add_cell_effect(
        "N1 LOCO",
        "Ridge threshold meta-calibration with one held-out model-domain cell",
        "52 leave-one-cell-out evaluations after 5-fold within-cell OOF probe fitting",
        "predicted threshold versus frozen hazard-drift threshold 0",
        f"{loco.sum():+.2f} summed utility delta; 59.5% of +1,545.85 P3d target",
        loco,
    )
    add_cell_effect(
        "N1 LOMO",
        "Ridge threshold meta-calibration with all domains of one model held out",
        "13 leave-one-model-out evaluations after 5-fold within-cell OOF probe fitting",
        "predicted threshold versus frozen hazard-drift threshold 0",
        f"{lomo.sum():+.2f} summed utility delta; 63.2% of +1,545.85 P3d target",
        lomo,
    )
    add_cell_effect(
        "N2a",
        "Gradient-boosted probe substituted for matched logistic probe",
        "5 run-group OOF folds plus 5 task-group threshold folds per cell",
        "gradient boosting versus matched logistic baseline",
        f"{n2a.sum():+.2f} total versus hazard drift; controlled contrast {(n2a - baseline).sum():+.2f}",
        n2a - baseline,
    )
    add_cell_effect(
        "N2b",
        "Isotonic calibration added inside training folds",
        "5 run-group OOF folds plus 5 task-group threshold folds per cell",
        "isotonic-calibrated probe versus matched logistic baseline",
        f"{n2b.sum():+.2f} total versus hazard drift; controlled contrast {(n2b - baseline).sum():+.2f}",
        n2b - baseline,
    )
    add_cell_effect(
        "N2c",
        "Lag-1 and lag-2 temporal features added to the logistic probe",
        "5 run-group OOF folds plus 5 task-group threshold folds per cell",
        "lagged logistic probe versus matched unlagged logistic baseline",
        f"{n2c.sum():+.2f} total versus hazard drift; controlled contrast {(n2c - baseline).sum():+.2f}",
        n2c - baseline,
    )
    add_cell_effect(
        "N3",
        "Empirical-Bayes step hazards replace cell-local logistic hazards",
        "5 run-group OOF folds plus 5 task-group threshold folds per cell",
        "shrunk hazards versus matched cell-local logistic baseline",
        f"{n3.sum():+.2f} total versus hazard drift; controlled contrast {(n3 - baseline).sum():+.2f}",
        n3 - baseline,
    )
    add_cell_effect(
        "N4",
        "Step-2 churn modulates the calibrated threshold",
        "5 task-group folds per cell on identical trajectories",
        "two-parameter threshold versus matched one-parameter threshold",
        f"+1700.00 two-parameter total; controlled contrast {n4.sum():+.2f}",
        n4,
    )

    # N5: paired 256-token and 512-token arms.
    n5_control = load_policy_comparison(MATRIX / "mistral_small_24b_2409__gsm8k")
    n5_treatment = load_policy_comparison(V2 / "p4b_mistral_small_24b_2409__gsm8k_tok512")
    n5_tasks = (
        pd.read_csv(
            V2 / "p4b_mistral_small_24b_2409__gsm8k_tok512" / "trace_steps.csv",
            usecols=["run_id", "task_id"],
        )
        .drop_duplicates("run_id")
        .set_index("run_id")
    )
    n5 = (
        n5_control.join(n5_treatment, lsuffix="_256", rsuffix="_512", how="inner")
        .join(n5_tasks)
        .reset_index()
    )
    n5["loss_difference"] = n5["loss_512"] - n5["loss_256"]
    n5_low, n5_high = task_cluster_effect_interval(n5, "loss_difference", "N5")
    rows.append(
        algorithm_row(
            experiment="N5",
            design="Paired 256-token versus 512-token generation cap",
            corpus="Mistral-Small-22B on GSM8K; 500 tasks x 3 temperatures per arm",
            experimental_unit="matched trajectory",
            n_units=int(len(n5)),
            task_clusters=int(n5["task_id"].nunique()),
            folds="no fitted model; paired arm comparison",
            replications="seed 7 at temperatures 0.1, 0.6, and 1.0; one run per task-temperature",
            comparison="512-token loss indicator minus matched 256-token loss indicator",
            raw_aggregate=(
                f"{int(n5['loss_256'].sum())}/{len(n5)} losses in each arm; "
                f"{int((n5['loss_difference'] != 0).sum())} discordant pairs"
            ),
            effect_total=float(n5["loss_difference"].sum()),
            mean_effect=float(n5["loss_difference"].mean()),
            ci_low=n5_low,
            ci_high=n5_high,
            effect_unit="loss-risk difference",
            ci_method="10,000-draw percentile bootstrap over 500 task clusters",
        )
    )

    # N6: paired BF16 and 4-bit step-2 correctness.
    def step_two(path: Path, suffix: str) -> pd.DataFrame:
        frame = pd.read_csv(path / "trace_steps.csv", usecols=["run_id", "task_id", "step", "correct"])
        frame = frame[frame["step"] == 2].drop(columns="step").set_index("run_id")
        return frame.rename(columns={"task_id": f"task_id_{suffix}", "correct": f"correct_{suffix}"})

    n6 = step_two(V2 / "p8_qwen7b_none", "bf16").join(
        step_two(V2 / "p8_qwen7b_4bit", "4bit"),
        how="inner",
    )
    if not (n6["task_id_bf16"] == n6["task_id_4bit"]).all():
        raise ValueError("N6 task IDs do not align")
    n6 = n6.rename(columns={"task_id_bf16": "task_id"}).reset_index()
    n6["accuracy_difference"] = n6["correct_bf16"] - n6["correct_4bit"]
    n6_low, n6_high = task_cluster_effect_interval(n6, "accuracy_difference", "N6")
    rows.append(
        algorithm_row(
            experiment="N6",
            design="Paired BF16 versus 4-bit precision",
            corpus="Qwen2.5-7B on GSM8K; 500 tasks x 3 temperatures per arm",
            experimental_unit="matched trajectory at step 2",
            n_units=int(len(n6)),
            task_clusters=int(n6["task_id"].nunique()),
            folds="no fitted model; paired arm comparison",
            replications="seed 7 at temperatures 0.1, 0.6, and 1.0; one run per task-temperature",
            comparison="BF16 correctness minus matched 4-bit correctness at step 2",
            raw_aggregate=(
                f"{int(n6['correct_bf16'].sum())}/{len(n6)} BF16 correct versus "
                f"{int(n6['correct_4bit'].sum())}/{len(n6)} 4-bit correct; Z=9.79"
            ),
            effect_total=float(n6["accuracy_difference"].sum()),
            mean_effect=float(n6["accuracy_difference"].mean()),
            ci_low=n6_low,
            ci_high=n6_high,
            effect_unit="accuracy difference",
            ci_method="10,000-draw percentile bootstrap over 500 task clusters",
        )
    )
    pd.DataFrame(rows).to_csv(OUTPUT / "algorithm_v2_normalized_effects.csv", index=False)


def fold_t_interval(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    half_width = float(
        student_t.ppf(0.975, len(array) - 1)
        * array.std(ddof=1)
        / math.sqrt(len(array))
    )
    mean = float(array.mean())
    return mean - half_width, mean + half_width


def write_tournament_outputs() -> None:
    tournament = load_module("progress_ultimate_tournament", RESEARCH / "run_ultimate_multi_day_tournament.py")
    args = Namespace(input_dir=str(V2), include_all_cells=False, max_cells=None)
    frame, manifest = tournament.load_trace_frame(args)
    dummy = pd.DataFrame({"dummy": np.zeros(len(frame), dtype=np.float32)}, index=frame.index)
    store = tournament.build_sequence_store(frame, dummy)
    predictions = np.load(V2 / "ultimate_oof_predictions.npz")
    model_keys = [name.split("__", 1)[1] for name in predictions.files if name.startswith("q__")]
    all_indices = np.arange(store.n_runs)
    folds = np.full(store.n_runs, -1, dtype=int)
    splitter = GroupKFold(n_splits=5)
    for fold, (_, test) in enumerate(splitter.split(all_indices, groups=store.task_ids), start=1):
        folds[test] = fold
    run_metadata = pd.DataFrame(
        {
            "trajectory_id": store.trajectory_ids,
            "task_id": store.task_ids,
            "source_cell": store.source_cells,
            "fold": folds,
        }
    )
    run_metadata["domain"] = run_metadata["source_cell"].str.extract(
        r"_(arc|gpqa|gsm8k|math)$",
        expand=False,
    )
    if run_metadata["domain"].isna().any():
        raise ValueError("Unable to parse a tournament domain")
    display = {
        "linear": "Task-grouped linear baseline",
        "beta": "Truncated Beta mixture",
        "gru": "Causal GRU",
        "tcn": "Causal residual TCN",
        "ssm": "Selective SSM",
        "transformer": "Causal RoPE transformer",
        "fno": "Causal Fourier neural operator",
        "moe": "Five-expert causal MoE",
        "moe_hysteresis": "MoE plus hysteresis",
    }
    domains = ["arc", "gpqa", "gsm8k", "math"]
    summary_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    valid_mask = store.valid_mask_np()

    for key in model_keys + ["moe_hysteresis"]:
        source_key = "moe" if key == "moe_hysteresis" else key
        q = predictions[f"q__{source_key}"]
        repair = predictions[f"repair__{source_key}"]
        corruption = predictions[f"corruption__{source_key}"]
        policy = tournament.evaluate_stopping_policy(
            store,
            all_indices,
            q,
            repair,
            corruption,
            hysteresis=key == "moe_hysteresis",
        )
        policy = policy.merge(run_metadata, on="trajectory_id", how="left", validate="one_to_one")
        policy["delta_utility"] = policy["stop_utility"] - policy["never_stop_utility"]

        domain_auc: list[float] = []
        domain_step_utility: list[float] = []
        domain_token_utility: list[float] = []
        domain_loss_rate: list[float] = []
        for domain in domains:
            run_indices = np.flatnonzero(run_metadata["domain"].to_numpy() == domain)
            mask = store.valid_mask_np(run_indices)
            domain_policy = policy[policy["domain"] == domain]
            auc = tournament.safe_auc(store.y[run_indices][mask], q[run_indices][mask])
            step_utility = float(domain_policy["stop_utility"].mean())
            token_utility = float(domain_policy["stop_utility_token"].mean())
            loss_rate = float((domain_policy["delta_utility"] < -1.0e-12).mean())
            domain_auc.append(auc)
            domain_step_utility.append(step_utility)
            domain_token_utility.append(token_utility)
            domain_loss_rate.append(loss_rate)
            domain_rows.append(
                {
                    "configuration": display[key],
                    "domain": domain.upper(),
                    "trajectories": int(len(run_indices)),
                    "oof_auc": auc,
                    "step_utility": step_utility,
                    "token_utility": token_utility,
                    "loss_rate": loss_rate,
                }
            )

        task_auc: list[float] = []
        for task_id in np.unique(store.task_ids):
            run_indices = np.flatnonzero(store.task_ids == task_id)
            mask = store.valid_mask_np(run_indices)
            auc = tournament.safe_auc(store.y[run_indices][mask], q[run_indices][mask])
            if np.isfinite(auc):
                task_auc.append(auc)

        fold_auc: list[float] = []
        fold_step_utility: list[float] = []
        for fold in range(1, 6):
            run_indices = np.flatnonzero(folds == fold)
            mask = store.valid_mask_np(run_indices)
            fold_policy = policy[policy["fold"] == fold]
            auc = tournament.safe_auc(store.y[run_indices][mask], q[run_indices][mask])
            step_utility = float(fold_policy["stop_utility"].mean())
            token_utility = float(fold_policy["stop_utility_token"].mean())
            loss_rate = float((fold_policy["delta_utility"] < -1.0e-12).mean())
            fold_auc.append(auc)
            fold_step_utility.append(step_utility)
            fold_rows.append(
                {
                    "configuration": display[key],
                    "fold": fold,
                    "trajectories": int(len(run_indices)),
                    "task_groups": int(run_metadata.loc[folds == fold, "task_id"].nunique()),
                    "oof_auc": auc,
                    "step_utility": step_utility,
                    "token_utility": token_utility,
                    "loss_rate": loss_rate,
                }
            )
        auc_low, auc_high = fold_t_interval(fold_auc)
        step_low, step_high = fold_t_interval(fold_step_utility)
        step_utility, token_utility, win_tie_loss = tournament.policy_summary(policy)
        summary_rows.append(
            {
                "configuration": display[key],
                "micro_oof_auc": tournament.safe_auc(store.y[valid_mask], q[valid_mask]),
                "task_macro_auc": float(np.mean(task_auc)),
                "valid_task_auc_groups": len(task_auc),
                "all_task_groups": int(manifest["task_ids"]),
                "domain_macro_auc": float(np.mean(domain_auc)),
                "worst_domain_auc": float(np.min(domain_auc)),
                "micro_step_utility": step_utility,
                "domain_macro_step_utility": float(np.mean(domain_step_utility)),
                "worst_domain_step_utility": float(np.min(domain_step_utility)),
                "micro_token_utility": token_utility,
                "domain_macro_token_utility": float(np.mean(domain_token_utility)),
                "worst_domain_loss_rate": float(np.max(domain_loss_rate)),
                "fold_auc_ci_low": auc_low,
                "fold_auc_ci_high": auc_high,
                "fold_step_utility_ci_low": step_low,
                "fold_step_utility_ci_high": step_high,
                "win_tie_loss": win_tie_loss,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("micro_oof_auc", ascending=False, kind="stable")
    summary.to_csv(OUTPUT / "tournament_balanced_summary.csv", index=False)
    pd.DataFrame(domain_rows).to_csv(OUTPUT / "tournament_per_domain.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT / "tournament_per_fold.csv", index=False)


def write_offline_replay_output() -> None:
    """Recompute the historical in-sample replay and task-cluster intervals.

    The original runner fits and evaluates on the same stored Qwen2.5-7B/GSM8K
    trajectories.  This function preserves that historical design, makes the
    in-sample status explicit, and adds uncertainty over the 500 task clusters.
    """
    active_stopping = load_module("progress_active_stopping", RESEARCH / "run_active_stopping.py")
    steps, _ = active_stopping.load_data()
    q_data, alpha_data, beta_data = active_stopping.prepare_training_data(steps)
    q_model, alpha_model, beta_model = active_stopping.train_models(q_data, alpha_data, beta_data)

    records: list[dict[str, Any]] = []
    for run_id, group in steps.groupby("run_id", sort=False):
        group = group.sort_values("step")
        stop_count = len(group)
        for position in range(len(group)):
            if position + 1 < 2:
                continue
            features = group.iloc[[position]][active_stopping.FEATURES]
            q_t = q_model.predict_proba(features)[0, 1]
            alpha_t = alpha_model.predict_proba(features)[0, 1]
            beta_t = beta_model.predict_proba(features)[0, 1]
            drift = (
                (1.0 - q_t) * alpha_t
                - q_t * beta_t
                - active_stopping.STEP_COST
            )
            if drift <= 0.0:
                stop_count = position + 1
                break
        records.append(
            {
                "run_id": run_id,
                "task_id": group["task_id"].iloc[0],
                "final_tokens": float(group["raw_generation_tokens"].sum()),
                "stopped_tokens": float(group["raw_generation_tokens"].iloc[:stop_count].sum()),
                "final_correct": int(group["correct"].iloc[-1]),
                "stopped_correct": int(group["correct"].iloc[stop_count - 1]),
            }
        )

    frame = pd.DataFrame(records)
    frame["accuracy_difference"] = frame["stopped_correct"] - frame["final_correct"]
    clustered = (
        frame.groupby("task_id", sort=False)
        .agg(
            n=("run_id", "size"),
            final_tokens=("final_tokens", "sum"),
            stopped_tokens=("stopped_tokens", "sum"),
            accuracy_difference=("accuracy_difference", "sum"),
        )
        .to_numpy(dtype=float)
    )
    accuracy_low, accuracy_high = cluster_ratio_bootstrap(
        clustered,
        numerator_column=3,
        denominator_column=0,
        label="offline-replay|accuracy-difference",
    )
    token_ratio_low, token_ratio_high = cluster_ratio_bootstrap(
        clustered,
        numerator_column=2,
        denominator_column=1,
        label="offline-replay|token-ratio",
    )
    final_tokens = float(frame["final_tokens"].sum())
    stopped_tokens = float(frame["stopped_tokens"].sum())
    row = {
        "design": "in-sample offline replay on stored trajectories",
        "model_domain": "Qwen2.5-7B / GSM8K",
        "trajectories": int(len(frame)),
        "task_clusters": int(frame["task_id"].nunique()),
        "final_tokens": int(final_tokens),
        "stopped_tokens": int(stopped_tokens),
        "token_savings": 1.0 - stopped_tokens / final_tokens,
        "token_savings_ci_low": 1.0 - token_ratio_high,
        "token_savings_ci_high": 1.0 - token_ratio_low,
        "final_accuracy": float(frame["final_correct"].mean()),
        "stopped_accuracy": float(frame["stopped_correct"].mean()),
        "accuracy_difference": float(frame["accuracy_difference"].mean()),
        "accuracy_difference_ci_low": accuracy_low,
        "accuracy_difference_ci_high": accuracy_high,
        "ci_method": "10,000-draw percentile bootstrap over 500 task clusters",
        "fit_evaluation_separation": "none; development diagnostic only",
    }
    pd.DataFrame([row]).to_csv(OUTPUT / "offline_replay_metrics.csv", index=False)


def write_manifest() -> None:
    files = {}
    for path in sorted(OUTPUT.glob("*.csv")):
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema": "progress-report-review-metrics-v1",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "step_cost": STEP_COST,
        "source_artifacts": {
            "canonical_matrix": "research/outputs/experiment_matrix/*/trace_steps.csv",
            "n1_n4_cache": "research/outputs/experiments_v2/algov2_cache/*.json",
            "n2_n3_cache": "research/outputs/experiments_v2/algov2_cache_n2_n3/*.json",
            "n5_n6_arms": "research/outputs/experiments_v2/p4b_* and p8_*",
            "tournament_manifest": "research/outputs/experiments_v2/ultimate_tournament_manifest.json",
            "tournament_predictions": "research/outputs/experiments_v2/ultimate_oof_predictions.npz",
            "offline_replay": "research/outputs/real_traces_bf16_ladder/qwen2p5_7b/trace_steps.csv",
        },
        "files": files,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    boundary_frame = load_boundary_frame()
    write_boundary_outputs(boundary_frame)
    write_algorithm_outputs()
    write_tournament_outputs()
    write_offline_replay_output()
    write_manifest()
    print(f"Wrote review metrics to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
