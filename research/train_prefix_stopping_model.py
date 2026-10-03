"""Fit, calibrate and evaluate a frozen, deployable prefix-only probability model.

This script intentionally does not import a live metrics or gold-ledger reader.
Only archived training-corpus labels enter fitting/calibration/evaluation.
The train/calibration/evaluation role is fixed by public task identity before fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from online_generation import complete_json_object
from online_stopping_controller import Observation, answer_key
from prefix_stopping_model import (
    ARTIFACT_SCHEMA, FEATURE_CONTRACT, FEATURE_NAMES, SUPPORTED_DOMAINS,
    PrefixStoppingModel, prefix_features,
)
from real_trace_experiments import TaskSpec, parse_generation, verify_answer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "research/outputs/semester2/prefix_model_v1"
DEFAULT_SOURCES = (
    ROOT / "research/outputs/experiments_v2/global_qwen2p5_0p5b_gsm8k/trace_steps.csv",
    ROOT / "research/outputs/experiments_v2/global_qwen2p5_0p5b_math/trace_steps.csv",
)
SPLIT_SALT = "prefix-model-v1-task-split-fixed-20261002"
STEP_COST = 0.05
PROTOCOL = {
    "split": "task hash: 60% train, 20% calibration, 20% evaluation",
    "split_salt": SPLIT_SALT,
    "base": "unweighted LogisticRegression C=1, lbfgs, max_iter=2000",
    "scaler": "StandardScaler fitted on train tasks only",
    "calibrator": "unweighted one-dimensional Platt logistic C=1000, max_iter=2000",
    "calibration_role": "fit calibrators on independent calibration tasks only",
    "policy": "first p_next-q_current-0.05 <= 0 at steps 2..4; otherwise step 5",
    "selection": "latest nonempty observed answer",
    "threshold_selection": "fixed cost; no evaluation/live/trap threshold tuning",
    "targets": "offline archive-gold regrading of reconstructed selected current/next candidate",
    "label_reconstruction": "same verify_answer as archive; reconstructed live-parser candidate, never live gold",
    "bootstrap": "2000 task-cluster percentile replicates, seed 20261002",
}


def task_split(task_key: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}::{task_key}".encode()).digest()
    uniform = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if uniform < 0.6 else "calibration" if uniform < 0.8 else "evaluation"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def source_record(path: Path) -> dict:
    raw = path.read_bytes()
    return {
        "path": path.resolve().relative_to(ROOT).as_posix(),
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_lf_sha256": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
    }


def validate_freeze(records: list[dict]) -> dict:
    manifest_path = ROOT / "data_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    indexed: dict[str, str] = {}
    def walk(value):
        if isinstance(value, dict):
            if "path" in value and "canonical_lf_sha256" in value:
                indexed[str(value["path"]).replace("\\", "/")] = value["canonical_lf_sha256"]
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(manifest)
    for record in records:
        frozen = indexed.get(record["path"])
        if frozen is None or frozen != record["canonical_lf_sha256"]:
            raise ValueError(f"source missing from freeze or hash mismatch: {record['path']}")
    return source_record(manifest_path)


def excluded_public_task_ids() -> tuple[set[str], list[dict]]:
    """Only public, label-free task files are read for exclusion."""
    paths = (
        ROOT / "research/outputs/semester2/online_stopping_20261002/live_public_tasks.jsonl",
        ROOT / "research/adversarial_tasks_v1.jsonl",
    )
    ids: set[str] = set()
    records = []
    for path in paths:
        if not path.exists():
            continue
        records.append(source_record(path))
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if any(name in record for name in ("expected_answer", "correct", "gold")):
                raise ValueError("exclusion input must be a label-free public task record")
            ids.add(str(record["task_id"]))
    return ids, records


def load_examples(paths: list[Path], excluded_ids: set[str]):
    rows, trajectories, label_audit, exclusions = [], [], [], []
    census = Counter()
    usecols = (
        "run_id", "task_id", "domain", "step", "answer", "correct", "expected_answer",
        "raw_generation_tokens", "raw_text",
    )
    grade_cache = {}
    for path in paths:
        frame = pd.read_csv(path, usecols=usecols, keep_default_na=False)
        for run_id, trace in frame.groupby("run_id", sort=True):
            census["source_trajectories"] += 1
            trace = trace.sort_values("step")
            task_id, domain = str(trace.iloc[0].task_id), str(trace.iloc[0].domain)
            if task_id in excluded_ids:
                census["excluded_public_live_or_trap"] += 1
                exclusions.append({"source": path.parent.name, "run_id": run_id, "task_id": task_id,
                                   "reason": "public live/trap task ID exclusion"})
                continue
            if (domain not in SUPPORTED_DOMAINS or trace.task_id.nunique() != 1
                    or trace.domain.nunique() != 1 or trace.step.tolist() != [1, 2, 3, 4, 5]):
                census["excluded_invalid_horizon_or_task"] += 1
                exclusions.append({"source": path.parent.name, "run_id": run_id, "task_id": task_id,
                                   "reason": "invalid domain/task identity/complete five-step horizon"})
                continue
            if trace.expected_answer.astype(str).nunique() != 1:
                raise ValueError("inconsistent archived gold within trajectory")
            task = TaskSpec(task_id, domain, "archived", "", "number" if domain == "gsm8k" else "math",
                            str(trace.iloc[0].expected_answer), "offline training label reconstruction")
            observations, correctness = [], []
            identifier = f"{path.parent.name}::{run_id}"
            for record in trace.itertuples(index=False):
                strict = complete_json_object(str(record.raw_text))
                parsed = strict or parse_generation(
                    str(record.raw_text), "number" if domain == "gsm8k" else "math", "minimal_json",
                )
                candidate = str(parsed.get("answer", "")).strip()
                candidate_disagrees = answer_key(candidate) != answer_key(str(record.answer))
                if record.correct not in (0, 1):
                    raise ValueError("archived correctness must be binary")
                grade_key = (domain, task.expected_answer, candidate)
                if grade_key not in grade_cache:
                    grade_cache[grade_key] = int(verify_answer(task, candidate)) if candidate else 0
                regraded = grade_cache[grade_key]
                label_audit.append({
                    "trajectory_id": identifier, "task_key": f"{domain}::{task_id}",
                    "domain": domain, "step": int(record.step), "split": task_split(f"{domain}::{task_id}"),
                    "saved_candidate": str(record.answer), "reconstructed_candidate": candidate,
                    "saved_correct": int(record.correct), "reconstructed_correct": regraded,
                    "candidate_disagrees": int(candidate_disagrees),
                    "label_disagrees": int(regraded != int(record.correct)),
                    "strict_json": int(strict is not None),
                })
                census["reconstructed_candidate_disagreement_rows"] += int(candidate_disagrees)
                census["reconstructed_label_disagreement_rows"] += int(regraded != int(record.correct))
                observations.append(Observation(
                    step=int(record.step), answer=candidate,
                    confidence=float(strict["confidence"]) if strict else None,
                    parse_success=strict is not None, generated_tokens=int(record.raw_generation_tokens),
                    thought=str(parsed.get("thought", "")), raw_text=str(record.raw_text),
                    model_stop_flag=bool(parsed.get("stop", False)),
                ))
                correctness.append(regraded)
            task_key = f"{domain}::{task_id}"
            split = task_split(task_key)
            selected_labels, selected_steps = [], []
            for length in range(1, 6):
                available = [i for i, item in enumerate(observations[:length]) if item.answer.strip()]
                selected = available[-1] if available else None
                selected_labels.append(correctness[selected] if selected is not None else 0)
                selected_steps.append(selected + 1 if selected is not None else length)
            for index in range(5):
                features = prefix_features(tuple(observations[:index + 1]), domain)
                rows.append({
                    "trajectory_id": identifier, "task_key": task_key, "domain": domain,
                    "split": split, "step": index + 1, "q_target": selected_labels[index],
                    "next_target": selected_labels[index + 1] if index < 4 else np.nan,
                    **dict(zip(FEATURE_NAMES, features)),
                })
            trajectories.append({
                "trajectory_id": identifier, "task_key": task_key, "domain": domain,
                "split": split, "observations": tuple(observations),
                "selected_labels": selected_labels, "selected_steps": selected_steps,
            })
            census["included_trajectories"] += 1
            census["strict_json_rows"] += sum(item.parse_success for item in observations)
            census["included_step_rows"] += 5
    if not trajectories:
        raise ValueError("no complete runtime-compatible archived trajectories")
    return (
        pd.DataFrame(rows), trajectories, dict(census), pd.DataFrame(label_audit),
        pd.DataFrame(exclusions, columns=["source", "run_id", "task_id", "reason"]),
    )


def _logit(probability: float) -> float:
    probability = min(1 - 1e-6, max(1e-6, probability))
    return math.log(probability / (1 - probability))


def fit_component(frame: pd.DataFrame, target: str) -> dict:
    eligible = frame.loc[frame[target].notna()]
    train = eligible.loc[eligible.split == "train"]
    calibration = eligible.loc[eligible.split == "calibration"]
    if train.empty or calibration.empty:
        raise ValueError("training and calibration rows are required")
    x_train = train.loc[:, FEATURE_NAMES].to_numpy(float)
    x_calibration = calibration.loc[:, FEATURE_NAMES].to_numpy(float)
    y_train, y_calibration = train[target].to_numpy(int), calibration[target].to_numpy(int)
    if len(np.unique(y_train)) == 1:
        scaler = StandardScaler().fit(x_train)
        coefficient = np.zeros(len(FEATURE_NAMES))
        intercept = _logit(float(y_train.mean()))
        calibration_logits = np.repeat(intercept, len(calibration))
    else:
        pipeline = Pipeline((
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, class_weight=None)),
        )).fit(x_train, y_train)
        scaler, logistic = pipeline.named_steps["scale"], pipeline.named_steps["logistic"]
        coefficient, intercept = logistic.coef_[0], float(logistic.intercept_[0])
        calibration_logits = pipeline.decision_function(x_calibration)
    if len(np.unique(y_calibration)) == 1:
        slope, offset = 0.0, _logit(float(y_calibration.mean()))
        calibrator_kind = "constant prevalence fallback: calibration has one class"
    else:
        platt = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=2000, class_weight=None)
        platt.fit(calibration_logits.reshape(-1, 1), y_calibration)
        slope, offset = float(platt.coef_[0, 0]), float(platt.intercept_[0])
        calibrator_kind = "independent task-group Platt logistic"
    return {
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coefficient": coefficient.tolist(), "intercept": intercept,
        "calibration": {"kind": calibrator_kind, "slope": slope, "intercept": offset},
        "train_rows": len(train), "calibration_rows": len(calibration),
        "train_tasks": int(train.task_key.nunique()),
        "calibration_tasks": int(calibration.task_key.nunique()),
        "train_prevalence": float(y_train.mean()),
        "calibration_prevalence": float(y_calibration.mean()),
    }


def probability_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    bin_ids = np.minimum((probabilities * 10).astype(int), 9)
    bins = []
    for index in range(10):
        mask = bin_ids == index
        bins.append({
            "bin": index, "count": int(mask.sum()),
            "mean_probability": float(probabilities[mask].mean()) if mask.any() else None,
            "prevalence": float(y[mask].mean()) if mask.any() else None,
        })
    ece = sum(
        item["count"] / len(y) * abs(item["mean_probability"] - item["prevalence"])
        for item in bins if item["count"]
    )
    return {
        "rows": len(y), "prevalence": float(y.mean()), "mean_probability": float(probabilities.mean()),
        "brier": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "auc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
        "ece_10_equal_width": float(ece), "calibration_bins": bins,
    }


def evaluate_probabilities(model: PrefixStoppingModel, frame: pd.DataFrame):
    evaluation = frame.loc[frame.split == "evaluation"].copy()
    metrics = {}
    for component, target in (("q_current", "q_target"), ("p_next", "next_target")):
        eligible = evaluation.loc[evaluation[target].notna()]
        probabilities, raw = [], []
        for record in eligible.loc[:, FEATURE_NAMES].to_numpy(float):
            probabilities.append(model._score(component, tuple(record)))
            raw.append(model._score(component, tuple(record), calibrated=False))
        evaluation.loc[eligible.index, component] = probabilities
        evaluation.loc[eligible.index, f"{component}_raw"] = raw
        y, p, p_raw = eligible[target].to_numpy(int), np.asarray(probabilities), np.asarray(raw)
        metrics[component] = {
            "calibrated": probability_metrics(y, p), "raw": probability_metrics(y, p_raw),
            "by_domain": {
                domain: probability_metrics(y[eligible.domain.to_numpy() == domain], p[eligible.domain.to_numpy() == domain])
                for domain in SUPPORTED_DOMAINS if (eligible.domain == domain).any()
            },
        }
    return metrics, evaluation


def replay_policies(model: PrefixStoppingModel, trajectories: list[dict]) -> pd.DataFrame:
    rows = []
    for trajectory in trajectories:
        if trajectory["split"] != "evaluation":
            continue
        observations = trajectory["observations"]
        learned = 5
        for length in range(2, 5):
            if model.estimate(observations[:length], trajectory["domain"]).gain <= 0:
                learned = length
                break
        baseline_correct = trajectory["selected_labels"][-1]
        baseline_tokens = sum(item.generated_tokens for item in observations)
        for policy, stop in (("learned_drift", learned), ("fixed_2", 2), ("fixed_3", 3), ("fixed_4", 4), ("never", 5)):
            correct = trajectory["selected_labels"][stop - 1]
            rows.append({
                "trajectory_id": trajectory["trajectory_id"], "task_key": trajectory["task_key"],
                "domain": trajectory["domain"], "policy": policy, "stop_step": stop,
                "selected_step": trajectory["selected_steps"][stop - 1],
                "active_correct": correct, "baseline_correct": baseline_correct,
                "accuracy_difference": correct - baseline_correct,
                "active_utility": correct - STEP_COST * (stop - 1),
                "baseline_utility": baseline_correct - STEP_COST * 4,
                "utility_difference": correct - baseline_correct + STEP_COST * (5 - stop),
                "active_completion_tokens": sum(item.generated_tokens for item in observations[:stop]),
                "baseline_completion_tokens": baseline_tokens,
            })
    return pd.DataFrame(rows)


def summarize_replay(frame: pd.DataFrame) -> dict:
    summary = {}
    for policy, rows in frame.groupby("policy", sort=True):
        aggregate = rows.groupby("task_key", sort=True).agg(
            n=("stop_step", "size"), accuracy=("accuracy_difference", "sum"),
            utility=("utility_difference", "sum"), stop=("stop_step", "sum"),
            active=("active_completion_tokens", "sum"), baseline=("baseline_completion_tokens", "sum"),
        )
        values = aggregate.to_numpy(float)
        rng = np.random.default_rng(20261002)
        sampled = rng.integers(0, len(values), size=(2000, len(values)))
        sums = values[sampled].sum(axis=1)
        replicates = {
            "accuracy_difference": sums[:, 1] / sums[:, 0],
            "utility_difference": sums[:, 2] / sums[:, 0],
            "completion_token_saving_fraction": 1 - sums[:, 4] / sums[:, 5],
            "mean_stop_step": sums[:, 3] / sums[:, 0],
        }
        summary[policy] = {
            "trajectories": len(rows), "tasks": len(aggregate),
            "accuracy": float(rows.active_correct.mean()), "baseline_accuracy": float(rows.baseline_correct.mean()),
            "accuracy_difference": float(rows.accuracy_difference.mean()),
            "utility": float(rows.active_utility.mean()), "baseline_utility": float(rows.baseline_utility.mean()),
            "utility_difference": float(rows.utility_difference.mean()),
            "completion_token_saving_fraction": float(1 - rows.active_completion_tokens.sum() / rows.baseline_completion_tokens.sum()),
            "mean_stop_step": float(rows.stop_step.mean()),
            "active_completion_tokens": int(rows.active_completion_tokens.sum()),
            "baseline_completion_tokens": int(rows.baseline_completion_tokens.sum()),
            "stop_step_counts": {str(k): int(v) for k, v in rows.stop_step.value_counts().sort_index().items()},
            "task_cluster_95_percentile_intervals": {
                name: np.quantile(values_, [0.025, 0.975]).tolist()
                for name, values_ in replicates.items()
            },
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = [path.resolve() for path in (args.source or DEFAULT_SOURCES)]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = [source_record(path) for path in paths]
    freeze = validate_freeze(sources)
    (output / "data_freeze_at_training.json").write_bytes((ROOT / "data_manifest_v1.json").read_bytes())
    excluded, public_inputs = excluded_public_task_ids()
    # Choose the task role from public identity before parser availability or labels.
    tasks = pd.concat([
        pd.read_csv(path, usecols=["task_id", "domain"], keep_default_na=False) for path in paths
    ]).drop_duplicates()
    tasks = tasks.loc[~tasks.task_id.isin(excluded)].copy()
    tasks["task_key"] = tasks.domain.astype(str) + "::" + tasks.task_id.astype(str)
    tasks["split"] = tasks.task_key.map(task_split)
    tasks = tasks[["task_key", "domain", "split"]].sort_values("task_key")
    if tasks.task_key.duplicated().any() or set(tasks.split) != {"train", "calibration", "evaluation"}:
        raise ValueError("all three disjoint task roles are required")
    tasks.to_csv(output / "task_split.csv", index=False, lineterminator="\n")
    frame, trajectories, census, label_audit, exclusions = load_examples(paths, excluded)
    label_audit.to_csv(output / "label_reconstruction_audit.csv", index=False, lineterminator="\n")
    exclusions.to_csv(output / "excluded_trajectories.csv", index=False, lineterminator="\n")
    protocol = {
        **PROTOCOL, "feature_contract": FEATURE_CONTRACT, "feature_names": list(FEATURE_NAMES),
        "sources": sources, "data_freeze": freeze, "public_exclusions": public_inputs,
        "excluded_public_task_id_count": len(excluded), "eligibility_census": census,
        "task_split_sha256": hashlib.sha256((output / "task_split.csv").read_bytes()).hexdigest(),
        "task_counts": {split: int((tasks.split == split).sum()) for split in ("train", "calibration", "evaluation")},
        "task_counts_by_domain": tasks.groupby(["domain", "split"]).size().to_dict(),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    # JSON keys cannot be tuples; preserve each domain/role count explicitly.
    protocol["task_counts_by_domain"] = [
        {"domain": domain, "split": split, "tasks": int(count)}
        for (domain, split), count in tasks.groupby(["domain", "split"]).size().items()
    ]
    # Commit the public-role and hyperparameter lock before either fit.
    write_json(output / "training_protocol.json", protocol)
    limits = [
        "Only archived Qwen2.5-0.5B GSM8K training and MATH traces enter fitting.",
        "Archived four-line output and live JSON prompts differ; this is a material transport limitation.",
        "Strict JSON confidence is reconstructed from raw output; fallback defaults are missing, not trusted confidence.",
        "Archives omit emitted EOS tokens while the live generator charges them.",
        "Reconstructed candidates are regraded against archived gold offline; source labels/data are unchanged.",
        "A row-level audit preserves old/reconstructed candidates and labels, including parser disagreements.",
        "Task-disjoint archive calibration is marginal validation, not a guarantee of conditional live calibration.",
        "The direct one-step drift policy is myopic; it does not approximate a proven Bellman continuation value.",
        "No live benchmark or adversarial trap outcomes select coefficients, calibrators, features, cost or threshold.",
        "Bootstrap intervals describe repeated task-cluster sampling from this held-out development panel.",
    ]
    artifact = {
        "schema": ARTIFACT_SCHEMA, "feature_contract": FEATURE_CONTRACT,
        "feature_names": list(FEATURE_NAMES), "domains": list(SUPPORTED_DOMAINS),
        "min_steps": 2, "max_steps": 5, "step_cost": STEP_COST,
        "selection": "latest nonempty observed answer",
        "protocol_sha256": hashlib.sha256((output / "training_protocol.json").read_bytes()).hexdigest(),
        "task_split_sha256": protocol["task_split_sha256"],
        "models": {"q_current": fit_component(frame, "q_target"), "p_next": fit_component(frame, "next_target")},
        "limitations": limits,
    }
    model_path = output / "prefix_model.json"
    write_json(model_path, artifact)
    model = PrefixStoppingModel.load(model_path)
    probability, predictions = evaluate_probabilities(model, frame)
    replay = replay_policies(model, trajectories)
    policies = summarize_replay(replay)
    predictions.to_csv(output / "heldout_prefix_predictions.csv", index=False, lineterminator="\n")
    replay.to_csv(output / "heldout_policy_replay.csv", index=False, lineterminator="\n")
    evaluation = {
        "artifact_sha256": model.artifact_sha256, "kind": "fixed task-disjoint archived development holdout",
        "probabilities": probability, "policies": policies, "task_counts": protocol["task_counts"],
        "eligibility_census": census, "limitations": limits, "live_compute_savings_measured": False,
        "uncertainty": PROTOCOL["bootstrap"],
    }
    write_json(output / "evaluation.json", evaluation)
    report = [
        "# Frozen prefix probability model",
        "",
        "Artifact: prefix_model.json; file-byte SHA-256: " + model.artifact_sha256 + ".",
        "This is a CPU-trained, standard-library deployable pair of unweighted logistic models.",
        "Each probability calibrator uses separate calibration tasks; evaluation tasks enter neither fit.",
        "",
        "| Target | Held-out rows | AUC | Calibrated Brier | Raw Brier | ECE (10 bins) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, results in probability.items():
        calibrated, raw = results["calibrated"], results["raw"]
        auc = "undefined (one class)" if calibrated["auc"] is None else f"{calibrated['auc']:.6f}"
        report.append(f"| {name} | {calibrated['rows']} | {auc} | {calibrated['brier']:.6f} | {raw['brier']:.6f} | {calibrated['ece_10_equal_width']:.6f} |")
    report += [
        "", "AUC measures ranking; held-out Brier/ECE describe marginal prediction quality, not conditional coverage.",
        "", "| Replay policy | Accuracy | Paired accuracy change | Step utility change | Completion-token saving | Mean stop |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, results in policies.items():
        report.append(f"| {name} | {results['accuracy']:.4f} | {results['accuracy_difference']:+.4f} | {results['utility_difference']:+.4f} | {100 * results['completion_token_saving_fraction']:.2f}% | {results['mean_stop_step']:.3f} |")
    report += ["", "Replay tokens are archived completion costs, not measured prospective model compute. Full task-cluster intervals and domain diagnostics are in evaluation.json.", "", "Limits:", ""]
    report += ["- " + limit for limit in limits]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "artifact": str(model_path), "artifact_sha256": model.artifact_sha256,
        "task_counts": protocol["task_counts"], "eligibility": census,
        "q_current": probability["q_current"]["calibrated"],
        "p_next": probability["p_next"]["calibrated"],
        "learned_replay": policies["learned_drift"],
    }, indent=2))


if __name__ == "__main__":
    main()
