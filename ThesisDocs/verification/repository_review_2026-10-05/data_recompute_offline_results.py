"""Recompute saved OOF scores and source/fold bindings; never fit a model."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

BASE = Path("research/outputs/experiments_v2")
DEST = Path("tmp/research_history_audit")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    return {"path": str(path).replace("\\", "/"), "sha256": sha(path), "bytes": Path(path).stat().st_size}


tracked = set(subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0"))
source_cache = {}
corpus = read(BASE/"committee_oof_peer_dynamics_v3/anonymous_minimal/peer_dynamics_manifest.json")["files"]
source_bindings = []
for source in corpus:
    p = Path(source["path"])
    raw = p.read_bytes()
    actual_raw = hashlib.sha256(raw).hexdigest()
    canonical = hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
    source_cache[p.as_posix()] = (actual_raw, canonical)
    source_bindings.append({"path": p.as_posix(), "actual_bytes": len(raw), "declared_bytes": source["bytes"], "declared_raw_sha256": source["raw_sha256"], "actual_raw_sha256": actual_raw, "raw_bytes_match": actual_raw == source["raw_sha256"], "declared_canonical_lf_sha256": source["canonical_lf_sha256"], "actual_canonical_lf_sha256": canonical, "canonical_lf_matches": canonical == source["canonical_lf_sha256"]})

specs = [
    ("strict_tabular", "strict_tabular_stopping_oof_v1", "safe_scalar_lgbm_predictions.csv", "safe_scalar_lgbm_metrics.json", "prepared_manifest.json", "oof_q"),
    ("strict_text", "strict_text_stopping_oof_v1", "tail512_word12_predictions.csv", "tail512_word12_metrics.json", "prepared_manifest.json", "oof_probability"),
    ("anonymous_peer_treatment", "committee_oof_peer_dynamics_v3/anonymous_minimal", "peer_dynamics_predictions.csv", "peer_dynamics_metrics.json", "peer_dynamics_manifest.json", "oof_probability"),
    ("anonymous_peer_baseline", "committee_oof_peer_dynamics_v3/anonymous_minimal_baseline", "peer_dynamics_predictions.csv", "peer_dynamics_metrics.json", "peer_dynamics_manifest.json", "oof_probability"),
    ("fixed13_peer_treatment", "committee_oof_peer_dynamics_fixed13_v2/anonymous_minimal", "peer_dynamics_predictions.csv", "peer_dynamics_metrics.json", "peer_dynamics_manifest.json", "oof_probability"),
    ("fixed13_peer_baseline", "committee_oof_peer_dynamics_fixed13_v2/anonymous_minimal_baseline", "peer_dynamics_predictions.csv", "peer_dynamics_metrics.json", "peer_dynamics_manifest.json", "oof_probability"),
]
frames = {}
arms = []
for name, directory, prediction_name, metrics_name, manifest_name, probability in specs:
    folder = BASE/directory
    prediction_path, metrics_path, manifest_path = [folder/p for p in (prediction_name, metrics_name, manifest_name)]
    frame = pd.read_csv(prediction_path, float_precision="round_trip")
    metrics = read(metrics_path)
    manifest = read(manifest_path)
    y = frame["correct"].to_numpy(dtype=np.int8)
    score = frame[probability].to_numpy(dtype=np.float64)
    assert set(np.unique(y)).issubset({0, 1}) and np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
    auc = float(roc_auc_score(y, score))
    fold_column = "outer_fold" if "outer_fold" in frame else "fold" if "fold" in frame else None
    checkpoint_checks = []
    if name == "strict_tabular":
        run_table = frame[["trajectory_id", "task_id"]].drop_duplicates("trajectory_id", keep="first").reset_index(drop=True)
        assert len(frame) == len(run_table)*5
        assert frame.groupby("trajectory_id")["step"].nunique().eq(5).all()
        expected = list(GroupKFold(n_splits=5).split(np.arange(len(run_table)), groups=run_table.task_id.to_numpy()))
        run_folds = np.zeros(len(run_table), dtype=np.int8)
        q_matrix = score.reshape(len(run_table), 5).astype(np.float32)
        for fold in range(1, 6):
            p = folder/"fold_checkpoints"/f"safe_scalar_lgbm_fold_{fold:02}.npz"
            with np.load(p, allow_pickle=False) as checkpoint:
                indices = checkpoint["test_indices"]
                binding = json.loads(str(checkpoint["binding_json"]))
                split_match = np.array_equal(indices, expected[fold-1][1])
                score_match = np.array_equal(q_matrix[indices], checkpoint["q"])
                digest_match = hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest() == binding["test_indices_sha256"]
                prepared_match = binding["prepared_manifest_sha256"] == sha(manifest_path)
                assert split_match and score_match and digest_match and prepared_match
                assert not (set(run_table.task_id.iloc[expected[fold-1][0]]) & set(run_table.task_id.iloc[indices]))
                run_folds[indices] = fold
                checkpoint_checks.append({**identity(p), "fold": fold, "test_run_indices_match_expected_GroupKFold": split_match, "CSV_probabilities_match_checkpoint_exact_float32": score_match, "test_indices_digest_matches": digest_match, "prepared_manifest_binding_matches": prepared_match, "binding": binding})
        frame["reconstructed_outer_fold"] = np.repeat(run_folds, 5)
        fold_column = "reconstructed_outer_fold"
    else:
        expected = list(GroupKFold(n_splits=5).split(np.arange(len(frame)), groups=frame.task_id.to_numpy()))
        for fold, (train, test) in enumerate(expected, 1):
            actual = np.flatnonzero(frame[fold_column].to_numpy() == fold)
            assert np.array_equal(actual, test)
            assert not (set(frame.task_id.iloc[train]) & set(frame.task_id.iloc[test]))
            if name == "strict_text":
                p = folder/"fold_checkpoints"/f"tail512_word12_fold_{fold:02}.npz"
                with np.load(p, allow_pickle=False) as checkpoint:
                    binding = json.loads(str(checkpoint["binding_json"]))
                    split_match = np.array_equal(checkpoint["test_indices"], test)
                    score_match = np.array_equal(score[test], checkpoint["probabilities"])
                    digest_match = hashlib.sha256(np.asarray(test, dtype=np.int64).tobytes()).hexdigest() == binding["test_indices_sha256"]
                    prepared_match = binding["prepared_manifest_sha256"] == sha(manifest_path)
                    assert split_match and score_match and digest_match and prepared_match
                    checkpoint_checks.append({**identity(p), "fold": fold, "test_row_indices_match_expected_GroupKFold": split_match, "CSV_probabilities_match_checkpoint_exact_float64": score_match, "test_indices_digest_matches": digest_match, "prepared_manifest_binding_matches": prepared_match, "binding": binding})
    per_fold = []
    for fold, part in frame.groupby(fold_column):
        per_fold.append({"fold": int(fold), "rows": len(part), "tasks": int(part.task_id.nunique()), "auc": float(roc_auc_score(part.correct, part[probability]))})
    input_files = manifest.get("files", manifest.get("input_manifest", {}).get("files", []))
    matching_corpus = []
    for source in input_files:
        p = Path(source["path"])
        if not p.as_posix().startswith("research/"):
            p = BASE/p
        actual_raw, canonical = source_cache[p.as_posix()]
        matching_corpus.append({"path": p.as_posix(), "raw_match": actual_raw == source.get("raw_sha256", source.get("sha256")), "canonical_lf_match": canonical == source["canonical_lf_sha256"]})
    current_source_hashes = []
    for basename, declared in manifest.get("source_hashes", {}).items():
        p = Path("research")/basename
        current_source_hashes.append({"path": p.as_posix(), "declared_sha256": declared, "current_sha256": sha(p), "current_bytes_match": sha(p) == declared})
    status_path = folder/("strict_tabular_status.json" if name == "strict_tabular" else "strict_text_status.json" if name == "strict_text" else "peer_dynamics_summary.json")
    status = read(status_path) if status_path.exists() else None
    saved_auc = metrics["oof_auc"]
    fold_auc_match = all(abs(a["auc"] - b) <= 1e-12 for a, b in zip(per_fold, metrics.get("fold_auc", []))) if "fold_auc" in metrics else None
    result = {"name": name, "directory": folder.as_posix(), "prediction_file": identity(prediction_path), "metrics_file": identity(metrics_path), "manifest_file": identity(manifest_path), "tracked_prediction_at_HEAD": prediction_path.as_posix() in tracked, "artifact_status": status, "manifest_status": manifest.get("status"), "rows": len(frame), "task_groups": int(frame.task_id.nunique()), "trajectories": int(frame.trajectory_id.nunique()), "labels": {str(int(k)): int(v) for k, v in frame.correct.value_counts().sort_index().items()}, "label_source": "Stored archived per-candidate correctness from the saved canonical traces; no regrading is performed by this audit.", "probability_column": probability, "auc_recomputed_from_saved_OOF_probabilities": auc, "auc_saved": saved_auc, "auc_matches_saved": abs(auc-saved_auc) <= 1e-12, "folds": per_fold, "fold_auc_matches_saved": fold_auc_match, "folds_match_independently_reconstructed_GroupKFold": True, "every_task_in_one_outer_test_fold": frame.groupby("task_id")[fold_column].nunique().eq(1).all().item(), "checkpoints": checkpoint_checks, "seed": manifest.get("seed"), "strict_contract": manifest.get("strict_contract"), "feature_count": metrics.get("feature_count", len(manifest.get("numeric_features", manifest.get("feature_columns", [])))), "learner_categorical_features": manifest.get("categorical_features", []), "fixed_panel_filter": manifest.get("fixed_panel_filter"), "source_corpus_files": len(input_files), "source_corpus_canonical_lf_all_match": all(f["canonical_lf_match"] for f in matching_corpus), "source_corpus_raw_matches": sum(f["raw_match"] for f in matching_corpus), "source_corpus_raw_nonmatches": [f["path"] for f in matching_corpus if not f["raw_match"]], "source_code_hashes_current_check": current_source_hashes, "saved_task_bootstrap_auc_CI": metrics.get("task_cluster_bootstrap_auc_95_ci"), "calibration_scope": "AUC calculated directly from already-persisted OOF probabilities. No new calibration, training, nested model selection, or fresh collection is performed. Strict code has inner task-disjoint calibration; this audit validates outputs and split/checkpoint bindings, not every historical fit action."}
    assert result["auc_matches_saved"] and result["source_corpus_canonical_lf_all_match"]
    arms.append(result)
    frames[name] = frame
    print(json.dumps({"name": name, "auc": auc, "rows": len(frame), "tasks": result["task_groups"], "corpus_raw_matches": result["source_corpus_raw_matches"], "corpus_canonical_matches": result["source_corpus_canonical_lf_all_match"]}), flush=True)

pairs = []
for treatment, control in [("anonymous_peer_treatment", "anonymous_peer_baseline"), ("fixed13_peer_treatment", "fixed13_peer_baseline")]:
    left, right = frames[treatment], frames[control]
    keys = ["task_id", "trajectory_id", "model_alias", "domain", "step", "correct", "fold"]
    aligned = left[keys].equals(right[keys])
    assert aligned
    a = next(arm for arm in arms if arm["name"] == treatment)
    b = next(arm for arm in arms if arm["name"] == control)
    pairs.append({"treatment": treatment, "baseline": control, "exact_same_order_keys_labels_and_folds": aligned, "auc_delta_recomputed": a["auc_recomputed_from_saved_OOF_probabilities"] - b["auc_recomputed_from_saved_OOF_probabilities"], "same_seed": a["seed"] == b["seed"], "same_source_corpus_hashes": a["source_corpus_canonical_lf_all_match"] and b["source_corpus_canonical_lf_all_match"]})

result = {"schema": "independently-verified-saved-offline-results-v1", "created_utc": datetime.now(timezone.utc).isoformat(), "head": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(), "audit_script": identity(__file__), "runtime": {"numpy": np.__version__, "pandas": pd.__version__}, "methods": ["ROC-AUC recomputed from persisted prediction CSVs without applying any new model or calibration", "All expected task GroupKFold test partitions independently reconstructed from saved row/run task groups", "Ten strict baseline fold NPZs loaded with allow_pickle=False and compared to CSV probabilities, partition indices, prepared-manifest hashes", "Shared 52 trace-step source files independently hashed in exact-byte and canonical-LF form", "Matched peer arms compared on exact task/trajectory/model/step/label/fold row order"], "source_corpus": {"files": len(source_bindings), "raw_exact_matches": sum(s["raw_bytes_match"] for s in source_bindings), "canonical_lf_matches": sum(s["canonical_lf_matches"] for s in source_bindings), "bindings": source_bindings}, "arms": arms, "matched_pairs": pairs, "qualification": ["These are historical analysis/OOF results over previously generated archived data, not new model generation or a prospective deployed-policy test.", "Raw peer-candidate correctness and selected-answer correctness are different targets and denominators; their AUCs cannot be pooled or treated as one leaderboard.", "Recorded OOF outputs can include calibration already applied by the original pipeline; 'saved probability AUC' does not recover missing pre-calibration logits.", "No fully nested hyperparameter-selection performance or fresh seed-replicated generation claim is made.", "Archived correctness labels are reproduced from saved artifacts, not independently regraded by this task.", "A task-consistent test fold and matched corpus hashes do not retroactively supply missing timestamped peer barriers or original unrecorded software/model revisions."], "all_requested_point_estimates_and_bindings_pass": all(a["auc_matches_saved"] and a["source_corpus_canonical_lf_all_match"] and a["every_task_in_one_outer_test_fold"] for a in arms)}
target = DEST/"data_verified_offline_results.json"
target.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
print(json.dumps({"receipt": identity(target), "all_pass": result["all_requested_point_estimates_and_bindings_pass"], "source_raw_matches": result["source_corpus"]["raw_exact_matches"], "source_canonical_matches": result["source_corpus"]["canonical_lf_matches"], "matched_pairs": pairs}), flush=True)
