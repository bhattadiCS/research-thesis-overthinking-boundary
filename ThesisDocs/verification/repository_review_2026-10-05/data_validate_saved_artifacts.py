"""Read saved CSV/JSON artifacts only; no experiments, grading, or test suite executes."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path("research/outputs/experiments_v2")
DEST = Path("tmp/research_history_audit")
csv.field_size_limit(10_000_000)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4_000_000), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(name, value):
    path = DEST/name
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return {"path": path.as_posix(), "sha256": sha(path)}


tracked = set(subprocess.check_output(["git", "ls-files", "-z"]).decode("utf-8").split("\0"))
families = load(DEST/"data_target_family_evidence.json")["families"]
fold_checks = []
binding_checks = []
for family in families:
    for p in sorted(Path(family["directory"]).rglob("*predictions.csv")):
        with p.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            fields = reader.fieldnames
            fold = next((key for key in ("outer_fold", "fold", "fold_id") if key in fields), None)
            groups = defaultdict(set)
            folds = Counter()
            domains = Counter()
            labels = Counter()
            label_field = "selected_correct" if "selected_correct" in fields else "correct" if "correct" in fields else None
            rows = 0
            for row in reader:
                rows += 1
                if "task_id" in fields:
                    groups[row["task_id"]].add(row[fold] if fold else "no-fold-column")
                if fold:
                    folds[row[fold]] += 1
                if "domain" in fields:
                    domains[row["domain"]] += 1
                if label_field:
                    labels[row[label_field]] += 1
        fold_checks.append({"path": p.as_posix(), "sha256": sha(p), "tracked_at_head": p.as_posix() in tracked, "columns": fields, "rows": rows, "unique_task_ids": len(groups), "fold_field": fold, "fold_counts": dict(folds), "every_task_in_one_test_fold": None if not fold else all(len(v) == 1 for v in groups.values()), "tasks_split_across_test_folds": [k for k, v in groups.items() if len(v) != 1], "domain_counts": dict(domains), "label_field": label_field, "label_counts": dict(labels), "scope": "Test-fold membership consistency only. Does not independently prove train/calibration split provenance or nested model selection."})
    for doc in family["documents"]:
        obj = doc["content"]
        if not isinstance(obj, dict) or "prepared_manifest_sha256" not in obj:
            continue
        expected = obj["prepared_manifest_sha256"]
        candidates = list(Path(family["directory"]).glob("*manifest.json"))
        matches = [p.as_posix() for p in candidates if sha(p) == expected]
        binding_checks.append({"metrics_path": doc["path"], "expected_prepared_manifest_sha256": expected, "matching_local_manifests": matches, "matches": bool(matches)})
    print(json.dumps({"family": family["directory"], "predictions_checked_so_far": len(fold_checks)}), flush=True)

metadata = []
for p in sorted(Path("research/outputs").rglob("metadata.json")):
    obj = load(p)
    model = obj.get("model", {})
    metadata.append({"path": p.as_posix(), "sha256": sha(p), "tracked_at_head": p.as_posix() in tracked, "model_alias": model.get("alias") if isinstance(model, dict) else model, "model_source": obj.get("model_source", model.get("hf_name") if isinstance(model, dict) else None), "backend": obj.get("backend"), "device": obj.get("device"), "seeds": obj.get("seeds", obj.get("seed")), "temperatures": obj.get("temperatures", obj.get("temperature")), "max_steps": obj.get("max_steps"), "max_new_tokens": obj.get("max_new_tokens"), "max_tasks": obj.get("max_tasks"), "completed_run_count": obj.get("completed_run_count"), "pending_run_count": obj.get("pending_run_count"), "task_manifest_count": len(obj["tasks"]) if isinstance(obj.get("tasks"), list) else None, "task_source": obj.get("task_source"), "declared_dataset_split": obj.get("dataset_split"), "prompt_mode": obj.get("prompt_mode"), "system_prompt_mode": obj.get("system_prompt_mode"), "model_revision": obj.get("model_revision", model.get("revision") if isinstance(model, dict) else None), "dataset_revision": obj.get("dataset_revision"), "checkpoint_reconciliation": obj.get("checkpoint_reconciliation")})
result = {"schema": "saved-target-artifact-readonly-check-v1", "created_utc": datetime.now(timezone.utc).isoformat(), "methods": ["Read every saved prediction CSV in the 33 identified target families using CSV parsing, including multiline-safe reader", "Verify task IDs never span saved outer test folds", "Verify metrics prepared-manifest SHA bindings against actual local manifest bytes", "No pickle/checkpoint/code execution; no retraining, new generation, regrading, or test execution"], "prediction_files": fold_checks, "prepared_manifest_bindings": binding_checks, "prediction_files_checked": len(fold_checks), "total_saved_prediction_rows_scanned": sum(row["rows"] for row in fold_checks), "task_partition_failures": [row["path"] for row in fold_checks if row["every_task_in_one_test_fold"] is False], "prepared_manifest_binding_failures": [row for row in binding_checks if not row["matches"]]}
reports = [write("data_target_artifact_validation.json", result), write("data_generation_metadata_summary.json", {"schema": "saved-generation-metadata-inventory-v1", "created_utc": datetime.now(timezone.utc).isoformat(), "metadata_files": len(metadata), "records": metadata, "scope": "These are recorded collection declarations and completion counters, not reconstructed historical package versions or proof that every trace was produced by a real backend."})]
print(json.dumps({"reports": reports, "prediction_files": len(fold_checks), "rows": result["total_saved_prediction_rows_scanned"], "partition_failures": result["task_partition_failures"], "manifest_failures": result["prepared_manifest_binding_failures"], "generation_metadata_files": len(metadata)}), flush=True)
