"""Inventory saved research artifacts without importing experiment code or changing evidence."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import subprocess

DEST = Path("tmp/research_history_audit")
DEST.mkdir(parents=True, exist_ok=True)


def git(*args):
    return subprocess.check_output(["git", *args]).decode("utf-8")


def write(name, value):
    p = DEST / name
    p.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return {"path": p.as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}


def group_for(path):
    bits = Path(path).parts
    if bits[:3] == ("research", "outputs", "experiments_v2") or bits[:3] == ("research", "outputs", "experiment_matrix"):
        return Path(*bits[:4]).as_posix() if len(bits) > 4 else Path(*bits[:3]).as_posix()
    return Path(*bits[:3]).as_posix() if len(bits) > 3 else Path(*bits[:2]).as_posix()


def target_group(name):
    return any(marker in name for marker in ("strict_", "selected_answer_", "committee_oof_", "prospective_protocol"))


head = git("rev-parse", "HEAD").strip()
tracked = set(git("ls-files", "-z", "--", "research").split("\0")) - {""}
head_paths = set(git("ls-tree", "-r", "--name-only", "-z", "HEAD", "--", "research").split("\0")) - {""}
ignored = set(git("ls-files", "--others", "--ignored", "--exclude-standard", "-z", "--", "research").split("\0")) - {""}
generated_paths = {p for p in tracked | head_paths if p.startswith(("research/outputs/", "research/reports/"))}
for source in (Path("research/outputs"), Path("research/reports")):
    generated_paths.update(p.as_posix() for p in source.rglob("*") if p.is_file())
groups = defaultdict(list)
files = []
json_catalog = []
json_errors = []
for name in sorted(generated_paths):
    p = Path(name)
    present = p.is_file()
    entry = {"path": name, "present_locally": present, "tracked_in_index": name in tracked, "tracked_at_head": name in head_paths, "ignored_local": name in ignored}
    if present:
        stat = p.stat()
        entry.update({"bytes": stat.st_size, "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(), "suffix": p.suffix.lower()})
    groups[group_for(name)].append(entry)
    files.append(entry)
    if present and p.suffix.lower() == ".json" and stat.st_size <= 2_000_000:
        raw = p.read_bytes()
        try:
            content = json.loads(raw.decode("utf-8-sig"))
        except Exception as error:
            json_errors.append({"path": name, "error": str(error)})
            continue
        if isinstance(content, dict):
            scalar = {k: v for k, v in content.items() if isinstance(v, (str, int, float, bool)) or v is None}
            json_catalog.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "keys": list(content), "scalar_metadata": scalar})

group_summary = []
for name, members in sorted(groups.items()):
    group_summary.append({"directory": name, "local_files": sum(e["present_locally"] for e in members), "tracked_files": sum(e["tracked_at_head"] for e in members), "untracked_nonignored_files": sum(e["present_locally"] and not e["tracked_in_index"] and not e["ignored_local"] for e in members), "ignored_local_files": sum(e["ignored_local"] for e in members), "local_bytes": sum(e.get("bytes", 0) for e in members), "extensions": dict(Counter(e.get("suffix", "missing") for e in members)), "trace_steps_present": any(e["path"].endswith("/trace_steps.csv") and e["present_locally"] for e in members), "trace_runs_present": any(e["path"].endswith("/trace_runs.csv") and e["present_locally"] for e in members), "metadata_files": [e["path"] for e in members if Path(e["path"]).name in ("metadata.json", "live_manifest.json", "protocol_manifest.json")], "scope": "Inventory of saved files; file presence alone does not prove full model collection or scientific eligibility."})
inventory = {"schema": "research-artifact-local-inventory-v1", "created_utc": datetime.now(timezone.utc).isoformat(), "head": head, "scope": ["all current committed/index and local files under research/outputs", "all current committed/index and local files under research/reports", "ignored local files included; no pickle, model, or experiment code executed"], "counts": {"paths": len(files), "local_files": sum(e["present_locally"] for e in files), "tracked_at_head": sum(e["tracked_at_head"] for e in files), "untracked_nonignored_local": sum(e["present_locally"] and not e["tracked_in_index"] and not e["ignored_local"] for e in files), "ignored_local": sum(e["ignored_local"] for e in files), "bytes": sum(e.get("bytes", 0) for e in files), "groups": len(group_summary), "json_documents_catalogued": len(json_catalog), "json_parse_errors": len(json_errors)}, "groups": group_summary, "files": files, "json_catalog": json_catalog, "json_parse_errors": json_errors}
paths = [write("data_local_artifact_inventory.json", inventory)]
print(json.dumps({"phase": "local_inventory", "counts": inventory["counts"]}), flush=True)

details = []
for folder in sorted(Path("research/outputs/experiments_v2").iterdir()):
    if not folder.is_dir() or not target_group(folder.name):
        continue
    documents = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".json", ".md") and p.stat().st_size <= 2_000_000:
            raw = p.read_bytes()
            documents.append({"path": p.as_posix(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "tracked_at_head": p.as_posix() in head_paths, "content": json.loads(raw.decode("utf-8-sig")) if p.suffix.lower() == ".json" else raw.decode("utf-8-sig")})
    files_in_folder = [p for p in folder.rglob("*") if p.is_file()]
    notes = []
    if any(p.name == "REJECTED_PROTOCOL.md" for p in files_in_folder):
        notes.append("explicitly_rejected_protocol")
    if any(p.name == "INCOMPLETE_NOT_FOR_COMPARISON.md" for p in files_in_folder):
        notes.append("explicitly_incomplete_pair_not_for_comparison")
    if "prospective_protocol" in folder.name:
        notes.append("protocol_smoke_or_preparation_only; inspect confirmation eligibility and actual event content")
    else:
        notes.append("saved retrospective training/OOF analysis; not new model generation")
    details.append({"directory": folder.as_posix(), "files": [p.as_posix() for p in files_in_folder], "tracked_files": sum(p.as_posix() in head_paths for p in files_in_folder), "classification_notes": notes, "documents": documents})
paths.append(write("data_target_family_evidence.json", {"schema": "research-target-family-saved-evidence-v1", "created_utc": datetime.now(timezone.utc).isoformat(), "head": head, "families": details}))

history_text = git("log", "--all", "--format=%x1e%H%x1f%aI%x1f%cI%x1f%s", "--name-status", "--", "research/outputs", "research/reports")
history = []
historical_paths = set()
for block in history_text.split("\x1e"):
    if not block.strip():
        continue
    lines = block.strip().splitlines()
    fields = lines[0].split("\x1f", 3)
    if len(fields) != 4:
        raise RuntimeError("Unexpected Git log record")
    changes = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 2:
            changes.append({"status": parts[0], "paths": parts[1:]})
            historical_paths.update(parts[1:])
    history.append({"commit": fields[0], "author_date": fields[1], "committer_date": fields[2], "subject": fields[3], "artifact_changes": changes})
head_commits = set(git("rev-list", "HEAD").splitlines())
for record in history:
    record["reachable_from_current_head"] = record["commit"] in head_commits
history_result = {"schema": "research-artifact-history-inventory-v1", "created_utc": datetime.now(timezone.utc).isoformat(), "head": head, "all_refs_commit_count": int(git("rev-list", "--all", "--count")), "head_reachable_commit_count": len(head_commits), "commits_touching_artifacts": len(history), "unique_historical_artifact_paths": len(historical_paths), "historical_paths_missing_locally": sorted(p for p in historical_paths if not Path(p).is_file()), "history": history, "method": "git log --all --name-status; backup/rewritten histories are explicitly separated from current-head ancestry, not counted as independent scientific runs."}
paths.append(write("data_artifact_history_inventory.json", history_result))
print(json.dumps({"phase": "history_and_target_documents", "target_families": len(details), "artifact_history_commits": len(history), "unique_historical_paths": len(historical_paths), "head_commits": len(head_commits), "reports": paths}), flush=True)
