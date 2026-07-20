"""Collect an auditable, barrier-synchronous fixed-fleet reasoning panel.

This is deliberately separate from ``real_trace_experiments.py``.  The legacy
collector completes one model's full trajectory before beginning the next one,
which is appropriate for retrospective trace analysis but cannot prove that a
peer vote was available at a live stopping decision.  This coordinator makes a
barrier -- one task, replica, and reasoning step across a frozen roster -- the
atomic source of truth.

The collector has two phases:

* ``--initialize`` writes a public task manifest (without gold labels) and a
  separately stored label ledger.  It rejects prompts found in historical
  output by default.
* ``--collect`` generates every roster member at each barrier, closes the
  barrier only after the complete roster is present, and emits a fixed-policy
  selected answer.  Correctness is never written to an event, barrier, or
  decision ledger.

Each canonical barrier is an atomically written JSON commit under
``barrier_commits/``.  The JSONL ledgers are deterministic derivatives, so a
hard interruption cannot leave a partially visible peer panel.  Run
``audit_prospective_protocol.py`` before fitting or reporting any policy.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from transformers import set_seed

from real_trace_experiments import (
    MODEL_CATALOG,
    ModelSpec,
    TaskSpec,
    conversation_prompt,
    load_model,
    load_tasks,
    normalize_answer,
    parse_generation,
    release_cuda_memory,
    render_prompt,
    safe_generate_batch_with_diagnostics,
)
from forced_choice_verifier import (
    ANONYMOUS_RATIONALE_V1,
    OPTION_LABELS,
    PROMPT_OPTIONS_ONLY_V1,
    SCORE_METHOD as VERIFIER_SCORE_METHOD,
    SUPPORTED_VARIANTS as SUPPORTED_VERIFIER_VARIANTS,
    TOKEN_ACCOUNTING_CONTRACT,
    frozen_option_continuations,
    render_verifier_prompt,
    score_forced_choice_prompt_batch,
    verifier_template_sha256,
)


SCHEMA_VERSION_V1 = "prospective-committee-v1"
SCHEMA_VERSION_V2 = "prospective-committee-v2"
PUBLIC_TASKS_NAME = "prospective_tasks.jsonl"
SEALED_LABELS_NAME = "sealed_gold_labels.jsonl"
PROTOCOL_MANIFEST_NAME = "protocol_manifest.json"
STATUS_NAME = "protocol_status.json"
COMMITS_DIR_NAME = "barrier_commits"
DEFAULT_HISTORICAL_ROOT = Path(__file__).resolve().parent / "outputs" / "experiments_v2"

# This is the historical fleet for which a retrospective signal exists.  The
# roster remains an argument because a truly prospective confirmation must
# pre-register it before collection rather than choose it from the test panel.
DEFAULT_ROSTER = (
    "deepseek_r1_distill_1p5b",
    "deepseek_r1_distill_7b",
    "llama_3p1_8b_instruct",
    "mistral_7b_instruct_v0p3",
    "mistral_small_24b_2409",
    "phi_4_mini_instruct",
    "qwen2p5_0p5b",
    "qwen2p5_3b",
    "qwen2p5_7b",
    "qwen2p5_14b",
    "qwen2p5_32b",
    "qwen_3p5_9b",
    "yi_1p5_9b_chat",
)

FORBIDDEN_PUBLIC_KEYS = {
    "correct",
    "expected_answer",
    "gold",
    "gold_answer",
    "label",
    "utility",
    "oracle_stop",
    "selected_correct",
}


class ProtocolError(RuntimeError):
    """Raised when an invariant needed for confirmatory use is violated."""


def verifier_is_enabled(args: argparse.Namespace) -> bool:
    return str(getattr(args, "verifier_mode", "off")) == "collect_only"


def schema_version_for_args(args: argparse.Namespace) -> str:
    """Keep v1 byte contracts untouched unless a post-barrier verifier is on."""

    return SCHEMA_VERSION_V2 if verifier_is_enabled(args) else SCHEMA_VERSION_V1


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def utc_strictly_after(values: Iterable[str]) -> str:
    """Return a UTC timestamp strictly later than every recorded input.

    Windows' wall clock can have a coarser resolution than a very short CUDA
    event.  A barrier must nevertheless have an unambiguous closure timestamp
    for the external auditor, so advance by one microsecond when necessary.
    """
    parsed: list[datetime] = []
    for value in values:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed.append(datetime.fromisoformat(text).astimezone(UTC))
    current = datetime.now(UTC)
    latest = max(parsed)
    if current <= latest:
        current = latest + timedelta(microseconds=1)
    return current.isoformat(timespec="microseconds").replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    """Convert NumPy/Torch scalars and non-finite floats into strict JSON.

    Strict JSON is important because commit hashes must be stable across Python
    versions and because JSON's spellings for NaN/Infinity are non-standard.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("Only scalar tensors can be serialized into a protocol ledger.")
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, prefix=f".{path.name}.") as handle:
        temporary_path = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_bytes(path, canonical_json_bytes(payload) + b"\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    content = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    atomic_write_bytes(path, content)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected a JSON object in {path}.")
    return payload


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ProtocolError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc
                if not isinstance(item, dict):
                    raise ProtocolError(f"Expected an object in {path}:{line_number}.")
                yield item
    except OSError as exc:
        raise ProtocolError(f"Cannot read {path}: {exc}") from exc


def require_no_forbidden_keys(record: dict[str, Any], context: str) -> None:
    found = FORBIDDEN_PUBLIC_KEYS & set(record)
    if found:
        raise ProtocolError(f"{context} exposes forbidden label/oracle fields: {sorted(found)}")


def normalize_roster(roster: Iterable[str]) -> tuple[str, ...]:
    aliases = tuple(str(alias).strip() for alias in roster if str(alias).strip())
    if not aliases:
        raise ProtocolError("The roster cannot be empty.")
    if len(set(aliases)) != len(aliases):
        raise ProtocolError("The roster contains duplicate model aliases.")
    unknown = sorted(set(aliases) - set(MODEL_CATALOG))
    if unknown:
        raise ProtocolError(f"Unknown roster aliases: {unknown}")
    return aliases


def prompt_hash(prompt: str) -> str:
    # Normalize line endings so the same frozen task has the same identity on
    # Windows and Linux.  Whitespace is otherwise preserved deliberately.
    return sha256_text(str(prompt).replace("\r\n", "\n").replace("\r", "\n"))


def opaque_task_id(task: TaskSpec) -> str:
    """Return a source-free task identifier used by every live ledger.

    Source task IDs and source-row indices are convenient retrospective keys,
    but they are forbidden policy inputs.  The fixed prompt hash already
    provides a collision-resistant identity, so use a short opaque derivative
    instead of preserving any source-specific numbering.
    """
    return f"task_{prompt_hash(task.prompt)[:24]}"


def task_record_from_spec(task: TaskSpec, dataset_revision: str | None) -> dict[str, Any]:
    return {
        "task_id": opaque_task_id(task),
        "domain": task.domain,
        "difficulty": task.difficulty,
        "prompt": task.prompt,
        "prompt_sha256": prompt_hash(task.prompt),
        "answer_type": task.answer_type,
        "notes": task.notes,
        "source": task.source,
        "dataset_revision": dataset_revision,
    }


def label_record_from_spec(task: TaskSpec) -> dict[str, Any]:
    return {
        "task_id": opaque_task_id(task),
        "prompt_sha256": prompt_hash(task.prompt),
        "expected_answer": task.expected_answer,
    }


def historical_prompt_hashes(root: Path) -> dict[str, list[str]]:
    """Find historical task prompts without loading their labels into policy data."""
    if not root.exists():
        return {}
    matches: dict[str, list[str]] = defaultdict(list)
    for metadata_path in root.rglob("metadata.json"):
        try:
            metadata = load_json(metadata_path)
        except ProtocolError:
            continue
        tasks = metadata.get("tasks")
        if not isinstance(tasks, list):
            continue
        for record in tasks:
            if not isinstance(record, dict) or not isinstance(record.get("prompt"), str):
                continue
            matches[prompt_hash(record["prompt"])].append(str(metadata_path))
    return dict(matches)


def write_if_absent_or_identical(path: Path, content: bytes) -> None:
    if path.exists():
        existing = path.read_bytes()
        if existing != content:
            raise ProtocolError(
                f"Refusing to overwrite an existing immutable artifact with different content: {path}"
            )
        return
    atomic_write_bytes(path, content)


def initialize_task_manifests(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    public_path = output_dir / PUBLIC_TASKS_NAME
    labels_path = output_dir / SEALED_LABELS_NAME
    if (output_dir / PROTOCOL_MANIFEST_NAME).exists():
        raise ProtocolError(
            "Cannot initialize tasks after protocol_manifest.json exists. Start a new output directory for a new protocol."
        )
    if not args.dataset_revision:
        raise ProtocolError("--dataset-revision is required for --initialize; use an immutable dataset revision SHA.")

    tasks = load_tasks(
        task_source=args.task_source,
        max_tasks=args.max_tasks,
        dataset_split=args.dataset_split,
        shuffle_seed=args.dataset_shuffle_seed,
        dataset_revision=args.dataset_revision,
    )
    if not tasks:
        raise ProtocolError("The requested initialization selected zero tasks.")
    public_rows = [task_record_from_spec(task, args.dataset_revision) for task in tasks]
    label_rows = [label_record_from_spec(task) for task in tasks]
    task_ids = [row["task_id"] for row in public_rows]
    hashes = [row["prompt_sha256"] for row in public_rows]
    if len(set(task_ids)) != len(task_ids):
        raise ProtocolError("The selected source yielded duplicate task_id values.")
    if len(set(hashes)) != len(hashes):
        raise ProtocolError("The selected source yielded duplicate prompts; use a deduplicated panel.")

    historical = historical_prompt_hashes(Path(args.historical_root))
    overlap = {value: historical[value] for value in hashes if value in historical}
    if overlap and not args.allow_historical_overlap:
        examples = sorted({path for paths in overlap.values() for path in paths})[:5]
        raise ProtocolError(
            f"Refusing {len(overlap)} task(s) already present in historical traces. "
            f"Use a task-disjoint source/split; examples: {examples}. "
            "For an explicitly non-confirmatory seed replication only, pass --allow-historical-overlap."
        )

    write_if_absent_or_identical(public_path, b"".join(canonical_json_bytes(row) + b"\n" for row in public_rows))
    write_if_absent_or_identical(labels_path, b"".join(canonical_json_bytes(row) + b"\n" for row in label_rows))
    summary = {
        "schema_version": schema_version_for_args(args),
        "created_at_utc": utc_now(),
        "task_count": len(public_rows),
        "task_manifest": public_path.name,
        "task_manifest_sha256": sha256_file(public_path),
        "sealed_labels": labels_path.name,
        "sealed_labels_sha256": sha256_file(labels_path),
        "dataset": {
            "source": args.task_source,
            "split": args.dataset_split,
            "shuffle_seed": args.dataset_shuffle_seed,
            "revision": args.dataset_revision,
        },
        "historical_overlap_count": len(overlap),
        "confirmatory_task_disjoint": not bool(overlap),
    }
    atomic_write_json(output_dir / "task_initialization.json", summary)
    print(
        f"Initialized {len(public_rows)} frozen public tasks in {output_dir}. "
        f"Task-disjoint={summary['confirmatory_task_disjoint']}",
        flush=True,
    )


def load_frozen_tasks(output_dir: Path) -> tuple[list[TaskSpec], list[dict[str, Any]], str, str]:
    public_path = output_dir / PUBLIC_TASKS_NAME
    labels_path = output_dir / SEALED_LABELS_NAME
    if not public_path.exists() or not labels_path.exists():
        raise ProtocolError(
            f"Missing frozen task artifacts. Run --initialize first: {public_path.name}, {labels_path.name}"
        )
    public_rows = list(iter_jsonl(public_path))
    label_rows = list(iter_jsonl(labels_path))
    if not public_rows:
        raise ProtocolError("The public task manifest is empty.")

    labels: dict[str, dict[str, Any]] = {}
    for label in label_rows:
        required = {"task_id", "prompt_sha256", "expected_answer"}
        if not required <= set(label):
            raise ProtocolError(f"A sealed-label record is missing {sorted(required - set(label))}.")
        task_id = str(label["task_id"])
        if task_id in labels:
            raise ProtocolError(f"Duplicate sealed label for task_id={task_id!r}.")
        labels[task_id] = label

    tasks: list[TaskSpec] = []
    seen_ids: set[str] = set()
    for row in public_rows:
        require_no_forbidden_keys(row, f"public task {row.get('task_id', '<unknown>')}")
        required = {
            "task_id",
            "domain",
            "difficulty",
            "prompt",
            "prompt_sha256",
            "answer_type",
            "notes",
            "source",
        }
        if not required <= set(row):
            raise ProtocolError(f"A public task is missing {sorted(required - set(row))}.")
        task_id = str(row["task_id"])
        if task_id in seen_ids:
            raise ProtocolError(f"Duplicate public task_id={task_id!r}.")
        seen_ids.add(task_id)
        if prompt_hash(str(row["prompt"])) != str(row["prompt_sha256"]):
            raise ProtocolError(f"Prompt hash mismatch for task_id={task_id!r}.")
        label = labels.get(task_id)
        if label is None:
            raise ProtocolError(f"No sealed label for task_id={task_id!r}.")
        if str(label["prompt_sha256"]) != str(row["prompt_sha256"]):
            raise ProtocolError(f"Sealed-label prompt hash mismatch for task_id={task_id!r}.")
        tasks.append(
            TaskSpec(
                task_id=task_id,
                domain=str(row["domain"]),
                difficulty=str(row["difficulty"]),
                prompt=str(row["prompt"]),
                answer_type=str(row["answer_type"]),
                expected_answer=str(label["expected_answer"]),
                notes=str(row["notes"]),
                source=str(row["source"]),
                source_index=-1,
            )
        )
    if set(labels) != seen_ids:
        extra = sorted(set(labels) - seen_ids)[:5]
        raise ProtocolError(f"Sealed labels contain task IDs not present in the public manifest: {extra}")
    return tasks, public_rows, sha256_file(public_path), sha256_file(labels_path)


def load_model_revisions(path: str | None, roster: tuple[str, ...], allow_unpinned: bool) -> dict[str, str]:
    if path is None:
        if not allow_unpinned:
            raise ProtocolError(
                "--model-revisions is required for collection. Supply a JSON object mapping every roster alias "
                "to a full Hugging Face commit SHA."
            )
        return {alias: "UNPINNED" for alias in roster}
    revision_path = Path(path)
    payload = load_json(revision_path)
    revisions = payload.get("model_revisions", payload)
    if not isinstance(revisions, dict):
        raise ProtocolError("--model-revisions must be a JSON object or contain a model_revisions object.")
    output: dict[str, str] = {}
    for alias in roster:
        revision = revisions.get(alias)
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{7,64}", revision):
            raise ProtocolError(f"Model revision for {alias!r} must be a 7-64 character hexadecimal commit SHA.")
        output[alias] = revision.lower()
    extra = sorted(set(revisions) - set(roster))
    if extra:
        raise ProtocolError(f"Revision map contains aliases outside the frozen roster: {extra}")
    return output


def runtime_identity(requested_device: str) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    resolved = requested_device if requested_device != "auto" else ("cuda" if cuda_available else "cpu")
    identity: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_available": cuda_available,
        "requested_device": requested_device,
        "resolved_device": resolved,
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        identity["cuda_runtime"] = str(torch.version.cuda)
        identity["gpu_name"] = properties.name
        identity["gpu_total_vram_bytes"] = int(properties.total_memory)
        identity["gpu_compute_capability"] = f"{properties.major}.{properties.minor}"
    return identity


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    identity = runtime_identity(args.device)
    if args.device == "cpu":
        if not args.allow_low_vram:
            raise ProtocolError("A confirmatory roster collection requires CUDA; --device cpu is only permitted with --allow-low-vram.")
        return identity
    if not identity["cuda_available"]:
        if not args.allow_low_vram:
            raise ProtocolError("CUDA is unavailable. Refusing a confirmatory GPU collection.")
        return identity
    vram_gb = float(identity["gpu_total_vram_bytes"]) / (1024**3)
    identity["gpu_total_vram_gib"] = vram_gb
    if vram_gb < args.min_vram_gb and not args.allow_low_vram:
        raise ProtocolError(
            f"Detected {vram_gb:.1f} GiB VRAM, below the registered minimum of {args.min_vram_gb:.1f} GiB. "
            "Use the intended Blackwell host, or pass --allow-low-vram only for a non-confirmatory smoke test."
        )
    return identity


def model_records(roster: tuple[str, ...], revisions: dict[str, str]) -> list[dict[str, Any]]:
    records = []
    for alias in roster:
        record = asdict(MODEL_CATALOG[alias])
        record["revision"] = revisions[alias]
        records.append(record)
    return records


def verifier_model_spec_from_args(args: argparse.Namespace) -> ModelSpec | None:
    """Return a ledger-distinct specification for the optional verifier.

    The verifier may reuse a base checkpoint that is also present in the main
    roster, but it must have a distinct alias in the event ledger.  Otherwise a
    post-barrier verifier event could be mistaken for a main roster generation.
    """

    if not verifier_is_enabled(args):
        return None
    base_alias = str(args.verifier_model)
    if base_alias not in MODEL_CATALOG:
        raise ProtocolError(f"Unknown --verifier-model alias {base_alias!r}.")
    ledger_alias = str(args.verifier_ledger_alias).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,128}", ledger_alias):
        raise ProtocolError("--verifier-ledger-alias must use only letters, digits, dot, underscore, or hyphen.")
    if ledger_alias in set(args.roster):
        raise ProtocolError("--verifier-ledger-alias must be distinct from every main roster alias.")
    base = MODEL_CATALOG[base_alias]
    return ModelSpec(
        alias=ledger_alias,
        hf_name=base.hf_name,
        family=base.family,
        parameter_count=base.parameter_count,
    )


def load_verifier_revision(args: argparse.Namespace, verifier_spec: ModelSpec | None) -> str | None:
    if verifier_spec is None:
        return None
    revision = args.verifier_model_revision
    if revision is None:
        if args.allow_unpinned_models:
            return "UNPINNED"
        raise ProtocolError(
            "--verifier-model-revision is required when --verifier-mode=collect_only; "
            "supply the frozen Hugging Face commit SHA."
        )
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{7,64}", revision):
        raise ProtocolError("--verifier-model-revision must be a 7-64 character hexadecimal commit SHA.")
    return revision.lower()


def verifier_record(spec: ModelSpec | None, revision: str | None) -> list[dict[str, Any]]:
    if spec is None or revision is None:
        return []
    record = asdict(spec)
    record["revision"] = revision
    return [record]


def verifier_spec_from_args(args: argparse.Namespace, verifier_spec: ModelSpec | None) -> dict[str, Any] | None:
    if verifier_spec is None:
        return None
    variants = tuple(str(value) for value in args.verifier_variants)
    if not variants or len(set(variants)) != len(variants) or any(value not in SUPPORTED_VERIFIER_VARIANTS for value in variants):
        raise ProtocolError(f"--verifier-variants must be a unique non-empty subset of {list(SUPPORTED_VERIFIER_VARIANTS)}.")
    active_variant = str(args.verifier_active_variant)
    if active_variant not in variants:
        raise ProtocolError("--verifier-active-variant must be listed in --verifier-variants.")
    if args.verifier_batch_size < 1:
        raise ProtocolError("--verifier-batch-size must be positive.")
    if args.verifier_rationale_max_chars < 1:
        raise ProtocolError("--verifier-rationale-max-chars must be positive.")
    continuations = frozen_option_continuations()
    return {
        "spec_id": "gpqa_forced_choice_v1",
        "mode": "collect_only",
        "ledger_alias": verifier_spec.alias,
        "base_model_alias": str(args.verifier_model),
        "variants": list(variants),
        "active_variant": active_variant,
        "option_labels": list(OPTION_LABELS),
        "continuations": continuations,
        "score_method": VERIFIER_SCORE_METHOD,
        "token_accounting_contract": TOKEN_ACCOUNTING_CONTRACT,
        "template_sha256": {variant: verifier_template_sha256(variant) for variant in variants},
        "rationale_max_chars": int(args.verifier_rationale_max_chars),
        "post_barrier_stage": "mandatory_after_main_barrier_close_v1",
    }


def configuration_from_args(args: argparse.Namespace) -> dict[str, Any]:
    configuration = {
        "replicas": [int(value) for value in args.replicas],
        "max_steps": int(args.max_steps),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "batch_size": int(args.batch_size),
        "seed_mode": args.seed_mode,
        "device": args.device,
        "quantization": args.quantization,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "prompt_mode": args.prompt_mode,
        "system_prompt_mode": args.system_prompt_mode,
        "extended_observables": bool(args.extended_observables),
    }
    # Preserve the byte-level v1 manifest contract when no verifier is
    # requested; only v2 protocols declare these additional collection knobs.
    if verifier_is_enabled(args):
        configuration.update(
            {
                "verifier_mode": str(args.verifier_mode),
                "verifier_model": str(args.verifier_model),
                "verifier_ledger_alias": str(args.verifier_ledger_alias),
                "verifier_variants": [str(value) for value in args.verifier_variants],
                "verifier_active_variant": str(args.verifier_active_variant),
                "verifier_batch_size": int(args.verifier_batch_size),
                "verifier_rationale_max_chars": int(args.verifier_rationale_max_chars),
            }
        )
    return configuration


def policy_from_args(args: argparse.Namespace, roster: tuple[str, ...]) -> dict[str, Any]:
    if args.policy == "leader" and args.leader not in roster:
        raise ProtocolError("The fixed leader must belong to the frozen roster.")
    if args.policy in {"consensus", "legal_consensus"} and args.leader not in roster:
        # It is only a no-vote fallback but must still be frozen and real.
        raise ProtocolError("The consensus fallback leader must belong to the frozen roster.")
    selection_rule_id = {
        "leader": "fixed_leader_v1",
        "consensus": "plurality_normalized_answer_v1",
        # This is deliberately a post-generation selector, not a claim that
        # decoding itself was constrained.  It is appropriate for the four
        # option ARC/GPQA task families and prevents an E/F/G/H parse failure
        # from winning merely because several models made the same mistake.
        "legal_consensus": "plurality_legal_mcq_answer_v1",
    }[args.policy]
    # Raw answer strings are used only inside the deterministic selector.  The
    # registered score-facing contract is intentionally aggregate-only.
    feature_contract = [
        "selected_support_fraction",
        "valid_answer_count",
        "selected_parse_success",
        "selected_confidence",
    ]
    feature_contract_id = "selection_observables_v1"
    if args.policy == "legal_consensus":
        feature_contract = [
            *feature_contract,
            "legal_answer_count",
            "invalid_answer_count",
        ]
        feature_contract_id = "selection_observables_legal_mcq_v1"
    return {
        "policy_type": args.policy,
        "leader_alias": args.leader,
        "selection_rule_id": selection_rule_id,
        "tie_break": "first_supporting_alias_in_frozen_roster_order",
        "peer_visibility": "post_complete_barrier_only",
        "requires_full_roster": True,
        "feature_contract_id": feature_contract_id,
        "feature_contracts": {feature_contract_id: feature_contract},
        "legal_mcq_options": ["A", "B", "C", "D"] if args.policy == "legal_consensus" else None,
        "uses_gold_or_correctness": False,
    }


def source_code_fingerprint() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "run_prospective_committee_protocol.py": sha256_file(Path(__file__).resolve()),
        "real_trace_experiments.py": sha256_file(root / "real_trace_experiments.py"),
        "forced_choice_verifier.py": sha256_file(root / "forced_choice_verifier.py"),
    }


def create_or_validate_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    roster: tuple[str, ...],
    revisions: dict[str, str],
    task_manifest_sha: str,
    labels_sha: str,
    task_count: int,
    runtime: dict[str, Any],
    verifier_members: list[dict[str, Any]],
    verifier_config: dict[str, Any] | None,
) -> tuple[dict[str, Any], str, str]:
    manifest_path = output_dir / PROTOCOL_MANIFEST_NAME
    models = model_records(roster, revisions)
    policy = policy_from_args(args, roster)
    configuration = configuration_from_args(args)
    # This collector pre-registers and closes live peer barriers, but collects a
    # full horizon so that a later development/calibration stage can fit a
    # stopping score.  It must never be reported as an end-to-end deployment
    # confirmation until a frozen score, calibration map, and threshold are
    # supplied to a separate online policy runner.
    confirmation_eligible = False
    expected = {
        "schema_version": schema_version_for_args(args),
        "protocol_id": args.protocol_id,
        "phase": args.phase,
        "task_manifest": {
            "path": PUBLIC_TASKS_NAME,
            "sha256": task_manifest_sha,
            "count": int(task_count),
        },
        "sealed_labels": {
            "path": SEALED_LABELS_NAME,
            "sha256": labels_sha,
            "count": int(task_count),
            "not_available_to_policy": True,
        },
        "models": models,
        "roster_sha256": sha256_bytes(canonical_json_bytes(models)),
        "policy": policy,
        "policy_sha256": sha256_bytes(canonical_json_bytes(policy)),
        "configuration": configuration,
        "code_sha256": source_code_fingerprint(),
        "randomness": {
            "event_seed_derivation": (
                "sha256(protocol_id|task_prompt_sha256|replica_id|step|model_alias|generation_kind_or_verifier_variant)"
                if verifier_config is not None
                else "sha256(protocol_id|task_prompt_sha256|replica_id|step|model_alias|generation_kind)"
            ),
            "seed_scope": "per_event" if args.seed_mode == "per_event" else "batch_seeded_nonconfirmatory",
        },
        "collection_mode": "full_horizon_observational",
        "stopping_policy": {
            "registered": False,
            "reason": "No frozen stopping score, calibration map, or threshold is installed in this collector.",
        },
        "confirmation_eligible": confirmation_eligible,
    }
    if verifier_config is not None:
        expected.update(
            {
                "verifier_roster": verifier_members,
                "verifier_roster_sha256": sha256_bytes(canonical_json_bytes(verifier_members)),
                "verifier_spec": verifier_config,
            }
        )
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ProtocolError(
                    f"Existing protocol manifest differs at {key!r}. A registered protocol is immutable; "
                    "use a new output directory rather than mixing conditions."
                )
        return manifest, sha256_bytes(canonical_json_bytes(manifest)), sha256_file(manifest_path)

    manifest = {
        **expected,
        "created_at_utc": utc_now(),
        "runtime_preflight": runtime,
        "immutable": True,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest, sha256_bytes(canonical_json_bytes(manifest)), sha256_file(manifest_path)


def derived_seed(
    protocol_id: str,
    task_prompt_sha256: str,
    replica_id: int,
    step: int,
    model_alias: str,
    generation_kind: str,
) -> int:
    material = "|".join(
        [protocol_id, task_prompt_sha256, str(replica_id), str(step), model_alias, generation_kind]
    )
    # Transformers/Python RNGs accept 32-bit seeds across all currently
    # supported CUDA versions.  The full identity remains in the ledger.
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big")


def barrier_id(protocol_id: str, task_prompt_sha256: str, replica_id: int, step: int) -> str:
    material = "|".join([protocol_id, task_prompt_sha256, str(replica_id), str(step)])
    return sha256_text(material)[:32]


def event_id(
    barrier: str,
    model_alias: str,
    generation_kind: str = "main",
    verifier_variant: str | None = None,
) -> str:
    """Derive a stable event ID while preserving the exact v1 main-event IDs."""

    if generation_kind == "main" and verifier_variant is None:
        return sha256_text(f"{barrier}|{model_alias}|main")[:32]
    suffix = verifier_variant if verifier_variant is not None else generation_kind
    return sha256_text(f"{barrier}|{model_alias}|{generation_kind}|{suffix}")[:32]


def task_prompt_token_count(tokenizer: Any, prompt: str) -> int:
    try:
        encoded = tokenizer(prompt, add_special_tokens=False)
        ids = encoded.get("input_ids", []) if isinstance(encoded, dict) else []
        return int(len(ids))
    except Exception as exc:
        raise ProtocolError(f"Failed to tokenize a rendered prompt for ledger accounting: {exc}") from exc


def existing_commits(
    commits_dir: Path,
    manifest_sha: str,
    protocol_manifest_file_sha: str,
    roster: tuple[str, ...],
    schema_version: str,
    verifier_config: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    commits: dict[str, dict[str, Any]] = {}
    if not commits_dir.exists():
        return commits
    for path in sorted(commits_dir.glob("*.json")):
        commit = load_json(path)
        recorded_hash = commit.get("payload_sha256")
        unsigned = dict(commit)
        unsigned.pop("payload_sha256", None)
        if not isinstance(recorded_hash, str) or recorded_hash != sha256_bytes(canonical_json_bytes(unsigned)):
            raise ProtocolError(f"Barrier commit integrity failure: {path}")
        if commit.get("schema_version") != schema_version:
            raise ProtocolError(f"Barrier commit has the wrong schema version: {path}")
        if (
            commit.get("protocol_manifest_sha256") != protocol_manifest_file_sha
            or commit.get("manifest_sha256") != manifest_sha
        ):
            raise ProtocolError(f"Barrier commit belongs to a different protocol manifest: {path}")
        barrier_identifier = str(commit.get("barrier_id", ""))
        if not barrier_identifier or path.stem != barrier_identifier:
            raise ProtocolError(f"Barrier commit filename/identifier mismatch: {path}")
        if barrier_identifier in commits:
            raise ProtocolError(f"Duplicate barrier commit identifier {barrier_identifier!r}.")
        events = commit.get("events")
        if not isinstance(events, list) or len(events) != len(roster):
            raise ProtocolError(f"Barrier commit does not contain the complete frozen roster: {path}")
        aliases = [str(row.get("model_alias", "")) for row in events if isinstance(row, dict)]
        if tuple(aliases) != roster:
            raise ProtocolError(f"Barrier event ordering/roster mismatch: {path}")
        for row in events:
            if not isinstance(row, dict):
                raise ProtocolError(f"Barrier event is not an object: {path}")
            require_no_forbidden_keys(row, f"event in {path.name}")
            event_hash = row.get("event_sha256")
            unsigned_event = dict(row)
            unsigned_event.pop("event_sha256", None)
            if not isinstance(event_hash, str) or event_hash != sha256_bytes(canonical_json_bytes(unsigned_event)):
                raise ProtocolError(f"Event hash integrity failure: {path}")
            if (
                row.get("manifest_sha256") != manifest_sha
                or row.get("protocol_manifest_sha256") != protocol_manifest_file_sha
            ):
                raise ProtocolError(f"Event manifest hash mismatch: {path}")
            if str(row.get("barrier_id", "")) != barrier_identifier or str(row.get("generation_kind", "")) != "main":
                raise ProtocolError(f"Main event scope or generation kind mismatch: {path}")
        main_hashes = commit.get("event_hashes")
        main_event_ids = {str(row.get("event_id", "")) for row in events if isinstance(row, dict)}
        if not isinstance(main_hashes, dict) or set(main_hashes) != main_event_ids:
            raise ProtocolError(f"Main event hashes are incomplete or malformed: {path}")
        if any(main_hashes.get(str(row.get("event_id", ""))) != row.get("event_sha256") for row in events):
            raise ProtocolError(f"Main event hash map mismatch: {path}")
        post_events = commit.get("post_events", [])
        if not isinstance(post_events, list):
            raise ProtocolError(f"Barrier commit post_events must be a list: {path}")
        if schema_version == SCHEMA_VERSION_V1 and post_events:
            raise ProtocolError(f"A v1 barrier commit may not contain verifier post-events: {path}")
        if schema_version == SCHEMA_VERSION_V2:
            if verifier_config is None:
                raise ProtocolError("A v2 commit cannot be resumed without its frozen verifier configuration.")
            expected_variants = {str(value) for value in verifier_config.get("variants", [])}
            observed_variants = {str(row.get("verifier_variant", "")) for row in post_events if isinstance(row, dict)}
            expected_alias = str(verifier_config.get("ledger_alias", ""))
            if not post_events or observed_variants != expected_variants or len(post_events) != len(expected_variants):
                raise ProtocolError(f"A v2 barrier commit does not contain exactly the frozen verifier variants: {path}")
            if not expected_alias or any(str(row.get("model_alias", "")) != expected_alias for row in post_events if isinstance(row, dict)):
                raise ProtocolError(f"A v2 verifier alias differs from the frozen ledger alias: {path}")
        post_hashes = commit.get("post_event_hashes", {})
        if not isinstance(post_hashes, dict) or set(post_hashes) != {
            str(row.get("event_id", "")) for row in post_events if isinstance(row, dict)
        }:
            raise ProtocolError(f"Verifier post-event hashes are incomplete or malformed: {path}")
        seen_post_ids: set[str] = set()
        for row in post_events:
            if not isinstance(row, dict) or str(row.get("generation_kind")) != "verifier":
                raise ProtocolError(f"Barrier post-event is not a verifier object: {path}")
            require_no_forbidden_keys(row, f"verifier event in {path.name}")
            post_identifier = str(row.get("event_id", ""))
            if (
                not post_identifier
                or post_identifier in seen_post_ids
                or post_identifier in {str(item.get("event_id", "")) for item in events}
            ):
                raise ProtocolError(f"Barrier post-event identifiers are not unique: {path}")
            seen_post_ids.add(post_identifier)
            event_hash = row.get("event_sha256")
            unsigned_event = dict(row)
            unsigned_event.pop("event_sha256", None)
            if not isinstance(event_hash, str) or event_hash != sha256_bytes(canonical_json_bytes(unsigned_event)):
                raise ProtocolError(f"Verifier event hash integrity failure: {path}")
            if post_hashes.get(post_identifier) != event_hash:
                raise ProtocolError(f"Verifier post-event hash map mismatch: {path}")
            if (
                row.get("manifest_sha256") != manifest_sha
                or row.get("protocol_manifest_sha256") != protocol_manifest_file_sha
            ):
                raise ProtocolError(f"Verifier event manifest hash mismatch: {path}")
            if str(row.get("barrier_id", "")) != barrier_identifier:
                raise ProtocolError(f"Verifier event scope mismatch: {path}")
        decision = commit.get("decision")
        if not isinstance(decision, dict):
            raise ProtocolError(f"Barrier commit is missing its policy decision: {path}")
        require_no_forbidden_keys(decision, f"decision in {path.name}")
        decision_hash = decision.get("decision_sha256")
        unsigned_decision = dict(decision)
        unsigned_decision.pop("decision_sha256", None)
        if not isinstance(decision_hash, str) or decision_hash != sha256_bytes(canonical_json_bytes(unsigned_decision)):
            raise ProtocolError(f"Decision hash integrity failure: {path}")
        barrier_record = commit.get("barrier")
        if not isinstance(barrier_record, dict):
            raise ProtocolError(f"Barrier record is missing: {path}")
        barrier_hash = barrier_record.get("barrier_sha256")
        unsigned_barrier = dict(barrier_record)
        unsigned_barrier.pop("barrier_sha256", None)
        if not isinstance(barrier_hash, str) or barrier_hash != sha256_bytes(canonical_json_bytes(unsigned_barrier)):
            raise ProtocolError(f"Barrier hash integrity failure: {path}")
        commits[barrier_identifier] = commit
    return commits


def histories_from_commits(commits: Iterable[dict[str, Any]]) -> dict[tuple[int, str, str], list[dict[str, Any]]]:
    histories: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for commit in commits:
        for event in commit["events"]:
            key = (int(event["replica_id"]), str(event["model_alias"]), str(event["task_id"]))
            histories[key].append(
                {
                    "step": int(event["step"]),
                    "thought": str(event.get("thought", "")),
                    "answer": str(event.get("answer", "")),
                    "answer_normalized": str(event.get("answer_normalized", "")),
                    "confidence": int(event.get("confidence", 50)),
                    "model_stop_flag": int(event.get("model_stop_flag", 0)),
                }
            )
    for key, history in histories.items():
        history.sort(key=lambda row: int(row["step"]))
        steps = [int(row["step"]) for row in history]
        if steps != list(range(1, len(steps) + 1)):
            raise ProtocolError(f"Non-contiguous own-history steps for replica/model/task key {key}.")
    return histories


def batched(items: list[TaskSpec], size: int) -> Iterator[list[TaskSpec]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def make_event_row(
    *,
    protocol_id: str,
    manifest_sha: str,
    protocol_manifest_file_sha: str,
    task: TaskSpec,
    replica_id: int,
    step: int,
    model_spec: ModelSpec,
    model_revision: str,
    barrier: str,
    seed: int,
    seed_scope: str,
    effective_rng_seed: int,
    batch_id: str,
    batch_index: int,
    batch_size: int,
    started_at_utc: str,
    ended_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    prompt_tokens: int,
    generated: dict[str, Any],
    prompt_mode: str,
) -> dict[str, Any]:
    parsed = parse_generation(str(generated["raw_text"]), task.answer_type, prompt_mode)
    answer = str(parsed["answer"])
    normalized = normalize_answer(answer, task.answer_type)
    completion_tokens = int(generated["generated_tokens"])
    row = json_safe(
        {
            "event_id": event_id(barrier, model_spec.alias),
            "barrier_id": barrier,
            "protocol_id": protocol_id,
            "protocol_manifest_sha256": protocol_manifest_file_sha,
            "manifest_sha256": manifest_sha,
            "generation_kind": "main",
            "task_id": task.task_id,
            "task_prompt_sha256": prompt_hash(task.prompt),
            "task_hash": prompt_hash(task.prompt),
            "domain": task.domain,
            "difficulty": task.difficulty,
            "answer_type": task.answer_type,
            "replica_id": int(replica_id),
            "step": int(step),
            "model_alias": model_spec.alias,
            "model_name": model_spec.hf_name,
            "model_revision": model_revision,
            "event_seed": int(seed),
            "seed_scope": seed_scope,
            "effective_rng_seed": int(effective_rng_seed),
            "batch_id": batch_id,
            "batch_index": int(batch_index),
            "batch_size": int(batch_size),
            "started_at_utc": started_at_utc,
            "ended_at_utc": ended_at_utc,
            "started_monotonic_ns": int(started_monotonic_ns),
            "ended_monotonic_ns": int(ended_monotonic_ns),
            "wall_clock_seconds": (ended_monotonic_ns - started_monotonic_ns) / 1_000_000_000.0,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": completion_tokens,
            "total_tokens": int(prompt_tokens + completion_tokens),
            "thought": str(parsed["thought"]),
            "answer": answer,
            "answer_normalized": normalized,
            "confidence": int(max(0, min(100, int(parsed["confidence"])))),
            "model_stop_flag": int(bool(parsed["stop"])),
            "parse_success": int(parsed["parse_success"]),
            "output_format_type": str(parsed["output_format_type"]),
            "answer_extraction_source": str(parsed["answer_extraction_source"]),
            "raw_text_sha256": sha256_text(str(generated["raw_text"])),
            "raw_text_length_chars": int(len(str(generated["raw_text"]))),
            "mean_token_logprob": generated.get("mean_token_logprob"),
            "entropy_mean": generated.get("mean_entropy"),
            "entropy_std": generated.get("entropy_std"),
            "answer_span_mean_logprob": generated.get("answer_span_mean_logprob"),
            "answer_span_min_logprob": generated.get("answer_span_min_logprob"),
            "answer_span_mean_entropy": generated.get("answer_span_mean_entropy"),
            "answer_span_std_entropy": generated.get("answer_span_std_entropy"),
            "response_hidden_norm": float(np.linalg.norm(generated["pooled_hidden"])),
            # Extended projections are recorded verbatim only when the frozen
            # manifest opts in.  They are derived from the current event and
            # needed to reproduce the fold-local representation probe; they
            # are never labels or future-barrier values.
            "mid_hidden_1_proj": str(generated.get("mid_hidden_1_proj", "")),
            "mid_hidden_2_proj": str(generated.get("mid_hidden_2_proj", "")),
            "hidden_projection_contract": (
                "mid_layer_mean_pool_random_projection_v1_dim64_seed42"
                if bool(generated.get("mid_hidden_1_proj") or generated.get("mid_hidden_2_proj"))
                else None
            ),
        }
    )
    row["event_sha256"] = sha256_bytes(canonical_json_bytes(row))
    return row


def make_verifier_event_row(
    *,
    protocol_id: str,
    manifest_sha: str,
    protocol_manifest_file_sha: str,
    task: TaskSpec,
    replica_id: int,
    step: int,
    barrier: str,
    barrier_record: Mapping[str, Any],
    verifier_model: ModelSpec,
    verifier_revision: str,
    verifier_config: Mapping[str, Any],
    variant: str,
    seed: int,
    started_at_utc: str,
    ended_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    scores: Any,
    prompt: str,
    batch_id: str,
    batch_index: int,
    batch_size: int,
) -> dict[str, Any]:
    """Serialize one post-barrier forced-choice event without label access."""

    if task.domain != "gpqa" or task.answer_type != "mcq":
        raise ProtocolError("The forced-choice verifier is restricted to frozen GPQA A--D tasks.")
    closed_ns = int(barrier_record["closed_monotonic_ns"])
    if int(started_monotonic_ns) <= closed_ns:
        raise ProtocolError("Verifier event must start strictly after the completed main barrier.")
    score_dict = scores.as_dict()
    option_tokens = {str(key): int(value) for key, value in score_dict["option_scoring_tokens"].items()}
    prompt_tokens = int(score_dict["base_prompt_tokens"]) * len(OPTION_LABELS)
    completion_tokens = int(sum(option_tokens.values()))
    row = json_safe(
        {
            "event_id": event_id(barrier, verifier_model.alias, "verifier", variant),
            "barrier_id": barrier,
            "protocol_id": protocol_id,
            "protocol_manifest_sha256": protocol_manifest_file_sha,
            "manifest_sha256": manifest_sha,
            "generation_kind": "verifier",
            "task_id": task.task_id,
            "task_prompt_sha256": prompt_hash(task.prompt),
            "task_hash": prompt_hash(task.prompt),
            "domain": task.domain,
            "difficulty": task.difficulty,
            "answer_type": task.answer_type,
            "replica_id": int(replica_id),
            "step": int(step),
            "model_alias": verifier_model.alias,
            "model_name": verifier_model.hf_name,
            "model_revision": verifier_revision,
            "event_seed": int(seed),
            "seed_scope": "per_event",
            "effective_rng_seed": int(seed),
            "batch_id": batch_id,
            "batch_index": int(batch_index),
            "batch_size": int(batch_size),
            "started_at_utc": started_at_utc,
            "ended_at_utc": ended_at_utc,
            "started_monotonic_ns": int(started_monotonic_ns),
            "ended_monotonic_ns": int(ended_monotonic_ns),
            "wall_clock_seconds": (ended_monotonic_ns - started_monotonic_ns) / 1_000_000_000.0,
            # Four independent continuation scores each consume the frozen
            # prompt.  This deliberately charges the full verifier cost.
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(prompt_tokens + completion_tokens),
            "verifier_spec_id": str(verifier_config["spec_id"]),
            "verifier_variant": variant,
            "post_barrier_stage": str(verifier_config["post_barrier_stage"]),
            "verifier_prompt_sha256": sha256_text(prompt),
            "verifier_template_sha256": str(verifier_config["template_sha256"][variant]),
            "verifier_option_labels": list(OPTION_LABELS),
            "verifier_option_logprobs": score_dict["option_logprobs"],
            "verifier_option_posteriors": score_dict["option_posteriors"],
            "verifier_argmax_option": score_dict["argmax_option"],
            "verifier_top1_margin": score_dict["top1_margin"],
            "verifier_entropy": score_dict["entropy"],
            "verifier_base_prompt_tokens": int(score_dict["base_prompt_tokens"]),
            "verifier_option_scoring_tokens": option_tokens,
            "verifier_score_method": score_dict["score_method"],
            "verifier_token_accounting_contract": score_dict["token_accounting_contract"],
            "triggered_by_decision_id": None,
        }
    )
    row["event_sha256"] = sha256_bytes(canonical_json_bytes(row))
    return row


def generate_verifier_events(
    *,
    args: argparse.Namespace,
    protocol_id: str,
    manifest_sha: str,
    protocol_manifest_file_sha: str,
    replica_id: int,
    step: int,
    verifier_model: ModelSpec,
    verifier_revision: str,
    verifier_config: Mapping[str, Any],
    tasks: list[TaskSpec],
    main_events_by_barrier: Mapping[str, list[dict[str, Any]]],
    barrier_records: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Score post-barrier GPQA options with no labels or future barriers.

    Exact continuation scoring is deterministic, so unlike stochastic main
    generation it does not claim a shared batch RNG.  The current core scores
    one prompt at a time; every four-option microbatch remains explicitly
    accounted for in its event token ledger.
    """

    if not tasks:
        return {}, "not-run"
    if any(task.domain != "gpqa" or task.answer_type != "mcq" for task in tasks):
        raise ProtocolError("--verifier-mode=collect_only currently requires a GPQA multiple-choice task manifest.")
    offload_dir = output_dir / "model_offload" / verifier_model.alias
    offload_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, actual_device, backend = load_model(
        model_spec=verifier_model,
        device=args.device,
        quantization=args.quantization,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        offload_folder=str(offload_dir),
        model_revision=verifier_revision if verifier_revision != "UNPINNED" else None,
    )
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    try:
        for variant in verifier_config["variants"]:
            for batch_number, task_batch in enumerate(batched(tasks, int(args.verifier_batch_size)), start=1):
                prompts: list[str] = []
                barrier_batch: list[str] = []
                barrier_batch_records: list[Mapping[str, Any]] = []
                for task in task_batch:
                    barrier = barrier_id(protocol_id, prompt_hash(task.prompt), replica_id, step)
                    main_events = main_events_by_barrier.get(barrier)
                    barrier_record = barrier_records.get(barrier)
                    if main_events is None or barrier_record is None:
                        raise ProtocolError(f"Verifier cannot find the closed main barrier for task {task.task_id!r}.")
                    thoughts: Sequence[str] | None = None
                    seed_material = None
                    if variant == ANONYMOUS_RATIONALE_V1:
                        thoughts = [str(event.get("thought", "")) for event in main_events]
                        seed_material = "|".join(
                            [protocol_id, prompt_hash(task.prompt), str(replica_id), str(step), verifier_model.alias, variant]
                        )
                    prompts.append(
                        render_verifier_prompt(
                            task.prompt,
                            variant=str(variant),
                            thoughts=thoughts,
                            seed_material=seed_material,
                            rationale_max_chars=int(verifier_config["rationale_max_chars"]),
                        )
                    )
                    barrier_batch.append(barrier)
                    barrier_batch_records.append(barrier_record)
                started_at_utc = utc_strictly_after([str(record["closed_at_utc"]) for record in barrier_batch_records])
                started_monotonic_ns = max(
                    time.perf_counter_ns(),
                    max(int(record["closed_monotonic_ns"]) for record in barrier_batch_records) + 1,
                )
                scores_batch = score_forced_choice_prompt_batch(
                    model,
                    tokenizer,
                    prompts,
                    device=actual_device,
                    continuations=verifier_config["continuations"],
                    sequence_microbatch_size=max(4, int(args.verifier_batch_size) * len(OPTION_LABELS)),
                )
                ended_at_utc = utc_strictly_after([started_at_utc])
                ended_monotonic_ns = max(time.perf_counter_ns(), started_monotonic_ns + 1)
                batch_id = sha256_text(
                    f"{protocol_id}|{replica_id}|{step}|{verifier_model.alias}|{variant}|{batch_number}|{'|'.join(barrier_batch)}"
                )[:32]
                for batch_index, (task, barrier, barrier_record, prompt, scores) in enumerate(
                    zip(task_batch, barrier_batch, barrier_batch_records, prompts, scores_batch), start=1
                ):
                    seed = derived_seed(
                        protocol_id,
                        prompt_hash(task.prompt),
                        replica_id,
                        step,
                        verifier_model.alias,
                        f"verifier:{variant}",
                    )
                    rows[barrier].append(
                        make_verifier_event_row(
                            protocol_id=protocol_id,
                            manifest_sha=manifest_sha,
                            protocol_manifest_file_sha=protocol_manifest_file_sha,
                            task=task,
                            replica_id=replica_id,
                            step=step,
                            barrier=barrier,
                            barrier_record=barrier_record,
                            verifier_model=verifier_model,
                            verifier_revision=verifier_revision,
                            verifier_config=verifier_config,
                            variant=str(variant),
                            seed=seed,
                            started_at_utc=started_at_utc,
                            ended_at_utc=ended_at_utc,
                            started_monotonic_ns=started_monotonic_ns,
                            ended_monotonic_ns=ended_monotonic_ns,
                            scores=scores,
                            prompt=prompt,
                            batch_id=batch_id,
                            batch_index=batch_index,
                            batch_size=len(task_batch),
                        )
                    )
    finally:
        del model
        del tokenizer
        gc.collect()
        release_cuda_memory()
    return rows, backend


def generate_model_events(
    *,
    args: argparse.Namespace,
    protocol_id: str,
    manifest_sha: str,
    protocol_manifest_file_sha: str,
    replica_id: int,
    step: int,
    model_spec: ModelSpec,
    model_revision: str,
    tasks: list[TaskSpec],
    histories: dict[tuple[int, str, str], list[dict[str, Any]]],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    """Generate one main event per task for an alias at a single live barrier.

    ``per_event`` mode is intentionally singleton generation.  Hugging Face's
    standard batched sampler has one global RNG stream, so claiming per-event
    independent seeds while batching would be false.  ``batch_seeded`` exists
    only for non-confirmatory throughput profiling and is visibly labelled in
    every event and manifest.
    """
    offload_dir = output_dir / "model_offload" / model_spec.alias
    offload_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, actual_device, backend = load_model(
        model_spec=model_spec,
        device=args.device,
        quantization=args.quantization,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        offload_folder=str(offload_dir),
        model_revision=model_revision if model_revision != "UNPINNED" else None,
    )
    rows: list[dict[str, Any]] = []
    try:
        requested_batch_size = 1 if args.seed_mode == "per_event" else max(1, int(args.batch_size))
        for batch_number, task_batch in enumerate(batched(tasks, requested_batch_size), start=1):
            prompts: list[str] = []
            seeds: list[int] = []
            barriers: list[str] = []
            for task in task_batch:
                history = histories[(replica_id, model_spec.alias, task.task_id)]
                observed_steps = [int(item["step"]) for item in history]
                if observed_steps != list(range(1, step)):
                    raise ProtocolError(
                        f"Cannot generate step {step}: own history for {model_spec.alias}/{task.task_id} is "
                        f"{observed_steps}, not {list(range(1, step))}."
                    )
                prompts.append(
                    render_prompt(
                        tokenizer,
                        conversation_prompt(
                            task=task,
                            history=history,
                            step=step,
                            max_steps=args.max_steps,
                            prompt_mode=args.prompt_mode,
                        ),
                        args.system_prompt_mode,
                    )
                )
                seeds.append(
                    derived_seed(
                        protocol_id,
                        prompt_hash(task.prompt),
                        replica_id,
                        step,
                        model_spec.alias,
                        "main",
                    )
                )
                barriers.append(barrier_id(protocol_id, prompt_hash(task.prompt), replica_id, step))

            if args.seed_mode == "per_event":
                # The batch necessarily has one task. set_seed touches Python,
                # NumPy, CPU Torch, and CUDA Torch RNGs, so the seed in the
                # ledger is exactly the RNG seed that produced this event.
                effective_rng_seed = seeds[0]
                set_seed(effective_rng_seed)
                seed_scope = "per_event"
            else:
                # Keep this mode useful for performance profiling without
                # allowing it to masquerade as independent-rollout evidence.
                effective_rng_seed = int.from_bytes(
                    hashlib.sha256("|".join(str(value) for value in seeds).encode("utf-8")).digest()[:4],
                    "big",
                )
                set_seed(effective_rng_seed)
                seed_scope = "batch_seeded_nonconfirmatory"

            batch_identifier = sha256_text(
                f"{protocol_id}|{replica_id}|{step}|{model_spec.alias}|{batch_number}|{'|'.join(barriers)}"
            )[:32]
            started_at_utc = utc_now()
            started_monotonic_ns = time.perf_counter_ns()
            generated_rows, _batch_metrics = safe_generate_batch_with_diagnostics(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=prompts,
                actual_device=actual_device,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                answer_types=[task.answer_type for task in task_batch],
                prompt_mode=args.prompt_mode,
                enable_extended_observables=bool(args.extended_observables),
            )
            ended_monotonic_ns = time.perf_counter_ns()
            ended_at_utc = utc_now()
            if len(generated_rows) != len(task_batch):
                raise ProtocolError(
                    f"Generation returned {len(generated_rows)} rows for a requested batch of {len(task_batch)}."
                )
            for index, (task, prompt, event_seed_value, barrier, generated) in enumerate(
                zip(task_batch, prompts, seeds, barriers, generated_rows, strict=True)
            ):
                row = make_event_row(
                    protocol_id=protocol_id,
                    manifest_sha=manifest_sha,
                    protocol_manifest_file_sha=protocol_manifest_file_sha,
                    task=task,
                    replica_id=replica_id,
                    step=step,
                    model_spec=model_spec,
                    model_revision=model_revision,
                    barrier=barrier,
                    seed=event_seed_value,
                    seed_scope=seed_scope,
                    effective_rng_seed=effective_rng_seed,
                    batch_id=batch_identifier,
                    batch_index=index,
                    batch_size=len(task_batch),
                    started_at_utc=started_at_utc,
                    ended_at_utc=ended_at_utc,
                    started_monotonic_ns=started_monotonic_ns,
                    ended_monotonic_ns=ended_monotonic_ns,
                    prompt_tokens=task_prompt_token_count(tokenizer, prompt),
                    generated=generated,
                    prompt_mode=args.prompt_mode,
                )
                rows.append(row)
                histories[(replica_id, model_spec.alias, task.task_id)].append(
                    {
                        "step": step,
                        "thought": row["thought"],
                        "answer": row["answer"],
                        "answer_normalized": row["answer_normalized"],
                        "confidence": row["confidence"],
                        "model_stop_flag": row["model_stop_flag"],
                    }
                )
    finally:
        del model
        del tokenizer
        gc.collect()
        release_cuda_memory()
    return rows, backend


def select_fixed_policy(
    events: list[dict[str, Any]],
    roster: tuple[str, ...],
    policy: dict[str, Any],
) -> dict[str, Any]:
    by_alias = {str(event["model_alias"]): event for event in events}
    if tuple(by_alias) != roster or len(by_alias) != len(roster):
        raise ProtocolError("Policy selection received an incomplete or unordered roster barrier.")
    answer_types = {str(event.get("answer_type", "")) for event in events}
    if len(answer_types) != 1:
        raise ProtocolError("Policy selection received a barrier with inconsistent answer types.")
    answer_type = next(iter(answer_types))
    leader = by_alias[str(policy["leader_alias"])]
    legal_options = {str(value) for value in policy.get("legal_mcq_options") or []}
    restrict_to_legal = policy["policy_type"] == "legal_consensus" and answer_type == "mcq"
    legal_answer_count = 0
    invalid_answer_count = 0
    selection_fallback = False
    if policy["policy_type"] == "leader":
        selected = leader
        selected_support = sum(
            int(event["answer_normalized"] == selected["answer_normalized"] and bool(selected["answer_normalized"]))
            for event in events
        )
        vote_count = selected_support
        valid_vote_count = sum(int(bool(event["answer_normalized"])) for event in events)
    else:
        votes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            answer = str(event["answer_normalized"])
            if answer and restrict_to_legal:
                if answer in legal_options:
                    legal_answer_count += 1
                else:
                    invalid_answer_count += 1
                    continue
            if answer:
                votes[answer].append(event)
        valid_vote_count = sum(len(value) for value in votes.values())
        if votes:
            ranked = sorted(
                votes.items(),
                key=lambda item: (-len(item[1]), min(roster.index(str(event["model_alias"])) for event in item[1])),
            )
            answer, supporters = ranked[0]
            selected = min(supporters, key=lambda event: roster.index(str(event["model_alias"])))
            vote_count = len(supporters)
            selected_support = vote_count
        else:
            selected = leader
            vote_count = 0
            selected_support = 0
            selection_fallback = True
    decision_created_monotonic_ns = time.perf_counter_ns()
    consumed_event_ids = [str(event["event_id"]) for event in events]
    consumed_event_hashes = [str(event["event_sha256"]) for event in events]
    decision = {
        "decision_id": sha256_text(f"{events[0]['barrier_id']}|decision")[:32],
        "barrier_id": events[0]["barrier_id"],
        "protocol_id": events[0]["protocol_id"],
        "protocol_manifest_sha256": events[0]["protocol_manifest_sha256"],
        "manifest_sha256": events[0]["manifest_sha256"],
        "task_id": events[0]["task_id"],
        "task_prompt_sha256": events[0]["task_prompt_sha256"],
        "task_hash": events[0]["task_hash"],
        "domain": events[0]["domain"],
        "replica_id": int(events[0]["replica_id"]),
        "step": int(events[0]["step"]),
        "policy_type": policy["policy_type"],
        "policy_sha256": sha256_bytes(canonical_json_bytes(policy)),
        "selected_model_alias": selected["model_alias"],
        "selected_event_id": selected["event_id"],
        "selected_answer": selected["answer"],
        "selected_answer_normalized": selected["answer_normalized"],
        "selected_answer_hash": sha256_text(str(selected["answer_normalized"])),
        "selected_parse_success": int(selected["parse_success"]),
        "selected_confidence": int(selected["confidence"]),
        "selected_vote_count": int(vote_count),
        "selected_support_fraction": float(selected_support / len(roster)),
        "valid_answer_count": int(valid_vote_count),
        "legal_answer_count": int(legal_answer_count) if restrict_to_legal else None,
        "invalid_answer_count": int(invalid_answer_count) if restrict_to_legal else None,
        "selection_fallback": bool(selection_fallback),
        "fleet_size": int(len(roster)),
        "peer_visibility": "post_complete_barrier_only",
        "decision_created_at_utc": utc_now(),
        "decision_created_monotonic_ns": int(decision_created_monotonic_ns),
        "consumed_event_ids": consumed_event_ids,
        "consumed_event_sha256s": consumed_event_hashes,
        "consumed_event_count": len(events),
        "consumed_prompt_tokens": int(sum(int(event["prompt_tokens"]) for event in events)),
        "consumed_completion_tokens": int(sum(int(event["completion_tokens"]) for event in events)),
        "consumed_total_tokens": int(sum(int(event["total_tokens"]) for event in events)),
        "feature_contract": list(policy["feature_contracts"][str(policy["feature_contract_id"])]),
        "feature_contract_id": str(policy["feature_contract_id"]),
        "feature_contract_sha256": sha256_bytes(
            canonical_json_bytes(policy["feature_contracts"][str(policy["feature_contract_id"])])
        ),
        "selection_rule_id": policy["selection_rule_id"],
        # A collection run intentionally continues through the frozen horizon.
        # It still records a terminal, explicit action at every barrier so an
        # auditor cannot mistake an absent decision for an online decision.
        "decision_stage": "final",
        "action": "continue",
        "decision_at_utc": utc_now(),
        "decision_monotonic_ns": int(decision_created_monotonic_ns),
        "barrier_complete": True,
        "selection_action": "select_answer",
        "stopping_action": "continue_collection",
        "stopping_score": None,
    }
    decision["decision_sha256"] = sha256_bytes(canonical_json_bytes(decision))
    return decision


def prepare_closed_main_barrier(
    *,
    protocol_id: str,
    manifest_sha: str,
    protocol_manifest_file_sha: str,
    task: TaskSpec,
    replica_id: int,
    step: int,
    events: list[dict[str, Any]],
    roster: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Close a complete main panel before any post-barrier verifier runs."""

    barrier = barrier_id(protocol_id, prompt_hash(task.prompt), replica_id, step)
    if len(events) != len(roster):
        raise ProtocolError(f"Cannot close barrier {barrier}: expected {len(roster)} events, received {len(events)}.")
    ordered = sorted(events, key=lambda row: roster.index(str(row["model_alias"])))
    if tuple(str(row["model_alias"]) for row in ordered) != roster:
        raise ProtocolError(f"Barrier event ordering/roster mismatch for {barrier}.")
    if any(str(row["barrier_id"]) != barrier or str(row.get("generation_kind")) != "main" for row in ordered):
        raise ProtocolError(f"Cannot close barrier {barrier}: its main event scope or kind is invalid.")
    closed_at_utc = utc_strictly_after([str(row["ended_at_utc"]) for row in ordered])
    closed_monotonic_ns = max(time.perf_counter_ns(), max(int(row["ended_monotonic_ns"]) for row in ordered) + 1)
    token_totals = {
        "prompt_tokens": int(sum(int(row["prompt_tokens"]) for row in ordered)),
        "completion_tokens": int(sum(int(row["completion_tokens"]) for row in ordered)),
        "total_tokens": int(sum(int(row["total_tokens"]) for row in ordered)),
    }
    barrier_record = {
        "barrier_id": barrier,
        "protocol_id": protocol_id,
        "protocol_manifest_sha256": protocol_manifest_file_sha,
        "manifest_sha256": manifest_sha,
        "task_id": task.task_id,
        "task_prompt_sha256": prompt_hash(task.prompt),
        "task_hash": prompt_hash(task.prompt),
        "domain": task.domain,
        "replica_id": int(replica_id),
        "step": int(step),
        "roster": list(roster),
        "roster_size": len(roster),
        "main_event_ids": [str(row["event_id"]) for row in ordered],
        "expected_aliases": list(roster),
        "completed_aliases": list(roster),
        "event_hashes": {str(row["event_id"]): str(row["event_sha256"]) for row in ordered},
        "complete_roster": True,
        "barrier_complete": True,
        "status": "complete",
        "peer_visibility": "post_complete_barrier_only",
        "opened_at_utc": min(str(row["started_at_utc"]) for row in ordered),
        "closed_at_utc": closed_at_utc,
        "opened_monotonic_ns": int(min(int(row["started_monotonic_ns"]) for row in ordered)),
        "closed_monotonic_ns": int(closed_monotonic_ns),
        **token_totals,
        "fleet_prompt_tokens": token_totals["prompt_tokens"],
        "fleet_completion_tokens": token_totals["completion_tokens"],
        "fleet_total_tokens": token_totals["total_tokens"],
    }
    barrier_record["barrier_sha256"] = sha256_bytes(canonical_json_bytes(barrier_record))
    return ordered, barrier_record


def write_barrier_commit(
    *,
    commits_dir: Path,
    protocol_id: str,
    manifest_sha: str,
    protocol_manifest_file_sha: str,
    task: TaskSpec,
    replica_id: int,
    step: int,
    events: list[dict[str, Any]],
    roster: tuple[str, ...],
    policy: dict[str, Any],
    schema_version: str,
    barrier_record: dict[str, Any] | None = None,
    post_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if barrier_record is None:
        ordered, barrier_record = prepare_closed_main_barrier(
            protocol_id=protocol_id,
            manifest_sha=manifest_sha,
            protocol_manifest_file_sha=protocol_manifest_file_sha,
            task=task,
            replica_id=replica_id,
            step=step,
            events=events,
            roster=roster,
        )
    else:
        if len(events) != len(roster):
            raise ProtocolError("Prepared barrier commit has an incomplete main roster.")
        ordered = sorted(events, key=lambda row: roster.index(str(row["model_alias"])))
        expected_ids = [str(row["event_id"]) for row in ordered]
        if (
            tuple(str(row["model_alias"]) for row in ordered) != roster
            or str(barrier_record.get("task_id")) != task.task_id
            or int(barrier_record.get("replica_id", -1)) != int(replica_id)
            or int(barrier_record.get("step", -1)) != int(step)
            or list(barrier_record.get("main_event_ids", [])) != expected_ids
        ):
            raise ProtocolError("Prepared barrier record no longer matches its frozen main event panel.")
    barrier = str(barrier_record["barrier_id"])
    post_events = [] if post_events is None else list(post_events)
    post_ids: set[str] = set()
    for post_event in post_events:
        if not isinstance(post_event, dict) or str(post_event.get("generation_kind")) != "verifier":
            raise ProtocolError("post_events must contain only verifier event objects.")
        if str(post_event.get("barrier_id")) != barrier:
            raise ProtocolError("A verifier event belongs to a different main barrier.")
        identifier = str(post_event.get("event_id", ""))
        if not identifier or identifier in post_ids:
            raise ProtocolError("Verifier post-events must have unique non-empty event IDs per barrier.")
        post_ids.add(identifier)
        if int(post_event.get("started_monotonic_ns", -1)) <= int(barrier_record["closed_monotonic_ns"]):
            raise ProtocolError("A verifier post-event precedes main-barrier closure.")
    # This observational collector intentionally keeps the main selection
    # unchanged while recording verifier evidence.  An override policy can only
    # be frozen after a separate development/calibration stage succeeds.
    decision = select_fixed_policy(ordered, roster, policy)
    payload = {
        "schema_version": schema_version,
        "protocol_id": protocol_id,
        "protocol_manifest_sha256": protocol_manifest_file_sha,
        "manifest_sha256": manifest_sha,
        "barrier_id": barrier,
        "barrier": barrier_record,
        "events": ordered,
        "event_hashes": {str(row["event_id"]): str(row["event_sha256"]) for row in ordered},
        "post_events": post_events,
        "post_event_hashes": {str(row["event_id"]): str(row["event_sha256"]) for row in post_events},
        "decision": decision,
    }
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    path = commits_dir / f"{barrier}.json"
    if path.exists():
        existing = load_json(path)
        if existing.get("payload_sha256") != payload["payload_sha256"]:
            raise ProtocolError(
                f"Refusing to replace existing barrier {barrier}; its immutable payload differs from this run."
            )
        return existing
    atomic_write_json(path, payload)
    return payload


def rebuild_derived_ledgers(
    output_dir: Path,
    commits: dict[str, dict[str, Any]],
    *,
    schema_version: str,
) -> dict[str, Any]:
    ordered = sorted(
        commits.values(),
        key=lambda commit: (
            int(commit["barrier"]["replica_id"]),
            int(commit["barrier"]["step"]),
            str(commit["barrier"]["task_id"]),
        ),
    )
    events = [
        event
        for commit in ordered
        for event in [*commit["events"], *list(commit.get("post_events", []))]
    ]
    barriers = [commit["barrier"] for commit in ordered]
    decisions = [commit["decision"] for commit in ordered]
    events_path = output_dir / "events.jsonl"
    barriers_path = output_dir / "barriers.jsonl"
    decisions_path = output_dir / "decisions.jsonl"
    atomic_write_jsonl(events_path, events)
    atomic_write_jsonl(barriers_path, barriers)
    atomic_write_jsonl(decisions_path, decisions)
    ledger_manifest = {
        "schema_version": schema_version,
        "rebuilt_at_utc": utc_now(),
        "barrier_count": len(barriers),
        "event_count": len(events),
        "decision_count": len(decisions),
        "events_sha256": sha256_file(events_path),
        "barriers_sha256": sha256_file(barriers_path),
        "decisions_sha256": sha256_file(decisions_path),
    }
    atomic_write_json(output_dir / "ledger_manifest.json", ledger_manifest)
    return ledger_manifest


def write_status(
    output_dir: Path,
    *,
    manifest_sha: str,
    protocol_manifest_file_sha: str,
    expected_barriers: int,
    commits: dict[str, dict[str, Any]],
    schema_version: str,
    state: str,
    error: str | None = None,
) -> None:
    main_events = [event for commit in commits.values() for event in commit["events"]]
    verifier_events = [event for commit in commits.values() for event in commit.get("post_events", [])]
    events = [*main_events, *verifier_events]
    payload = {
        "schema_version": schema_version,
        "updated_at_utc": utc_now(),
        "protocol_manifest_sha256": protocol_manifest_file_sha,
        "manifest_sha256": manifest_sha,
        "state": state,
        "expected_barriers": int(expected_barriers),
        "completed_barriers": int(len(commits)),
        "pending_barriers": int(max(expected_barriers - len(commits), 0)),
        "completed_events": int(len(events)),
        "completed_main_events": int(len(main_events)),
        "completed_verifier_events": int(len(verifier_events)),
        "fleet_prompt_tokens": int(sum(int(event["prompt_tokens"]) for event in events)),
        "fleet_completion_tokens": int(sum(int(event["completion_tokens"]) for event in events)),
        "fleet_total_tokens": int(sum(int(event["total_tokens"]) for event in events)),
        "error": error,
    }
    atomic_write_json(output_dir / STATUS_NAME, payload)


def collect(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roster = normalize_roster(args.roster)
    if len(set(args.replicas)) != len(args.replicas) or any(int(value) < 0 for value in args.replicas):
        raise ProtocolError("Replica IDs must be unique non-negative integers.")
    if args.max_steps < 1 or args.max_new_tokens < 1:
        raise ProtocolError("--max-steps and --max-new-tokens must both be positive.")
    if args.temperature < 0:
        raise ProtocolError("--temperature cannot be negative.")
    if args.seed_mode == "batch_seeded" and args.phase == "confirmatory":
        raise ProtocolError("batch_seeded mode cannot be used for a confirmatory protocol.")

    tasks, _public_rows, task_manifest_sha, labels_sha = load_frozen_tasks(output_dir)
    verifier_model = verifier_model_spec_from_args(args)
    verifier_revision = load_verifier_revision(args, verifier_model)
    verifier_config = verifier_spec_from_args(args, verifier_model)
    if verifier_config is not None and any(task.domain != "gpqa" or task.answer_type != "mcq" for task in tasks):
        raise ProtocolError("--verifier-mode=collect_only requires a frozen GPQA manifest with A--D multiple-choice tasks only.")
    schema_version = schema_version_for_args(args)
    revisions = load_model_revisions(args.model_revisions, roster, args.allow_unpinned_models)
    runtime = preflight(args)
    manifest, manifest_sha, protocol_manifest_file_sha = create_or_validate_manifest(
        args,
        output_dir,
        roster,
        revisions,
        task_manifest_sha,
        labels_sha,
        len(tasks),
        runtime,
        verifier_record(verifier_model, verifier_revision),
        verifier_config,
    )
    commits_dir = output_dir / COMMITS_DIR_NAME
    commits_dir.mkdir(parents=True, exist_ok=True)
    commits = existing_commits(
        commits_dir,
        manifest_sha,
        protocol_manifest_file_sha,
        roster,
        schema_version,
        verifier_config,
    )
    expected_barriers = len(tasks) * len(args.replicas) * args.max_steps
    if commits and not args.resume:
        raise ProtocolError("Committed barriers already exist; use --resume or a new output directory.")
    histories = histories_from_commits(commits.values())
    policy = manifest["policy"]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol_id": args.protocol_id,
                    "task_count": len(tasks),
                    "roster_size": len(roster),
                    "replicas": args.replicas,
                    "max_steps": args.max_steps,
                    "expected_barriers": expected_barriers,
                    "expected_main_events": expected_barriers * len(roster),
                    "expected_verifier_events": expected_barriers * len(verifier_config["variants"]) if verifier_config else 0,
                    "completed_barriers": len(commits),
                    "confirmation_eligible": manifest["confirmation_eligible"],
                    "runtime": runtime,
                },
                indent=2,
            ),
            flush=True,
        )
        return

    try:
        write_status(
            output_dir,
            manifest_sha=manifest_sha,
            protocol_manifest_file_sha=protocol_manifest_file_sha,
            expected_barriers=expected_barriers,
            commits=commits,
            schema_version=schema_version,
            state="running",
        )
        for replica_id in args.replicas:
            for step in range(1, args.max_steps + 1):
                pending_tasks = [
                    task
                    for task in tasks
                    if barrier_id(args.protocol_id, prompt_hash(task.prompt), replica_id, step) not in commits
                ]
                if not pending_tasks:
                    continue
                events_by_barrier: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for alias in roster:
                    spec = MODEL_CATALOG[alias]
                    print(
                        f"Collecting replica={replica_id} step={step}/{args.max_steps} model={alias} "
                        f"tasks={len(pending_tasks)}",
                        flush=True,
                    )
                    model_events, backend = generate_model_events(
                        args=args,
                        protocol_id=args.protocol_id,
                        manifest_sha=manifest_sha,
                        protocol_manifest_file_sha=protocol_manifest_file_sha,
                        replica_id=replica_id,
                        step=step,
                        model_spec=spec,
                        model_revision=revisions[alias],
                        tasks=pending_tasks,
                        histories=histories,
                        output_dir=output_dir,
                    )
                    if len(model_events) != len(pending_tasks):
                        raise ProtocolError(f"Model {alias} did not produce all pending events.")
                    for row in model_events:
                        events_by_barrier[str(row["barrier_id"])].append(row)
                prepared_main_events: dict[str, list[dict[str, Any]]] = {}
                prepared_barriers: dict[str, dict[str, Any]] = {}
                for task in pending_tasks:
                    identifier = barrier_id(args.protocol_id, prompt_hash(task.prompt), replica_id, step)
                    ordered_main, closed_barrier = prepare_closed_main_barrier(
                        protocol_id=args.protocol_id,
                        manifest_sha=manifest_sha,
                        protocol_manifest_file_sha=protocol_manifest_file_sha,
                        task=task,
                        replica_id=replica_id,
                        step=step,
                        events=events_by_barrier[identifier],
                        roster=roster,
                    )
                    prepared_main_events[identifier] = ordered_main
                    prepared_barriers[identifier] = closed_barrier
                verifier_events_by_barrier: dict[str, list[dict[str, Any]]] = {}
                verifier_backend = "not-run"
                if verifier_model is not None and verifier_revision is not None and verifier_config is not None:
                    print(
                        f"Collecting mandatory post-barrier verifier variants={verifier_config['variants']} "
                        f"tasks={len(pending_tasks)}",
                        flush=True,
                    )
                    verifier_events_by_barrier, verifier_backend = generate_verifier_events(
                        args=args,
                        protocol_id=args.protocol_id,
                        manifest_sha=manifest_sha,
                        protocol_manifest_file_sha=protocol_manifest_file_sha,
                        replica_id=replica_id,
                        step=step,
                        verifier_model=verifier_model,
                        verifier_revision=verifier_revision,
                        verifier_config=verifier_config,
                        tasks=pending_tasks,
                        main_events_by_barrier=prepared_main_events,
                        barrier_records=prepared_barriers,
                        output_dir=output_dir,
                    )
                for task in pending_tasks:
                    identifier = barrier_id(args.protocol_id, prompt_hash(task.prompt), replica_id, step)
                    commit = write_barrier_commit(
                        commits_dir=commits_dir,
                        protocol_id=args.protocol_id,
                        manifest_sha=manifest_sha,
                        protocol_manifest_file_sha=protocol_manifest_file_sha,
                        task=task,
                        replica_id=replica_id,
                        step=step,
                        events=prepared_main_events[identifier],
                        roster=roster,
                        policy=policy,
                        schema_version=schema_version,
                        barrier_record=prepared_barriers[identifier],
                        post_events=verifier_events_by_barrier.get(identifier, []),
                    )
                    commits[identifier] = commit
                # The derived view is refreshed only after every canonical
                # commit for this live step is durable.
                rebuild_derived_ledgers(output_dir, commits, schema_version=schema_version)
                write_status(
                    output_dir,
                    manifest_sha=manifest_sha,
                    protocol_manifest_file_sha=protocol_manifest_file_sha,
                    expected_barriers=expected_barriers,
                    commits=commits,
                    schema_version=schema_version,
                    state="running",
                )
                print(
                    f"Closed {len(pending_tasks)} complete barriers for replica={replica_id}, step={step}; "
                    f"backend={backend}; verifier_backend={verifier_backend}",
                    flush=True,
                )
        rebuild_derived_ledgers(output_dir, commits, schema_version=schema_version)
        final_state = "complete" if len(commits) == expected_barriers else "incomplete"
        write_status(
            output_dir,
            manifest_sha=manifest_sha,
            protocol_manifest_file_sha=protocol_manifest_file_sha,
            expected_barriers=expected_barriers,
            commits=commits,
            schema_version=schema_version,
            state=final_state,
        )
        if final_state != "complete":
            raise ProtocolError(f"Collection ended with only {len(commits)}/{expected_barriers} barriers committed.")
        print(f"Prospective barrier collection complete: {output_dir}", flush=True)
    except Exception as exc:
        write_status(
            output_dir,
            manifest_sha=manifest_sha,
            protocol_manifest_file_sha=protocol_manifest_file_sha,
            expected_barriers=expected_barriers,
            commits=commits,
            schema_version=schema_version,
            state="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a fixed-roster, barrier-synchronous prospective reasoning panel."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--initialize", action="store_true", help="Write immutable public task and sealed-label manifests, then exit.")
    action.add_argument("--collect", action="store_true", help="Collect/resume the frozen protocol.")
    action.add_argument("--dry-run", action="store_true", help="Validate an existing frozen protocol without loading models.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol-id", default="prospective_committee_v1")
    parser.add_argument("--phase", choices=["development", "calibration", "confirmatory"], default="confirmatory")
    parser.add_argument("--task-source", choices=["builtin", "gsm8k", "math", "arc", "gpqa"], default="gsm8k")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--dataset-shuffle-seed", type=int, default=17)
    parser.add_argument("--dataset-revision", default=None, help="Immutable dataset revision SHA; required for --initialize.")
    parser.add_argument("--max-tasks", type=int, default=300)
    parser.add_argument("--historical-root", default=str(DEFAULT_HISTORICAL_ROOT))
    parser.add_argument("--allow-historical-overlap", action="store_true", help="Mark an initialization as non-task-disjoint replication.")
    parser.add_argument("--roster", nargs="+", default=list(DEFAULT_ROSTER))
    parser.add_argument("--model-revisions", default=None, help="JSON mapping frozen roster aliases to Hugging Face commit SHAs.")
    parser.add_argument("--allow-unpinned-models", action="store_true", help="Non-confirmatory smoke-test escape hatch.")
    parser.add_argument("--replicas", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed-mode", choices=["per_event", "batch_seeded"], default="per_event")
    parser.add_argument(
        "--policy",
        choices=["leader", "consensus", "legal_consensus"],
        default="consensus",
        help=(
            "Frozen answer selector. legal_consensus rejects post-generation non-A--D MCQ answers "
            "before plurality; it does not claim constrained decoding."
        ),
    )
    parser.add_argument("--leader", default="qwen2p5_32b")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit", "auto"], default="4bit")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--attn-implementation", choices=["auto", "sdpa", "flash_attention_2", "eager"], default="sdpa")
    parser.add_argument("--prompt-mode", choices=["structured_four_line", "minimal_json", "answer_only"], default="minimal_json")
    parser.add_argument("--system-prompt-mode", choices=["default", "short", "none"], default="default")
    parser.add_argument(
        "--verifier-mode",
        choices=["off", "collect_only"],
        default="off",
        help=(
            "Optional prospective GPQA forced-choice verifier. collect_only records separately timestamped "
            "post-barrier evidence but does not alter the frozen main selection policy."
        ),
    )
    parser.add_argument("--verifier-model", default="qwen2p5_32b", help="MODEL_CATALOG alias for the forced-choice verifier base checkpoint.")
    parser.add_argument(
        "--verifier-ledger-alias",
        default="gpqa_fc_qwen2p5_32b_v1",
        help="Distinct immutable ledger alias for verifier events (must not collide with the main roster).",
    )
    parser.add_argument(
        "--verifier-model-revision",
        default=None,
        help="Pinned 7-64 hex Hugging Face commit SHA for the verifier checkpoint.",
    )
    parser.add_argument(
        "--verifier-variants",
        nargs="+",
        choices=list(SUPPORTED_VERIFIER_VARIANTS),
        default=[PROMPT_OPTIONS_ONLY_V1, ANONYMOUS_RATIONALE_V1],
        help="Frozen verifier prompt variants to collect at every completed main barrier.",
    )
    parser.add_argument(
        "--verifier-active-variant",
        choices=list(SUPPORTED_VERIFIER_VARIANTS),
        default=PROMPT_OPTIONS_ONLY_V1,
        help="Pre-registered development variant; collect_only still records every requested variant.",
    )
    parser.add_argument(
        "--verifier-batch-size",
        type=int,
        default=4,
        help="Maximum four-option scoring microbatch size; token accounting remains exact per option.",
    )
    parser.add_argument(
        "--verifier-rationale-max-chars",
        type=int,
        default=768,
        help="Frozen per-rationale character cap for anonymous_rationale_v1.",
    )
    parser.add_argument(
        "--extended-observables",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Record answer-span diagnostics and the two 64-D hidden projections. "
            "Required to prospectively confirm the hidden-representation probe; increases scoring-pass memory."
        ),
    )
    parser.add_argument("--min-vram-gb", type=float, default=80.0)
    parser.add_argument("--allow-low-vram", action="store_true", help="Permit a non-confirmatory low-VRAM/CPU smoke test.")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.initialize:
            initialize_task_manifests(args)
        else:
            collect(args)
    except ProtocolError as exc:
        raise SystemExit(f"PROTOCOL ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
