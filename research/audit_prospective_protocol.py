#!/usr/bin/env python3
"""Read-only, fail-closed audit for prospective fixed-roster barrier experiments.

This tool is intentionally separate from the legacy trace collectors.  Historical
``trace_steps.csv`` files cannot prove live barrier availability: they lack the
per-event timestamps, frozen roster binding, and independently addressable token
ledger required for a deployable committee stopping claim.

Canonical artifact layout
=========================

The collector should create one immutable directory with these files::

    protocol_manifest.json
    prospective_tasks.jsonl
    sealed_gold_labels.jsonl             # opaque to this runtime auditor
    events.jsonl
    barriers.jsonl
    decisions.jsonl                 # required once manifest.status == "complete"

``protocol_manifest.json`` must contain a supported prospective schema
(``prospective-barrier-*`` or ``prospective-committee-*``)
``schema_version``, ``protocol_id``, frozen ``roster``, ``roster_sha256``, task
manifest descriptor, independent ``replicate_ids``, generation settings, a
feature contract, hardware provenance, and a terminal or in-progress status.
The roster hash is the SHA-256 of the canonical normalized roster documented by
``canonical_roster_hash`` below.  Event, barrier, and decision records bind to
the *canonical JSON SHA-256* of the manifest via ``manifest_sha256``.  The
committee collector additionally binds the raw immutable manifest file via
``protocol_manifest_sha256`` and uses ``barrier_commits/`` as its atomic source
of truth.

``events.jsonl`` has one generation event per line.  Required canonical fields
are ``event_id``, ``barrier_id``, ``task_id``, ``task_hash``, ``replicate_id``,
``step``, ``model_alias``, ``model_revision``, ``generation_kind`` (``main``,
``k2``, or ``verifier``), ``generation_seed``, ``seed_scope="per_event"``, UTC and monotonic start/end
timestamps, prompt/completion/total token counts, and ``event_sha256``.  The
event hash is computed over the event object with its hash field removed.

``barriers.jsonl`` has one record per task/replicate/step.  It names exactly one
``main_event_ids`` list covering the full frozen roster, records opening/closing
timestamps and fleet token totals, and supplies the event hashes.  A barrier is
usable only after it is closed.

``decisions.jsonl`` records the actual policy decision, not a retrospective
classifier row.  A decision must occur after its barrier closes, consume only
events from that barrier, record the exact token total of those event IDs, bind
to a frozen feature-contract hash, and choose an alias under the frozen selection
rule.  K2/verifier features are rejected unless separately generated and
explicitly consumed by that decision.

The auditor never writes to the supplied artifact directory.  ``--self-test``
uses a disposable system temporary directory only.

Exit statuses:
    0  VERIFIED: terminal protocol is internally coherent and complete.
   10  INCOMPLETE: valid provenance but collection/decision coverage is not terminal.
   20  REJECTED: a causal, roster, chronology, hash, token, or feature-contract
       violation was found; no deployable claim may use these artifacts.
   30  INVALID: required files or required schema fields are missing or malformed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from forced_choice_verifier import (
    ANONYMOUS_RATIONALE_V1,
    OPTION_LABELS,
    PROMPT_OPTIONS_ONLY_V1,
    SCORE_METHOD as VERIFIER_SCORE_METHOD,
    TOKEN_ACCOUNTING_CONTRACT,
    ForcedChoiceVerifierError,
    frozen_option_continuations,
    render_verifier_prompt,
    verifier_template_sha256,
)


EXIT_VERIFIED = 0
EXIT_INCOMPLETE = 10
EXIT_REJECTED = 20
EXIT_INVALID = 30

AUDIT_VERSION = 2
SUPPORTED_SCHEMA_PREFIXES = ("prospective-barrier-", "prospective-committee-")

MANIFEST_CANDIDATES = ("protocol_manifest.json", "prospective_protocol_manifest.json")
TASK_MANIFEST_CANDIDATES = ("prospective_tasks.jsonl", "tasks.jsonl")
SEALED_LABEL_CANDIDATES = ("sealed_gold_labels.jsonl",)
EVENT_CANDIDATES = ("events.jsonl", "event_ledger.jsonl")
BARRIER_CANDIDATES = ("barriers.jsonl", "barrier_ledger.jsonl")
DECISION_CANDIDATES = ("decisions.jsonl", "decision_ledger.jsonl")
STATUS_CANDIDATES = ("protocol_status.json",)
COMMIT_DIRECTORY_NAME = "barrier_commits"

FORBIDDEN_RUNTIME_FIELDS = {
    "correct",
    "is_correct",
    "utility",
    "expected_answer",
    "gold_answer",
    "gold_label",
    "target",
    "target_label",
    "offline_label",
}
FORBIDDEN_FEATURE_COLUMNS = {
    "correct",
    "is_correct",
    "utility",
    "expected_answer",
    "gold_answer",
    "gold_label",
    "target",
    "target_label",
    "task_id",
    "task_source_index",
    "run_id",
    "trajectory_id",
    "source_cell",
    "answer",
    "answer_normalized",
    "answer_key",
    "thought",
    "raw_text",
}
POST_QUERY_PREFIXES = {
    "k2_": "k2",
    "verifier_": "verifier",
}
TERMINAL_ACTIONS = {"stop", "continue"}
ALLOWED_ACTIONS = TERMINAL_ACTIONS | {"query_k2", "query_verifier"}
ALLOWED_GENERATION_KINDS = {"main", "k2", "verifier"}


@dataclass(frozen=True)
class Finding:
    severity: str
    category: str
    code: str
    message: str
    evidence: dict[str, Any]


class Findings:
    def __init__(self) -> None:
        self.items: list[Finding] = []

    def add(self, category: str, code: str, message: str, **evidence: Any) -> None:
        severity = {
            "invalid": "error",
            "rejected": "error",
            "incomplete": "warning",
            "warning": "warning",
            "info": "info",
        }.get(category, "warning")
        self.items.append(Finding(severity, category, code, message, evidence))

    def has(self, category: str) -> bool:
        return any(item.category == category for item in self.items)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.items]


@dataclass(frozen=True)
class RosterMember:
    alias: str
    model_source: str
    model_revision: str
    tokenizer_revision: str


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    task_hash: str
    public_prompt: str | None = None


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    barrier_id: str
    task_id: str
    task_hash: str
    replicate_id: str
    step: int
    model_alias: str
    model_revision: str
    generation_kind: str
    generation_seed: int
    seed_scope: str
    started_at: datetime
    completed_at: datetime
    started_monotonic_ns: int
    completed_monotonic_ns: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    event_sha256: str
    triggered_by_decision_id: str | None
    thought: str
    verifier_variant: str | None
    verifier_prompt_sha256: str | None


@dataclass(frozen=True)
class BarrierRecord:
    barrier_id: str
    task_id: str
    task_hash: str
    replicate_id: str
    step: int
    main_event_ids: tuple[str, ...]
    expected_aliases: tuple[str, ...]
    completed_aliases: tuple[str, ...]
    opened_at: datetime
    closed_at: datetime
    opened_monotonic_ns: int
    closed_monotonic_ns: int
    fleet_prompt_tokens: int
    fleet_completion_tokens: int
    fleet_total_tokens: int


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    barrier_id: str
    task_id: str
    task_hash: str
    replicate_id: str
    step: int
    stage: str
    action: str
    selected_alias: str
    selected_event_id: str
    selected_answer_hash: str
    decision_at: datetime
    decision_monotonic_ns: int
    consumed_event_ids: tuple[str, ...]
    consumed_prompt_tokens: int
    consumed_completion_tokens: int
    consumed_total_tokens: int
    feature_contract_id: str
    feature_contract_sha256: str
    is_stopping_decision: bool


@dataclass
class ManifestState:
    path: Path
    raw: dict[str, Any]
    canonical_sha256: str
    file_sha256: str
    protocol_id: str
    status: str
    roster: tuple[RosterMember, ...]
    roster_by_alias: dict[str, RosterMember]
    verifier_by_alias: dict[str, RosterMember]
    verifier_spec: dict[str, Any] | None
    replicate_ids: set[str]
    seed_scope: str
    max_steps: int
    task_records: dict[str, TaskRecord]
    sealed_gold_labels: dict[str, Any] | None
    policy_spec_sha256: str | None
    feature_contracts: dict[str, tuple[str, ...]]
    feature_contract_hashes: dict[str, str]
    selection_rule_id: str
    confirmation_eligible: bool
    require_commit_source: bool
    policy_type: str


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON shared with the prospective committee coordinator.

    UTF-8 text is deliberately preserved rather than escaped.  Hash validation
    must remain stable for non-ASCII prompts and answers as well as ASCII data.
    """

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def as_nonempty_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def as_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def as_positive_int(value: Any) -> int | None:
    value_int = as_nonnegative_int(value)
    return value_int if value_int is not None and value_int > 0 else None


def normalized_identifier(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_seed_scope(value: Any) -> str | None:
    text = as_nonempty_string(value)
    if text is None:
        return None
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"per_event", "event"}:
        return "per_event"
    if normalized in {"batch", "per_batch", "global", "run", "batch_seeded_nonconfirmatory"}:
        return normalized
    return normalized


def parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def utc_display(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


def first_present(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def hash_without(record: Mapping[str, Any], field_name: str) -> str:
    payload = dict(record)
    payload.pop(field_name, None)
    return canonical_json_hash(payload)


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def resolve_under_root(root: Path, requested: str | Path) -> Path | None:
    candidate = Path(requested)
    candidate = candidate if candidate.is_absolute() else root / candidate
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def discover_file(root: Path, override: str | Path | None, candidates: Sequence[str]) -> Path | None:
    if override is not None:
        return resolve_under_root(root, override)
    for name in candidates:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return root / candidates[0]


def read_json_object(path: Path, root: Path, findings: Findings, *, label: str, required: bool) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            findings.add("invalid", f"{label.upper()}_MISSING", f"{label} is missing.", path=display_path(path, root))
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        findings.add("invalid", f"{label.upper()}_UNREADABLE", f"{label} is not readable JSON.", path=display_path(path, root), error=str(error))
        return None
    if not isinstance(payload, dict):
        findings.add("invalid", f"{label.upper()}_NOT_OBJECT", f"{label} must be a JSON object.", path=display_path(path, root), actual_type=type(payload).__name__)
        return None
    return payload


def read_json_records(path: Path, root: Path, findings: Findings, *, label: str, required: bool) -> list[dict[str, Any]] | None:
    """Read a JSONL ledger or a JSON list/object ledger without writing sidecars."""

    if not path.is_file():
        if required:
            findings.add("invalid", f"{label.upper()}_MISSING", f"{label} is missing.", path=display_path(path, root))
        return None
    try:
        if path.suffix.lower() == ".jsonl":
            records: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        findings.add("invalid", f"{label.upper()}_ROW_NOT_OBJECT", f"{label} contains a non-object row.", path=display_path(path, root), line=line_number, actual_type=type(value).__name__)
                        continue
                    records.append(value)
            return records
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        findings.add("invalid", f"{label.upper()}_UNREADABLE", f"{label} could not be parsed as UTF-8 JSON/JSONL.", path=display_path(path, root), error=str(error))
        return None
    if isinstance(value, dict):
        key = label.lower().replace("_ledger", "")
        value = value.get(key, value.get("records"))
    if not isinstance(value, list):
        findings.add("invalid", f"{label.upper()}_NOT_LIST", f"{label} must be a JSONL sequence or JSON list.", path=display_path(path, root), actual_type=type(value).__name__)
        return None
    records = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            findings.add("invalid", f"{label.upper()}_ROW_NOT_OBJECT", f"{label} contains a non-object row.", path=display_path(path, root), index=index, actual_type=type(item).__name__)
            continue
        records.append(item)
    return records


def canonical_roster_members(value: Any, findings: Findings, *, label: str) -> tuple[RosterMember, ...]:
    if not isinstance(value, list) or not value:
        findings.add("invalid", f"{label.upper()}_INVALID", f"{label} must be a non-empty list of frozen model members.")
        return ()
    members: list[RosterMember] = []
    aliases: set[str] = set()
    for index, raw_member in enumerate(value):
        if not isinstance(raw_member, dict):
            findings.add("invalid", f"{label.upper()}_MEMBER_INVALID", f"{label} member must be an object.", index=index, actual_type=type(raw_member).__name__)
            continue
        alias = as_nonempty_string(first_present(raw_member, "alias", "model_alias"))
        source = as_nonempty_string(first_present(raw_member, "model_source", "hf_name", "source"))
        revision = as_nonempty_string(first_present(raw_member, "model_revision", "revision", "model_commit"))
        # A fixed tokenizer revision is mandatory.  The committee collector pins
        # it implicitly to the model revision when a separate tokenizer record
        # is not used, which is safe only because that equality is explicit here.
        tokenizer_revision = as_nonempty_string(first_present(raw_member, "tokenizer_revision", "tokenizer_commit", "revision"))
        if alias is None or source is None or revision is None or tokenizer_revision is None:
            findings.add(
                "invalid",
                f"{label.upper()}_MEMBER_FIELDS_MISSING",
                f"{label} member must pin alias, model source/revision, and tokenizer revision.",
                index=index,
                alias=alias,
            )
            continue
        if alias in aliases:
            findings.add("invalid", f"{label.upper()}_ALIAS_DUPLICATE", f"{label} has duplicate aliases.", alias=alias)
            continue
        aliases.add(alias)
        members.append(RosterMember(alias, source, revision, tokenizer_revision))
    return tuple(members)


def canonical_roster_hash(roster: Sequence[RosterMember]) -> str:
    return canonical_json_hash(
        [
            {
                "alias": member.alias,
                "model_source": member.model_source,
                "model_revision": member.model_revision,
                "tokenizer_revision": member.tokenizer_revision,
            }
            for member in roster
        ]
    )


def validate_frozen_policy_hash(manifest: Mapping[str, Any], policy: Mapping[str, Any], findings: Findings) -> None:
    """Require that a policy embedded in an immutable manifest is itself pinned."""

    claimed = manifest.get("policy_sha256")
    if not is_sha256(claimed):
        findings.add("invalid", "POLICY_HASH_INVALID", "Manifest must contain a SHA-256 policy_sha256 for its frozen policy.")
        return
    actual = canonical_json_hash(policy)
    if claimed != actual:
        findings.add("rejected", "POLICY_HASH_MISMATCH", "Frozen policy bytes differ from the manifest policy_sha256.", expected=claimed, actual=actual)


def feature_contracts_from_manifest(
    policy: Any,
    findings: Findings,
    *,
    require_stopping_contract: bool,
) -> tuple[dict[str, tuple[str, ...]], str | None, str | None]:
    """Validate either a deployable stopping contract or a selection-only policy.

    A full-horizon observational collector can have a sound immutable roster and
    barrier ledger without being a deployed stop policy.  It is useful evidence,
    but must never be upgraded into a stopping claim merely because it selected
    an answer after every complete barrier.
    """

    if not isinstance(policy, dict):
        findings.add("invalid", "POLICY_INVALID", "Manifest policy must be an object.")
        return {}, None, None
    policy_type = as_nonempty_string(policy.get("policy_type"))
    if policy_type is None:
        findings.add("invalid", "POLICY_TYPE_MISSING", "Manifest policy must declare a policy_type.")
    if policy.get("uses_gold_or_correctness") not in {False, None}:
        findings.add("rejected", "POLICY_USES_GOLD_OR_CORRECTNESS", "Runtime policy may not use gold answers, correctness, or utility labels.")
    peer_visibility = as_nonempty_string(policy.get("peer_visibility"))
    if peer_visibility is not None and peer_visibility != "post_complete_barrier_only":
        findings.add("rejected", "POLICY_PEER_VISIBILITY_INVALID", "Policy must prohibit peer visibility until the full barrier has closed.", peer_visibility=peer_visibility)
    selection_rule_id = as_nonempty_string(policy.get("selection_rule_id"))
    if require_stopping_contract and selection_rule_id is None:
        findings.add("invalid", "POLICY_SELECTION_RULE_MISSING", "Manifest policy must pin a selection_rule_id.")
    if selection_rule_id is None and policy_type is not None:
        # Selection-only ledgers still need a stable name for reporting.  This
        # value is not accepted as a stopping-rule identifier.
        selection_rule_id = f"observational:{policy_type}"
    if require_stopping_contract and policy.get("requires_full_roster") is not True:
        findings.add("invalid", "POLICY_FULL_ROSTER_NOT_REQUIRED", "Prospective strict policy must explicitly require the complete frozen roster.")
    raw_contracts = policy.get("feature_contracts")
    if raw_contracts is None:
        raw_columns = policy.get("feature_columns")
        if raw_columns is None and not require_stopping_contract:
            return {}, selection_rule_id, policy_type
        raw_contracts = {"default": raw_columns}
    if not isinstance(raw_contracts, dict) or not raw_contracts:
        if require_stopping_contract:
            findings.add("invalid", "POLICY_FEATURE_CONTRACTS_INVALID", "Manifest stopping policy must contain feature_columns or feature_contracts.")
        return {}, selection_rule_id, policy_type
    contracts: dict[str, tuple[str, ...]] = {}
    for contract_id, columns in raw_contracts.items():
        normalized_id = as_nonempty_string(contract_id)
        if normalized_id is None or not isinstance(columns, list) or not columns or not all(isinstance(column, str) and column.strip() for column in columns):
            findings.add("invalid", "POLICY_FEATURE_CONTRACT_INVALID", "Each feature contract must be a non-empty string list.", contract_id=contract_id)
            continue
        clean_columns = tuple(column.strip() for column in columns)
        duplicate_columns = sorted({column for column in clean_columns if clean_columns.count(column) > 1})
        if duplicate_columns:
            findings.add("invalid", "POLICY_FEATURE_DUPLICATES", "A frozen feature contract has duplicate columns.", contract_id=normalized_id, columns=duplicate_columns)
        forbidden = sorted(set(clean_columns) & FORBIDDEN_FEATURE_COLUMNS)
        suspicious = sorted(
            column for column in clean_columns
            if any(token in column.lower() for token in ("gold", "expected", "target_label", "correct_label"))
        )
        if forbidden or suspicious:
            findings.add(
                "rejected",
                "POLICY_FEATURE_CONTRACT_FORBIDDEN",
                "Frozen policy feature contract includes a label, gold answer, raw answer, task identifier, or provenance feature.",
                contract_id=normalized_id,
                forbidden=forbidden,
                suspicious=suspicious,
            )
        contracts[normalized_id] = clean_columns
    return contracts, selection_rule_id, policy_type


def read_task_manifest(root: Path, manifest: Mapping[str, Any], findings: Findings) -> dict[str, TaskRecord]:
    descriptor = manifest.get("task_manifest")
    if not isinstance(descriptor, dict):
        descriptor = {
            "path": manifest.get("task_manifest_path", TASK_MANIFEST_CANDIDATES[0]),
            "sha256": manifest.get("task_manifest_sha256"),
            "count": manifest.get("task_count"),
        }
    relative_path = as_nonempty_string(descriptor.get("path"))
    expected_hash = descriptor.get("sha256")
    expected_count_raw = descriptor.get("count")
    expected_count = as_positive_int(expected_count_raw) if expected_count_raw is not None else None
    if relative_path is None or not is_sha256(expected_hash) or (expected_count_raw is not None and expected_count is None):
        findings.add("invalid", "TASK_MANIFEST_DESCRIPTOR_INVALID", "task_manifest must pin a relative path and SHA-256; an optional count must be positive.", descriptor=descriptor)
        return {}
    path = resolve_under_root(root, relative_path)
    if path is None:
        findings.add("invalid", "TASK_MANIFEST_PATH_ESCAPES_ROOT", "Task manifest path must stay below the audited output directory.", path=relative_path)
        return {}
    if not path.is_file():
        findings.add("invalid", "TASK_MANIFEST_MISSING", "Pinned task manifest does not exist.", path=relative_path)
        return {}
    try:
        actual_hash = sha256_file(path)
    except OSError as error:
        findings.add("invalid", "TASK_MANIFEST_UNREADABLE", "Pinned task manifest could not be read.", path=relative_path, error=str(error))
        return {}
    if actual_hash != expected_hash:
        findings.add("rejected", "TASK_MANIFEST_HASH_MISMATCH", "Task manifest bytes differ from the hash frozen in protocol manifest.", expected=expected_hash, actual=actual_hash, path=relative_path)
        return {}
    rows = read_json_records(path, root, findings, label="task_manifest", required=True)
    if rows is None:
        return {}
    if not rows:
        findings.add("invalid", "TASK_MANIFEST_EMPTY", "Frozen task manifest must contain at least one task.", path=relative_path)
    if expected_count is not None and len(rows) != expected_count:
        findings.add("rejected", "TASK_MANIFEST_COUNT_MISMATCH", "Task manifest row count differs from its frozen descriptor.", expected=expected_count, actual=len(rows), path=relative_path)
    result: dict[str, TaskRecord] = {}
    for index, row in enumerate(rows, start=1):
        if set(row) & FORBIDDEN_RUNTIME_FIELDS:
            findings.add("rejected", "TASK_MANIFEST_GOLD_LABEL_EXPOSED", "Task manifest must not store gold labels in the runtime protocol directory.", index=index, fields=sorted(set(row) & FORBIDDEN_RUNTIME_FIELDS))
        task_id = as_nonempty_string(row.get("task_id"))
        task_hash = first_present(row, "task_hash", "task_prompt_sha256", "prompt_sha256")
        if task_id is None or not is_sha256(task_hash):
            findings.add("invalid", "TASK_MANIFEST_ROW_INVALID", "Each task record must contain task_id and a SHA-256 task_hash or prompt_sha256.", index=index)
            continue
        if task_id in result:
            findings.add("invalid", "TASK_MANIFEST_TASK_DUPLICATE", "Task manifest contains duplicate task_id values.", task_id=task_id)
            continue
        public_prompt = row.get("prompt")
        if public_prompt is not None and not isinstance(public_prompt, str):
            findings.add("invalid", "TASK_MANIFEST_PROMPT_INVALID", "A supplied public task prompt must be a string.", index=index)
            public_prompt = None
        # Older generic prospective schemas may use an opaque task hash that
        # is not the public-prompt hash.  Keep that compatible; a verifier
        # protocol below requires a prompt whose bytes do match before it can
        # validate the rendered forced-choice request.
        if isinstance(public_prompt, str) and hashlib.sha256(public_prompt.encode("utf-8")).hexdigest() != str(task_hash):
            public_prompt = None
        result[task_id] = TaskRecord(task_id, str(task_hash), public_prompt)
    return result


def read_opaque_sealed_labels(root: Path, manifest: Mapping[str, Any], findings: Findings) -> dict[str, Any] | None:
    """Verify label-file provenance without parsing or exposing its gold content.

    The online event/barrier/decision ledgers may never contain labels.  A sealed
    file is still required for a completed experiment so downstream offline
    evaluation can be tied to immutable gold labels.  This audit verifies only
    the filename, count, and bytes hash.
    """

    descriptor = manifest.get("sealed_gold_labels", manifest.get("sealed_labels", manifest.get("gold_labels")))
    if not isinstance(descriptor, dict):
        descriptor = {
            "path": manifest.get("sealed_gold_labels_path", SEALED_LABEL_CANDIDATES[0]),
            "sha256": manifest.get("sealed_gold_labels_sha256"),
            "count": manifest.get("sealed_gold_labels_count"),
        }
    relative_path = as_nonempty_string(descriptor.get("path"))
    expected_hash = descriptor.get("sha256")
    expected_count_raw = descriptor.get("count")
    expected_count = as_positive_int(expected_count_raw) if expected_count_raw is not None else None
    if relative_path is None or not is_sha256(expected_hash) or (expected_count_raw is not None and expected_count is None):
        findings.add("invalid", "SEALED_LABEL_DESCRIPTOR_INVALID", "Manifest must pin sealed_gold_labels path and SHA-256; an optional count must be positive.")
        return None
    if "not_available_to_policy" in descriptor and descriptor.get("not_available_to_policy") is not True:
        findings.add("rejected", "SEALED_LABEL_POLICY_VISIBILITY", "Sealed labels must be explicitly unavailable to the runtime policy.")
    path = resolve_under_root(root, relative_path)
    if path is None:
        findings.add("invalid", "SEALED_LABEL_PATH_ESCAPES_ROOT", "sealed_gold_labels path must stay below the audited output directory.", path=relative_path)
        return None
    if not path.is_file():
        findings.add("invalid", "SEALED_LABEL_FILE_MISSING", "Pinned sealed_gold_labels file does not exist.", path=relative_path)
        return None
    try:
        actual_hash = sha256_file(path)
        with path.open("rb") as handle:
            actual_count = sum(1 for line in handle if line.strip())
    except OSError as error:
        findings.add("invalid", "SEALED_LABEL_FILE_UNREADABLE", "Pinned sealed_gold_labels file could not be read for opaque provenance validation.", path=relative_path, error=str(error))
        return None
    if actual_hash != expected_hash:
        findings.add("rejected", "SEALED_LABEL_HASH_MISMATCH", "Sealed gold-label bytes differ from the frozen manifest hash.", path=relative_path, expected=expected_hash, actual=actual_hash)
    if actual_count <= 0:
        findings.add("invalid", "SEALED_LABEL_FILE_EMPTY", "Sealed gold-label file must contain at least one opaque record.", path=relative_path)
    if expected_count is not None and actual_count != expected_count:
        findings.add("rejected", "SEALED_LABEL_COUNT_MISMATCH", "Sealed gold-label record count differs from frozen manifest descriptor.", path=relative_path, expected=expected_count, actual=actual_count)
    return {
        "path": relative_path,
        "sha256": expected_hash,
        "count": expected_count if expected_count is not None else actual_count,
        "opaque": True,
    }


def read_policy_spec(root: Path, manifest: Mapping[str, Any], findings: Findings) -> tuple[dict[str, Any] | None, str | None]:
    """Load an optional separately frozen policy spec and bind its bytes hash.

    A small policy spec is useful when collection must begin after model/roster
    registration but before an evaluator is invoked.  If it exists, it must be
    explicitly hash-pinned by the manifest; an unpinned file is not evidence of
    a fixed policy.
    """

    descriptor = manifest.get("policy_spec")
    if descriptor is None:
        default = root / "policy_spec.json"
        if default.is_file():
            findings.add("rejected", "POLICY_SPEC_UNPINNED", "policy_spec.json exists but is not hash-pinned by protocol_manifest.json.")
        return None, None
    if isinstance(descriptor, str):
        descriptor = {"path": descriptor, "sha256": manifest.get("policy_spec_sha256")}
    if not isinstance(descriptor, dict):
        findings.add("invalid", "POLICY_SPEC_DESCRIPTOR_INVALID", "policy_spec must be an object with path and sha256.")
        return None, None
    relative_path = as_nonempty_string(descriptor.get("path"))
    expected_hash = descriptor.get("sha256")
    if relative_path is None or not is_sha256(expected_hash):
        findings.add("invalid", "POLICY_SPEC_DESCRIPTOR_INVALID", "policy_spec must pin a relative path and SHA-256.")
        return None, None
    path = resolve_under_root(root, relative_path)
    if path is None:
        findings.add("invalid", "POLICY_SPEC_PATH_ESCAPES_ROOT", "policy_spec path must stay below the audited output directory.", path=relative_path)
        return None, None
    payload = read_json_object(path, root, findings, label="policy_spec", required=True)
    if payload is None:
        return None, None
    try:
        actual_hash = sha256_file(path)
    except OSError as error:
        findings.add("invalid", "POLICY_SPEC_UNREADABLE", "Pinned policy spec could not be read.", path=relative_path, error=str(error))
        return None, None
    if actual_hash != expected_hash:
        findings.add("rejected", "POLICY_SPEC_HASH_MISMATCH", "policy_spec bytes differ from the hash frozen in protocol manifest.", expected=expected_hash, actual=actual_hash, path=relative_path)
    check_runtime_label_fields(payload, findings, kind="policy_spec", index=0)
    policy = payload.get("policy", payload)
    if not isinstance(policy, dict):
        findings.add("invalid", "POLICY_SPEC_POLICY_INVALID", "policy_spec must be a policy object or contain a policy object.")
        return None, str(expected_hash)
    return policy, str(expected_hash)


def validate_code_provenance(value: Any, findings: Findings) -> None:
    """Accept one code hash or a non-empty map of file names to code hashes."""

    if is_sha256(value):
        return
    if isinstance(value, dict) and value and all(isinstance(name, str) and is_sha256(digest) for name, digest in value.items()):
        return
    findings.add("invalid", "MANIFEST_CODE_HASH_INVALID", "Manifest must pin a SHA-256 code hash or a non-empty filename-to-hash map.")


def status_from_manifest_or_sidecar(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    canonical_sha256: str,
    file_sha256: str,
    findings: Findings,
) -> str:
    """Resolve mutable lifecycle state without weakening immutable provenance."""

    direct = as_nonempty_string(manifest.get("status"))
    if direct is not None:
        return direct.lower()
    status_path = root / STATUS_CANDIDATES[0]
    if not status_path.is_file():
        findings.add("incomplete", "PROTOCOL_STATUS_NOT_YET_PRESENT", "No mutable protocol status exists yet; treating the registered protocol as initialized.")
        return "initialized"
    status = read_json_object(status_path, root, findings, label="protocol_status", required=True)
    if status is None:
        return "invalid"
    raw_binding = status.get("protocol_manifest_sha256")
    canonical_binding = status.get("manifest_sha256")
    if not is_sha256(raw_binding) or raw_binding != file_sha256:
        findings.add("rejected", "STATUS_RAW_MANIFEST_BINDING_MISMATCH", "protocol_status.json must bind the raw immutable protocol_manifest.json bytes.", expected=file_sha256, actual=raw_binding)
    if canonical_binding is not None and (not is_sha256(canonical_binding) or canonical_binding != canonical_sha256):
        findings.add("rejected", "STATUS_MANIFEST_BINDING_MISMATCH", "protocol_status.json manifest_sha256 differs from the canonical manifest hash.", expected=canonical_sha256, actual=canonical_binding)
    state = as_nonempty_string(first_present(status, "state", "status"))
    if state is None:
        findings.add("invalid", "STATUS_STATE_MISSING", "protocol_status.json must contain a non-empty state.")
        return "invalid"
    normalized = state.lower()
    return {"running": "collecting", "incomplete": "collecting"}.get(normalized, normalized)


def validate_verifier_spec(
    raw_spec: Any,
    verifier_roster: Sequence[RosterMember],
    findings: Findings,
    *,
    main_roster: Sequence[RosterMember] = (),
) -> dict[str, Any] | None:
    """Fail closed on the exact forced-choice verifier event contract."""

    if raw_spec is None:
        if verifier_roster:
            findings.add("invalid", "VERIFIER_SPEC_MISSING", "A verifier_roster requires a hash-pinned verifier_spec.")
        return None
    if not isinstance(raw_spec, dict):
        findings.add("invalid", "VERIFIER_SPEC_INVALID", "verifier_spec must be an object.")
        return None
    spec_id = as_nonempty_string(raw_spec.get("spec_id"))
    mode = as_nonempty_string(raw_spec.get("mode"))
    ledger_alias = as_nonempty_string(raw_spec.get("ledger_alias"))
    active = as_nonempty_string(raw_spec.get("active_variant"))
    variants = raw_spec.get("variants")
    labels = raw_spec.get("option_labels")
    continuations = raw_spec.get("continuations")
    template_hashes = raw_spec.get("template_sha256")
    score_method = as_nonempty_string(raw_spec.get("score_method"))
    accounting = as_nonempty_string(raw_spec.get("token_accounting_contract"))
    stage = as_nonempty_string(raw_spec.get("post_barrier_stage"))
    rationale_cap = as_positive_int(raw_spec.get("rationale_max_chars"))
    supported_variants = {PROMPT_OPTIONS_ONLY_V1, ANONYMOUS_RATIONALE_V1}
    expected_templates = {variant: verifier_template_sha256(variant) for variant in variants} if isinstance(variants, list) else {}
    if (
        spec_id != "gpqa_forced_choice_v1"
        or mode != "collect_only"
        or ledger_alias is None
        or not isinstance(variants, list)
        or not variants
        or not all(isinstance(value, str) and value for value in variants)
        or len(set(variants)) != len(variants)
        or not set(variants).issubset(supported_variants)
        or active not in set(variants)
        or labels != list(OPTION_LABELS)
        or not isinstance(continuations, dict)
        or continuations != frozen_option_continuations()
        or not isinstance(template_hashes, dict)
        or set(template_hashes) != set(variants)
        or template_hashes != expected_templates
        or score_method != VERIFIER_SCORE_METHOD
        or accounting != TOKEN_ACCOUNTING_CONTRACT
        or stage != "mandatory_after_main_barrier_close_v1"
        or rationale_cap is None
    ):
        findings.add("invalid", "VERIFIER_SPEC_INVALID", "verifier_spec is missing or changes a frozen forced-choice contract.")
        return None
    aliases = {member.alias for member in verifier_roster}
    if aliases != {ledger_alias}:
        findings.add(
            "rejected",
            "VERIFIER_SPEC_ROSTER_MISMATCH",
            "verifier_spec ledger_alias must identify exactly the pinned verifier roster member.",
            ledger_alias=ledger_alias,
            verifier_aliases=sorted(aliases),
        )
    main_aliases = {member.alias for member in main_roster}
    if ledger_alias in main_aliases or aliases & main_aliases:
        findings.add(
            "rejected",
            "VERIFIER_ROSTER_ALIAS_COLLISION",
            "Verifier aliases must be distinct from every main-roster alias.",
            verifier_aliases=sorted(aliases),
            main_aliases=sorted(main_aliases),
        )
    return dict(raw_spec)


def validate_manifest(root: Path, path: Path, findings: Findings) -> ManifestState | None:
    raw = read_json_object(path, root, findings, label="protocol_manifest", required=True)
    if raw is None:
        return None
    try:
        file_sha256 = sha256_file(path)
    except OSError as error:
        findings.add("invalid", "MANIFEST_HASH_UNREADABLE", "Could not hash protocol_manifest.json bytes.", error=str(error))
        return None
    canonical_sha256 = canonical_json_hash(raw)
    schema_version = as_nonempty_string(raw.get("schema_version"))
    protocol_id = as_nonempty_string(raw.get("protocol_id"))
    created_at = parse_utc(raw.get("created_at_utc"))
    is_committee_schema = bool(schema_version and schema_version.startswith("prospective-committee-"))
    if schema_version is None or not schema_version.startswith(SUPPORTED_SCHEMA_PREFIXES):
        findings.add("invalid", "MANIFEST_SCHEMA_UNSUPPORTED", "Manifest schema_version must start with prospective-barrier- or prospective-committee-.", schema_version=schema_version)
    if protocol_id is None:
        findings.add("invalid", "MANIFEST_PROTOCOL_ID_MISSING", "Manifest must contain a non-empty protocol_id.")
    if created_at is None:
        findings.add("invalid", "MANIFEST_CREATED_TIMESTAMP_INVALID", "Manifest must contain timezone-aware created_at_utc.")
    if is_committee_schema and raw.get("immutable") is not True:
        findings.add("invalid", "MANIFEST_NOT_IMMUTABLE", "Committee protocol manifest must explicitly declare immutable=true.")
    validate_code_provenance(raw.get("code_sha256"), findings)

    randomness = raw.get("randomness") if isinstance(raw.get("randomness"), dict) else {}
    rng_scheme = as_nonempty_string(first_present(randomness, "event_seed_derivation", "rng_scheme_version")) or as_nonempty_string(raw.get("rng_scheme_version"))
    if rng_scheme is None:
        findings.add("invalid", "MANIFEST_RNG_SCHEME_MISSING", "Manifest must pin a per-event RNG derivation scheme.")
    seed_scope = normalize_seed_scope(first_present(randomness, "seed_scope") if randomness else raw.get("seed_scope"))
    if seed_scope is None:
        findings.add("invalid", "MANIFEST_SEED_SCOPE_MISSING", "Manifest must explicitly declare seed_scope=per_event for a confirmatory rollout.")
    elif seed_scope != "per_event":
        findings.add("rejected", "MANIFEST_SEED_SCOPE_NON_CONFIRMATORY", "Batch/global RNG seeding cannot establish independent per-event rollouts; use seed_scope=per_event.", seed_scope=seed_scope)

    confirmation_value = raw.get("confirmation_eligible", True)
    if not isinstance(confirmation_value, bool):
        findings.add("invalid", "MANIFEST_CONFIRMATION_ELIGIBILITY_INVALID", "confirmation_eligible must be a boolean when supplied.")
        confirmation_eligible = False
    else:
        confirmation_eligible = confirmation_value

    configuration = raw.get("configuration") if isinstance(raw.get("configuration"), dict) else {}
    raw_replicates = raw.get("replicate_ids", configuration.get("replicate_ids", configuration.get("replicas")))
    replicate_ids: set[str] = set()
    if not isinstance(raw_replicates, list) or not raw_replicates:
        findings.add("invalid", "MANIFEST_REPLICATES_INVALID", "Manifest must contain a non-empty replicate_ids/replicas list.")
    else:
        for raw_replicate in raw_replicates:
            replicate = normalized_identifier(raw_replicate)
            if replicate is None or replicate in replicate_ids:
                findings.add("invalid", "MANIFEST_REPLICATE_IDS_INVALID", "replicate IDs must be non-empty and unique.", value=raw_replicate)
            elif replicate is not None:
                replicate_ids.add(replicate)
    generation = raw.get("generation") if isinstance(raw.get("generation"), dict) else configuration
    max_steps = as_positive_int(generation.get("max_steps")) if isinstance(generation, dict) else None
    if max_steps is None:
        findings.add("invalid", "MANIFEST_MAX_STEPS_INVALID", "Manifest must pin a positive generation.max_steps or configuration.max_steps.")

    hardware = raw.get("hardware", raw.get("runtime_preflight"))
    gpu = hardware.get("gpu", hardware) if isinstance(hardware, dict) else None
    gpu_name = as_nonempty_string(first_present(gpu, "name", "gpu_name")) if isinstance(gpu, dict) else None
    vram_mib = as_positive_int(gpu.get("vram_mib")) if isinstance(gpu, dict) else None
    if vram_mib is None and isinstance(gpu, dict):
        vram_bytes = as_positive_int(gpu.get("gpu_total_vram_bytes"))
        vram_mib = (vram_bytes // (1024 * 1024)) if vram_bytes is not None else None
    if confirmation_eligible and (gpu_name is None or vram_mib is None):
        findings.add("invalid", "MANIFEST_HARDWARE_INVALID", "Confirmatory manifest must pin a GPU name and positive VRAM provenance.")
    elif not isinstance(gpu, dict):
        findings.add("invalid", "MANIFEST_HARDWARE_INVALID", "Manifest must contain a hardware/runtime_preflight object.")

    raw_roster = raw.get("roster", raw.get("models"))
    roster = canonical_roster_members(raw_roster, findings, label="roster")
    if roster and len(roster) < 2:
        findings.add("invalid", "ROSTER_TOO_SMALL", "A committee protocol requires at least two frozen roster members.", count=len(roster))
    roster_hash = raw.get("roster_sha256", raw.get("roster_hash"))
    roster_hash_candidates = {canonical_roster_hash(roster)} if roster else set()
    if isinstance(raw_roster, list):
        roster_hash_candidates.add(canonical_json_hash(raw_roster))
    if not is_sha256(roster_hash):
        findings.add("invalid", "ROSTER_HASH_INVALID", "Manifest must contain a SHA-256 roster_sha256.")
    elif roster_hash_candidates and roster_hash not in roster_hash_candidates:
        findings.add("rejected", "ROSTER_HASH_MISMATCH", "Manifest roster hash does not match its frozen roster object.", expected=sorted(roster_hash_candidates), actual=roster_hash)
    raw_verifier_roster = raw.get("verifier_roster", [])
    verifier_roster = canonical_roster_members(raw_verifier_roster, findings, label="verifier_roster") if raw.get("verifier_roster") is not None else ()
    verifier_roster_hash = raw.get("verifier_roster_sha256")
    if verifier_roster:
        verifier_hash_candidates = {canonical_roster_hash(verifier_roster)}
        if isinstance(raw_verifier_roster, list):
            verifier_hash_candidates.add(canonical_json_hash(raw_verifier_roster))
        if not is_sha256(verifier_roster_hash):
            findings.add("invalid", "VERIFIER_ROSTER_HASH_INVALID", "A non-empty verifier_roster requires verifier_roster_sha256.")
        elif verifier_roster_hash not in verifier_hash_candidates:
            findings.add(
                "rejected",
                "VERIFIER_ROSTER_HASH_MISMATCH",
                "verifier_roster_sha256 does not match the frozen verifier roster.",
                expected=sorted(verifier_hash_candidates),
                actual=verifier_roster_hash,
            )
    elif verifier_roster_hash not in {None, ""}:
        findings.add("invalid", "VERIFIER_ROSTER_HASH_UNEXPECTED", "A verifier_roster_sha256 was supplied without a verifier_roster.")
    verifier_spec = validate_verifier_spec(
        raw.get("verifier_spec"),
        verifier_roster,
        findings,
        main_roster=roster,
    )

    policy_spec, policy_spec_sha256 = read_policy_spec(root, raw, findings)
    policy_source = policy_spec if policy_spec is not None else raw.get("policy")
    if isinstance(policy_source, dict):
        validate_frozen_policy_hash(raw, raw.get("policy") if isinstance(raw.get("policy"), dict) else policy_source, findings)
    contracts, selection_rule_id, policy_type = feature_contracts_from_manifest(
        policy_source,
        findings,
        require_stopping_contract=confirmation_eligible,
    )
    task_records = read_task_manifest(root, raw, findings)
    sealed_gold_labels = read_opaque_sealed_labels(root, raw, findings)
    if sealed_gold_labels is not None and task_records and sealed_gold_labels.get("count") != len(task_records):
        findings.add("rejected", "SEALED_LABEL_TASK_COUNT_MISMATCH", "Opaque sealed-label count must equal the frozen public task count.", labels=sealed_gold_labels.get("count"), tasks=len(task_records))

    status = status_from_manifest_or_sidecar(
        root,
        raw,
        canonical_sha256=canonical_sha256,
        file_sha256=file_sha256,
        findings=findings,
    )
    require_commit_source = is_committee_schema or raw.get("canonical_ledger_source") == COMMIT_DIRECTORY_NAME
    if findings.has("invalid") or protocol_id is None or seed_scope is None or max_steps is None or selection_rule_id is None or policy_type is None:
        return None
    return ManifestState(
        path=path,
        raw=raw,
        canonical_sha256=canonical_sha256,
        file_sha256=file_sha256,
        protocol_id=protocol_id,
        status=status,
        roster=roster,
        roster_by_alias={member.alias: member for member in roster},
        verifier_by_alias={member.alias: member for member in verifier_roster},
        verifier_spec=verifier_spec,
        replicate_ids=replicate_ids,
        seed_scope=seed_scope,
        max_steps=max_steps,
        task_records=task_records,
        sealed_gold_labels=sealed_gold_labels,
        policy_spec_sha256=policy_spec_sha256,
        feature_contracts=contracts,
        feature_contract_hashes={key: canonical_json_hash(list(value)) for key, value in contracts.items()},
        selection_rule_id=selection_rule_id,
        confirmation_eligible=confirmation_eligible,
        require_commit_source=require_commit_source,
        policy_type=policy_type,
    )


def record_hash_valid(record: Mapping[str, Any], field: str, findings: Findings, *, kind: str, index: int) -> str | None:
    claimed = record.get(field)
    if not is_sha256(claimed):
        findings.add("invalid", f"{kind.upper()}_HASH_INVALID", f"{kind} must include a SHA-256 {field}.", index=index)
        return None
    actual = hash_without(record, field)
    if actual != claimed:
        findings.add("rejected", f"{kind.upper()}_HASH_MISMATCH", f"{kind} record differs from its claimed canonical hash.", index=index, expected=claimed, actual=actual)
    return str(claimed)


def validate_record_manifest_binding(
    record: Mapping[str, Any],
    manifest: ManifestState,
    findings: Findings,
    *,
    kind: str,
    index: int,
) -> None:
    """Verify both canonical-object and raw-file manifest provenance.

    ``manifest_sha256`` protects semantic manifest content, while
    ``protocol_manifest_sha256`` protects the actual immutable file used by the
    coordinator.  Keeping both prevents a JSON reformat/canonicalization mix-up
    from silently crossing protocol registrations.
    """

    canonical = record.get("manifest_sha256")
    raw_file = record.get("protocol_manifest_sha256")
    if record.get("protocol_id") != manifest.protocol_id or canonical != manifest.canonical_sha256:
        findings.add(
            "rejected",
            f"{kind.upper()}_MANIFEST_BINDING_MISMATCH",
            f"{kind} does not bind to the audited canonical protocol manifest.",
            index=index,
            protocol_id=record.get("protocol_id"),
            manifest_sha256=canonical,
            expected_manifest_sha256=manifest.canonical_sha256,
        )
    if manifest.require_commit_source:
        if raw_file != manifest.file_sha256:
            findings.add(
                "rejected",
                f"{kind.upper()}_RAW_MANIFEST_BINDING_MISMATCH",
                f"{kind} does not bind to the raw immutable protocol_manifest.json bytes.",
                index=index,
                protocol_manifest_sha256=raw_file,
                expected_protocol_manifest_sha256=manifest.file_sha256,
            )
    elif raw_file is not None and raw_file not in {manifest.file_sha256, manifest.canonical_sha256}:
        findings.add(
            "rejected",
            f"{kind.upper()}_RAW_MANIFEST_BINDING_MISMATCH",
            f"{kind} protocol_manifest_sha256 matches neither accepted manifest binding.",
            index=index,
            protocol_manifest_sha256=raw_file,
        )


def check_runtime_label_fields(record: Mapping[str, Any], findings: Findings, *, kind: str, index: int) -> None:
    prohibited = sorted(set(record) & FORBIDDEN_RUNTIME_FIELDS)
    if prohibited:
        findings.add("rejected", f"{kind.upper()}_LABEL_FIELD_EXPOSED", "Runtime ledger exposes an offline gold/label/utility field.", index=index, fields=prohibited)


def validate_verifier_event_payload(record: Mapping[str, Any], index: int, manifest: ManifestState, findings: Findings) -> None:
    """Validate the score, posterior, and explicit four-forward token ledger."""

    spec = manifest.verifier_spec
    if spec is None:
        findings.add("rejected", "VERIFIER_EVENT_WITHOUT_SPEC", "Verifier event is present without a frozen verifier_spec.", index=index)
        return
    labels = ("A", "B", "C", "D")
    variant = as_nonempty_string(record.get("verifier_variant"))
    template_hash = record.get("verifier_template_sha256")
    prompt_hash = record.get("verifier_prompt_sha256")
    event_labels = record.get("verifier_option_labels")
    logits_raw = record.get("verifier_option_logprobs")
    posteriors_raw = record.get("verifier_option_posteriors")
    option_tokens_raw = record.get("verifier_option_scoring_tokens")
    base_tokens = as_positive_int(record.get("verifier_base_prompt_tokens"))
    argmax = as_nonempty_string(record.get("verifier_argmax_option"))
    margin_raw = record.get("verifier_top1_margin")
    entropy_raw = record.get("verifier_entropy")
    try:
        logits = {label: float(logits_raw[label]) for label in labels} if isinstance(logits_raw, dict) and set(logits_raw) == set(labels) else None
        posteriors = {label: float(posteriors_raw[label]) for label in labels} if isinstance(posteriors_raw, dict) and set(posteriors_raw) == set(labels) else None
        option_tokens = {label: as_positive_int(option_tokens_raw[label]) for label in labels} if isinstance(option_tokens_raw, dict) and set(option_tokens_raw) == set(labels) else None
        margin = float(margin_raw)
        entropy = float(entropy_raw)
    except (TypeError, ValueError, KeyError):
        logits = None
        posteriors = None
        option_tokens = None
        margin = float("nan")
        entropy = float("nan")
    contract_ok = (
        record.get("verifier_spec_id") == spec.get("spec_id")
        and variant in set(spec.get("variants", []))
        and record.get("post_barrier_stage") == spec.get("post_barrier_stage")
        and record.get("verifier_score_method") == spec.get("score_method")
        and record.get("verifier_token_accounting_contract") == spec.get("token_accounting_contract")
        and event_labels == list(labels)
        and is_sha256(prompt_hash)
        and is_sha256(template_hash)
        and isinstance(spec.get("template_sha256"), dict)
        and template_hash == spec["template_sha256"].get(variant)
        and record.get("domain") == "gpqa"
        and record.get("answer_type") == "mcq"
    )
    numeric_ok = (
        logits is not None
        and posteriors is not None
        and option_tokens is not None
        and base_tokens is not None
        and all(math.isfinite(value) for value in logits.values())
        and all(math.isfinite(value) and value >= 0.0 for value in posteriors.values())
        and all(value is not None for value in option_tokens.values())
        and math.isfinite(margin)
        and math.isfinite(entropy)
    )
    posterior_ok = False
    if numeric_ok and logits is not None and posteriors is not None:
        maximum = max(logits.values())
        weights = {label: math.exp(logits[label] - maximum) for label in labels}
        denominator = sum(weights.values())
        expected = {label: weights[label] / denominator for label in labels}
        ranked = sorted(labels, key=lambda label: (-expected[label], label))
        expected_margin = expected[ranked[0]] - expected[ranked[1]]
        expected_entropy = -sum(value * math.log(value) for value in expected.values() if value > 0.0)
        posterior_ok = (
            abs(sum(posteriors.values()) - 1.0) <= 1e-6
            and all(abs(posteriors[label] - expected[label]) <= 1e-5 for label in labels)
            and argmax == ranked[0]
            and abs(margin - expected_margin) <= 1e-5
            and abs(entropy - expected_entropy) <= 1e-5
        )
    token_ok = False
    if numeric_ok and option_tokens is not None and base_tokens is not None:
        event_prompt_tokens = as_nonnegative_int(record.get("prompt_tokens"))
        event_completion_tokens = as_nonnegative_int(record.get("completion_tokens"))
        token_ok = (
            event_prompt_tokens is not None
            and event_completion_tokens is not None
            and event_prompt_tokens == int(base_tokens) * len(labels)
            and event_completion_tokens == sum(int(value) for value in option_tokens.values())
        )
    if not (contract_ok and numeric_ok and posterior_ok and token_ok):
        findings.add(
            "rejected",
            "VERIFIER_EVENT_CONTRACT_INVALID",
            "Verifier event does not match its frozen exact four-option score/posterior/token contract.",
            index=index,
            contract_ok=contract_ok,
            numeric_ok=numeric_ok,
            posterior_ok=posterior_ok,
            token_ok=token_ok,
        )


def validate_event(record: Mapping[str, Any], index: int, manifest: ManifestState, findings: Findings) -> EventRecord | None:
    check_runtime_label_fields(record, findings, kind="event", index=index)
    event_hash = record_hash_valid(record, "event_sha256", findings, kind="event", index=index)
    event_id = as_nonempty_string(record.get("event_id"))
    barrier_id = as_nonempty_string(record.get("barrier_id"))
    task_id = as_nonempty_string(record.get("task_id"))
    task_hash = first_present(record, "task_hash", "task_prompt_sha256", "prompt_sha256")
    replicate_id = normalized_identifier(first_present(record, "replicate_id", "replica_id"))
    step = as_positive_int(record.get("step"))
    alias = as_nonempty_string(record.get("model_alias"))
    revision = as_nonempty_string(record.get("model_revision"))
    generation_kind = as_nonempty_string(record.get("generation_kind"))
    seed = as_nonnegative_int(first_present(record, "generation_seed", "event_seed"))
    seed_scope = normalize_seed_scope(record.get("seed_scope"))
    started_at = parse_utc(first_present(record, "started_at_utc", "generation_started_at_utc", "event_started_at_utc"))
    completed_at = parse_utc(first_present(record, "completed_at_utc", "ended_at_utc", "generation_completed_at_utc", "event_completed_at_utc"))
    started_ns = as_nonnegative_int(first_present(record, "started_monotonic_ns", "generation_started_monotonic_ns", "event_started_monotonic_ns"))
    completed_ns = as_nonnegative_int(first_present(record, "completed_monotonic_ns", "ended_monotonic_ns", "generation_completed_monotonic_ns", "event_completed_monotonic_ns"))
    prompt_tokens = as_nonnegative_int(record.get("prompt_tokens"))
    completion_tokens = as_nonnegative_int(record.get("completion_tokens"))
    total_tokens = as_nonnegative_int(record.get("total_tokens"))
    required = {
        "event_id": event_id,
        "barrier_id": barrier_id,
        "task_id": task_id,
        "task_hash": task_hash if is_sha256(task_hash) else None,
        "replicate_id": replicate_id,
        "step": step,
        "model_alias": alias,
        "model_revision": revision,
        "generation_kind": generation_kind,
        "generation_seed": seed,
        "seed_scope": seed_scope,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "started_monotonic_ns": started_ns,
        "completed_monotonic_ns": completed_ns,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "event_sha256": event_hash,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        findings.add("invalid", "EVENT_FIELDS_MISSING", "Event lacks required canonical fields.", index=index, missing=missing)
        return None
    validate_record_manifest_binding(record, manifest, findings, kind="event", index=index)
    if task_id not in manifest.task_records or manifest.task_records[task_id].task_hash != task_hash:
        findings.add("rejected", "EVENT_TASK_MANIFEST_MISMATCH", "Event task_id/task_hash is not in the frozen task manifest.", index=index, task_id=task_id)
    if replicate_id not in manifest.replicate_ids:
        findings.add("rejected", "EVENT_REPLICATE_NOT_FROZEN", "Event uses a replicate_id absent from the frozen manifest.", index=index, replicate_id=replicate_id)
    if step > manifest.max_steps:
        findings.add("rejected", "EVENT_STEP_OUT_OF_RANGE", "Event step exceeds frozen generation.max_steps.", index=index, step=step, max_steps=manifest.max_steps)
    if generation_kind not in ALLOWED_GENERATION_KINDS:
        findings.add("invalid", "EVENT_GENERATION_KIND_INVALID", "Event generation_kind must be main, k2, or verifier.", index=index, generation_kind=generation_kind)
    if seed_scope != "per_event":
        findings.add("rejected", "EVENT_SEED_SCOPE_NON_CONFIRMATORY", "Each event must declare seed_scope=per_event; batch/global seeding is non-confirmatory.", index=index, seed_scope=seed_scope)
    elif seed_scope != manifest.seed_scope:
        findings.add("rejected", "EVENT_SEED_SCOPE_MANIFEST_MISMATCH", "Event seed_scope differs from frozen manifest seed_scope.", index=index, event_seed_scope=seed_scope, manifest_seed_scope=manifest.seed_scope)
    roster_member = manifest.roster_by_alias.get(alias)
    verifier_member = manifest.verifier_by_alias.get(alias)
    if generation_kind in {"main", "k2"}:
        if roster_member is None or roster_member.model_revision != revision:
            findings.add("rejected", "EVENT_ROSTER_MEMBER_MISMATCH", "Main/K2 event alias or revision differs from frozen roster.", index=index, alias=alias, revision=revision)
    elif generation_kind == "verifier" and (verifier_member is None or verifier_member.model_revision != revision):
        findings.add("rejected", "EVENT_VERIFIER_MEMBER_MISMATCH", "Verifier event alias or revision is not pinned in verifier_roster.", index=index, alias=alias, revision=revision)
    if generation_kind == "verifier":
        validate_verifier_event_payload(record, index, manifest, findings)
    if started_at > completed_at or started_ns >= completed_ns:
        findings.add("rejected", "EVENT_TIME_ORDER_INVALID", "Event end timestamp must not precede start, and its monotonic end must strictly follow start.", index=index, started_at=utc_display(started_at), completed_at=utc_display(completed_at), started_monotonic_ns=started_ns, completed_monotonic_ns=completed_ns)
    if total_tokens != prompt_tokens + completion_tokens:
        findings.add("rejected", "EVENT_TOKEN_TOTAL_MISMATCH", "Event total_tokens must equal prompt_tokens + completion_tokens.", index=index, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
    parent = as_nonempty_string(record.get("triggered_by_decision_id"))
    thought_value = record.get("thought", "")
    if generation_kind == "main" and manifest.verifier_spec is not None and not isinstance(thought_value, str):
        findings.add(
            "rejected",
            "VERIFIER_MAIN_THOUGHT_INVALID",
            "Verifier-enabled main events must retain a string thought field so the anonymized prompt can be reproduced.",
            index=index,
        )
    verifier_variant = as_nonempty_string(record.get("verifier_variant")) if generation_kind == "verifier" else None
    verifier_prompt_hash = record.get("verifier_prompt_sha256") if generation_kind == "verifier" else None
    return EventRecord(
        event_id=event_id,
        barrier_id=barrier_id,
        task_id=task_id,
        task_hash=str(task_hash),
        replicate_id=replicate_id,
        step=step,
        model_alias=alias,
        model_revision=revision,
        generation_kind=generation_kind,
        generation_seed=seed,
        seed_scope=seed_scope,
        started_at=started_at,
        completed_at=completed_at,
        started_monotonic_ns=started_ns,
        completed_monotonic_ns=completed_ns,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        event_sha256=event_hash,
        triggered_by_decision_id=parent,
        thought=thought_value if isinstance(thought_value, str) else "",
        verifier_variant=verifier_variant,
        verifier_prompt_sha256=str(verifier_prompt_hash) if isinstance(verifier_prompt_hash, str) else None,
    )


def validate_barrier(
    record: Mapping[str, Any],
    index: int,
    manifest: ManifestState,
    findings: Findings,
    *,
    commit: Mapping[str, Any] | None = None,
) -> BarrierRecord | None:
    check_runtime_label_fields(record, findings, kind="barrier", index=index)
    record_hash_valid(record, "barrier_sha256", findings, kind="barrier", index=index)
    barrier_id = as_nonempty_string(record.get("barrier_id"))
    task_id = as_nonempty_string(record.get("task_id"))
    task_hash = first_present(record, "task_hash", "task_prompt_sha256", "prompt_sha256")
    replicate_id = normalized_identifier(first_present(record, "replicate_id", "replica_id"))
    step = as_positive_int(record.get("step"))
    committed_events = commit.get("events") if isinstance(commit, Mapping) and isinstance(commit.get("events"), list) else []
    raw_main_ids = first_present(record, "main_event_ids", "event_ids")
    if raw_main_ids is None and committed_events:
        raw_main_ids = [event.get("event_id") for event in committed_events if isinstance(event, Mapping)]
    raw_expected = first_present(record, "expected_aliases", "roster")
    raw_completed = first_present(record, "completed_aliases")
    if raw_completed is None and record.get("complete_roster") is True:
        raw_completed = raw_expected
    opened_at = parse_utc(first_present(record, "opened_at_utc", "barrier_opened_at_utc"))
    closed_at = parse_utc(first_present(record, "closed_at_utc", "barrier_closed_at_utc"))
    opened_ns = as_nonnegative_int(first_present(record, "opened_monotonic_ns", "barrier_opened_monotonic_ns"))
    closed_ns = as_nonnegative_int(first_present(record, "closed_monotonic_ns", "barrier_closed_monotonic_ns"))
    fleet_prompt = as_nonnegative_int(first_present(record, "fleet_prompt_tokens", "prompt_tokens"))
    fleet_completion = as_nonnegative_int(first_present(record, "fleet_completion_tokens", "completion_tokens"))
    fleet_total = as_nonnegative_int(first_present(record, "fleet_total_tokens", "total_tokens"))
    main_ids = tuple(str(value) for value in raw_main_ids) if isinstance(raw_main_ids, list) and all(as_nonempty_string(value) for value in raw_main_ids) else ()
    expected_aliases = tuple(str(value) for value in raw_expected) if isinstance(raw_expected, list) and all(as_nonempty_string(value) for value in raw_expected) else ()
    completed_aliases = tuple(str(value) for value in raw_completed) if isinstance(raw_completed, list) and all(as_nonempty_string(value) for value in raw_completed) else ()
    required = {
        "barrier_id": barrier_id,
        "task_id": task_id,
        "task_hash": task_hash if is_sha256(task_hash) else None,
        "replicate_id": replicate_id,
        "step": step,
        "main_event_ids": main_ids if main_ids else None,
        "expected_aliases": expected_aliases if expected_aliases else None,
        "completed_aliases": completed_aliases if completed_aliases else None,
        "opened_at_utc": opened_at,
        "closed_at_utc": closed_at,
        "opened_monotonic_ns": opened_ns,
        "closed_monotonic_ns": closed_ns,
        "fleet_prompt_tokens": fleet_prompt,
        "fleet_completion_tokens": fleet_completion,
        "fleet_total_tokens": fleet_total,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        findings.add("invalid", "BARRIER_FIELDS_MISSING", "Barrier lacks required canonical fields.", index=index, missing=missing)
        return None
    validate_record_manifest_binding(record, manifest, findings, kind="barrier", index=index)
    complete = record.get("barrier_complete", record.get("complete_roster"))
    status = str(record.get("status", "complete" if complete is True else "")).lower()
    if complete is not True or status != "complete":
        findings.add("rejected", "BARRIER_NOT_COMPLETE", "A barrier ledger record must explicitly state complete roster/barrier=true and status=complete (or be covered by an atomic complete commit).", index=index, status=status, barrier_complete=complete)
    if task_id not in manifest.task_records or manifest.task_records[task_id].task_hash != task_hash:
        findings.add("rejected", "BARRIER_TASK_MANIFEST_MISMATCH", "Barrier task_id/task_hash is not in the frozen task manifest.", index=index, task_id=task_id)
    if replicate_id not in manifest.replicate_ids:
        findings.add("rejected", "BARRIER_REPLICATE_NOT_FROZEN", "Barrier uses a replicate absent from frozen manifest.", index=index, replicate_id=replicate_id)
    if step > manifest.max_steps:
        findings.add("rejected", "BARRIER_STEP_OUT_OF_RANGE", "Barrier step exceeds frozen generation.max_steps.", index=index, step=step)
    roster_aliases = tuple(member.alias for member in manifest.roster)
    if set(expected_aliases) != set(roster_aliases) or len(expected_aliases) != len(set(expected_aliases)):
        findings.add("rejected", "BARRIER_EXPECTED_ROSTER_MISMATCH", "Barrier expected_aliases must be exactly the frozen roster without duplicates.", index=index, expected=sorted(roster_aliases), actual=sorted(expected_aliases))
    if set(completed_aliases) != set(roster_aliases) or len(completed_aliases) != len(set(completed_aliases)):
        findings.add("rejected", "BARRIER_COMPLETED_ROSTER_MISMATCH", "Barrier completed_aliases must be exactly the frozen roster without duplicates.", index=index, expected=sorted(roster_aliases), actual=sorted(completed_aliases))
    if len(main_ids) != len(set(main_ids)):
        findings.add("rejected", "BARRIER_MAIN_EVENT_IDS_DUPLICATE", "Barrier main_event_ids must not contain duplicates.", index=index)
    raw_hashes = record.get("event_hashes")
    if raw_hashes is None and isinstance(commit, Mapping):
        raw_hashes = commit.get("event_hashes")
    if not isinstance(raw_hashes, dict) or not raw_hashes or not all(isinstance(key, str) and is_sha256(value) for key, value in raw_hashes.items()):
        findings.add("invalid", "BARRIER_EVENT_HASHES_INVALID", "Barrier or its atomic commit must map every main event (by ID or alias) to a SHA-256 hash.", index=index)
    if opened_at > closed_at or opened_ns >= closed_ns:
        findings.add("rejected", "BARRIER_TIME_ORDER_INVALID", "Barrier close must not precede open, and its monotonic close must strictly follow open.", index=index)
    if fleet_total != fleet_prompt + fleet_completion:
        findings.add("rejected", "BARRIER_TOKEN_TOTAL_MISMATCH", "Barrier fleet_total_tokens must equal fleet prompt + completion tokens.", index=index)
    return BarrierRecord(
        barrier_id=barrier_id,
        task_id=task_id,
        task_hash=str(task_hash),
        replicate_id=replicate_id,
        step=step,
        main_event_ids=main_ids,
        expected_aliases=expected_aliases,
        completed_aliases=completed_aliases,
        opened_at=opened_at,
        closed_at=closed_at,
        opened_monotonic_ns=opened_ns,
        closed_monotonic_ns=closed_ns,
        fleet_prompt_tokens=fleet_prompt,
        fleet_completion_tokens=fleet_completion,
        fleet_total_tokens=fleet_total,
    )


def validate_decision(record: Mapping[str, Any], index: int, manifest: ManifestState, findings: Findings) -> DecisionRecord | None:
    check_runtime_label_fields(record, findings, kind="decision", index=index)
    record_hash_valid(record, "decision_sha256", findings, kind="decision", index=index)
    decision_id = as_nonempty_string(record.get("decision_id"))
    barrier_id = as_nonempty_string(record.get("barrier_id"))
    task_id = as_nonempty_string(record.get("task_id"))
    task_hash = first_present(record, "task_hash", "task_prompt_sha256", "prompt_sha256")
    replicate_id = normalized_identifier(first_present(record, "replicate_id", "replica_id"))
    step = as_positive_int(record.get("step"))
    is_stopping_decision = manifest.confirmation_eligible
    stage = as_nonempty_string(first_present(record, "decision_stage", "stage"))
    if stage is None and not is_stopping_decision:
        stage = "selection"
    action = as_nonempty_string(first_present(record, "action", "stopping_action", "selection_action"))
    selected_alias = as_nonempty_string(first_present(record, "selected_alias", "selected_model_alias"))
    selected_event_id = as_nonempty_string(record.get("selected_event_id"))
    selected_answer_hash = as_nonempty_string(record.get("selected_answer_hash"))
    if selected_answer_hash is None and isinstance(record.get("selected_answer"), str):
        selected_answer_hash = hashlib.sha256(record["selected_answer"].encode("utf-8")).hexdigest()
    decision_at = parse_utc(first_present(record, "decision_at_utc", "decision_created_at_utc", "decided_at_utc"))
    decision_ns = as_nonnegative_int(first_present(record, "decision_monotonic_ns", "decision_created_monotonic_ns", "decided_monotonic_ns"))
    raw_consumed = record.get("consumed_event_ids")
    consumed_ids = tuple(str(value) for value in raw_consumed) if isinstance(raw_consumed, list) and all(as_nonempty_string(value) for value in raw_consumed) else ()
    consumed_prompt = as_nonnegative_int(record.get("consumed_prompt_tokens"))
    consumed_completion = as_nonnegative_int(record.get("consumed_completion_tokens"))
    consumed_total = as_nonnegative_int(record.get("consumed_total_tokens"))
    contract_id = as_nonempty_string(record.get("feature_contract_id"))
    contract_hash = record.get("feature_contract_sha256")
    if not is_stopping_decision and contract_id is None:
        contract_id = "observational"
    if not is_stopping_decision and not is_sha256(contract_hash):
        raw_contract = record.get("feature_contract", [])
        contract_hash = canonical_json_hash(raw_contract) if isinstance(raw_contract, list) else canonical_json_hash([])
    required = {
        "decision_id": decision_id,
        "barrier_id": barrier_id,
        "task_id": task_id,
        "task_hash": task_hash if is_sha256(task_hash) else None,
        "replicate_id": replicate_id,
        "step": step,
        "decision_stage": stage,
        "selected_alias": selected_alias,
        "selected_event_id": selected_event_id,
        "selected_answer_hash": selected_answer_hash,
        "decision_at_utc": decision_at,
        "decision_monotonic_ns": decision_ns,
        "consumed_event_ids": consumed_ids if consumed_ids else None,
        "consumed_prompt_tokens": consumed_prompt,
        "consumed_completion_tokens": consumed_completion,
        "consumed_total_tokens": consumed_total,
        "feature_contract_id": contract_id if is_stopping_decision else "observational",
        "feature_contract_sha256": contract_hash if is_sha256(contract_hash) else None,
    }
    if is_stopping_decision:
        required["action"] = action
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        findings.add("invalid", "DECISION_FIELDS_MISSING", "Decision lacks required canonical fields.", index=index, missing=missing)
        return None
    validate_record_manifest_binding(record, manifest, findings, kind="decision", index=index)
    if is_stopping_decision and record.get("barrier_complete") is not True:
        findings.add("rejected", "DECISION_BARRIER_NOT_DECLARED_COMPLETE", "Decision must explicitly declare barrier_complete=true.", index=index)
    if task_id not in manifest.task_records or manifest.task_records[task_id].task_hash != task_hash:
        findings.add("rejected", "DECISION_TASK_MANIFEST_MISMATCH", "Decision task_id/task_hash is not in frozen task manifest.", index=index, task_id=task_id)
    if replicate_id not in manifest.replicate_ids or step > manifest.max_steps:
        findings.add("rejected", "DECISION_SCOPE_NOT_FROZEN", "Decision replicate or step is outside frozen manifest scope.", index=index, replicate_id=replicate_id, step=step)
    if is_stopping_decision and action not in ALLOWED_ACTIONS:
        findings.add("invalid", "DECISION_ACTION_INVALID", "Decision action must be stop, continue, query_k2, or query_verifier.", index=index, action=action)
    if not is_stopping_decision and as_nonempty_string(record.get("selection_action")) is None:
        findings.add("invalid", "OBSERVATIONAL_SELECTION_ACTION_MISSING", "Selection-only decision must declare selection_action (for example select_answer).", index=index)
    if selected_alias not in manifest.roster_by_alias:
        findings.add("rejected", "DECISION_SELECTED_ALIAS_NOT_ROSTER", "Decision selected_alias is not a frozen main-roster member.", index=index, selected_alias=selected_alias)
    if selected_event_id is None:
        findings.add("invalid", "DECISION_SELECTED_EVENT_ID_MISSING", "Decision must identify the event that supplied its selected answer.", index=index)
    if is_stopping_decision and record.get("selection_rule_id") != manifest.selection_rule_id:
        findings.add("rejected", "DECISION_SELECTION_RULE_MISMATCH", "Decision selection_rule_id differs from frozen manifest policy.", index=index, expected=manifest.selection_rule_id, actual=record.get("selection_rule_id"))
    if manifest.policy_spec_sha256 is not None and record.get("policy_spec_sha256") != manifest.policy_spec_sha256:
        findings.add("rejected", "DECISION_POLICY_SPEC_MISMATCH", "Decision does not bind to the hash-pinned policy_spec.json.", index=index, expected=manifest.policy_spec_sha256, actual=record.get("policy_spec_sha256"))
    if len(consumed_ids) != len(set(consumed_ids)):
        findings.add("rejected", "DECISION_CONSUMED_EVENT_IDS_DUPLICATE", "Decision consumed_event_ids must not double-count an event.", index=index)
    if consumed_total != consumed_prompt + consumed_completion:
        findings.add("rejected", "DECISION_TOKEN_TOTAL_MISMATCH", "Decision consumed_total_tokens must equal consumed prompt + completion tokens.", index=index)
    expected_contract_hash = manifest.feature_contract_hashes.get(contract_id)
    if is_stopping_decision and (expected_contract_hash is None or expected_contract_hash != contract_hash):
        findings.add("rejected", "DECISION_FEATURE_CONTRACT_MISMATCH", "Decision does not bind to a frozen manifest feature contract.", index=index, feature_contract_id=contract_id, expected=expected_contract_hash, actual=contract_hash)
    raw_features = record.get("features")
    if is_stopping_decision and isinstance(raw_features, dict):
        unexpected = sorted(set(raw_features) - set(manifest.feature_contracts.get(contract_id, ())))
        forbidden = sorted(set(raw_features) & FORBIDDEN_FEATURE_COLUMNS)
        if unexpected or forbidden:
            findings.add("rejected", "DECISION_FEATURE_PAYLOAD_INVALID", "Decision feature payload contains fields outside frozen contract or forbidden fields.", index=index, unexpected=unexpected, forbidden=forbidden)
    return DecisionRecord(
        decision_id=decision_id,
        barrier_id=barrier_id,
        task_id=task_id,
        task_hash=str(task_hash),
        replicate_id=replicate_id,
        step=step,
        stage=stage,
        action=action,
        selected_alias=selected_alias,
        selected_event_id=selected_event_id,
        selected_answer_hash=selected_answer_hash,
        decision_at=decision_at,
        decision_monotonic_ns=decision_ns,
        consumed_event_ids=consumed_ids,
        consumed_prompt_tokens=consumed_prompt,
        consumed_completion_tokens=consumed_completion,
        consumed_total_tokens=consumed_total,
        feature_contract_id=contract_id,
        feature_contract_sha256=str(contract_hash),
        is_stopping_decision=is_stopping_decision,
    )


def read_barrier_commits(
    root: Path,
    manifest: ManifestState,
    findings: Findings,
) -> dict[str, dict[str, Any]]:
    """Read and cryptographically validate atomic barrier-commit payloads.

    JSONL ledgers are convenient derived views, but cannot on their own prove
    that a complete peer barrier was durable before selection.  A committee
    schema therefore treats each committed payload as canonical and later
    checks that the JSONLs are exact derivatives of it.
    """

    commits_dir = root / COMMIT_DIRECTORY_NAME
    if not commits_dir.is_dir():
        category = "rejected" if manifest.status == "complete" else "incomplete"
        if manifest.require_commit_source:
            findings.add(category, "BARRIER_COMMITS_DIRECTORY_MISSING", "Committee protocol requires a barrier_commits directory as its atomic canonical source.", path=display_path(commits_dir, root))
        return {}
    commits: dict[str, dict[str, Any]] = {}
    for index, path in enumerate(sorted(commits_dir.glob("*.json")), start=1):
        payload = read_json_object(path, root, findings, label="barrier_commit", required=True)
        if payload is None:
            continue
        payload_hash = record_hash_valid(payload, "payload_sha256", findings, kind="barrier_commit", index=index)
        if payload.get("schema_version") != manifest.raw.get("schema_version"):
            findings.add("rejected", "BARRIER_COMMIT_SCHEMA_MISMATCH", "Barrier commit schema does not match its protocol manifest.", path=display_path(path, root), schema_version=payload.get("schema_version"))
        validate_record_manifest_binding(payload, manifest, findings, kind="barrier_commit", index=index)
        barrier_id = as_nonempty_string(payload.get("barrier_id"))
        if barrier_id is None:
            findings.add("invalid", "BARRIER_COMMIT_ID_MISSING", "Barrier commit must contain barrier_id.", path=display_path(path, root))
            continue
        if path.stem != barrier_id:
            findings.add("rejected", "BARRIER_COMMIT_FILENAME_MISMATCH", "Barrier commit filename must equal its barrier_id.", path=display_path(path, root), barrier_id=barrier_id)
        if barrier_id in commits:
            findings.add("rejected", "BARRIER_COMMIT_ID_DUPLICATE", "Multiple atomic commits use the same barrier_id.", barrier_id=barrier_id)
            continue
        barrier = payload.get("barrier")
        events = payload.get("events")
        decision = payload.get("decision")
        if not isinstance(barrier, dict) or not isinstance(events, list) or not isinstance(decision, dict):
            findings.add("invalid", "BARRIER_COMMIT_PAYLOAD_INVALID", "Barrier commit must contain object barrier, list events, and object decision.", barrier_id=barrier_id)
            continue
        if barrier.get("barrier_id") != barrier_id or decision.get("barrier_id") != barrier_id:
            findings.add("rejected", "BARRIER_COMMIT_SCOPE_MISMATCH", "Commit barrier/decision IDs must match the enclosing barrier_id.", barrier_id=barrier_id)
        record_hash_valid(barrier, "barrier_sha256", findings, kind="barrier", index=index)
        record_hash_valid(decision, "decision_sha256", findings, kind="decision", index=index)
        validate_record_manifest_binding(barrier, manifest, findings, kind="barrier", index=index)
        validate_record_manifest_binding(decision, manifest, findings, kind="decision", index=index)
        event_ids: set[str] = set()
        event_aliases: set[str] = set()
        for event_index, event in enumerate(events, start=1):
            if not isinstance(event, dict):
                findings.add("invalid", "BARRIER_COMMIT_EVENT_NOT_OBJECT", "Barrier commit event must be a JSON object.", barrier_id=barrier_id, index=event_index)
                continue
            check_runtime_label_fields(event, findings, kind="event", index=event_index)
            event_hash = record_hash_valid(event, "event_sha256", findings, kind="event", index=event_index)
            validate_record_manifest_binding(event, manifest, findings, kind="event", index=event_index)
            event_id = as_nonempty_string(event.get("event_id"))
            alias = as_nonempty_string(event.get("model_alias"))
            if event_id is None or alias is None:
                findings.add("invalid", "BARRIER_COMMIT_EVENT_IDENTITY_MISSING", "Atomic event must identify event_id and model_alias.", barrier_id=barrier_id, index=event_index)
                continue
            if event.get("barrier_id") != barrier_id:
                findings.add("rejected", "BARRIER_COMMIT_EVENT_SCOPE_MISMATCH", "Atomic event belongs to a different barrier.", barrier_id=barrier_id, event_id=event_id)
            if event_id in event_ids or alias in event_aliases:
                findings.add("rejected", "BARRIER_COMMIT_EVENT_DUPLICATE", "Atomic barrier must contain one unique event per event ID and alias.", barrier_id=barrier_id, event_id=event_id, alias=alias)
            event_ids.add(event_id)
            event_aliases.add(alias)
            if event_hash is None:
                continue
        post_events = payload.get("post_events", [])
        if not isinstance(post_events, list):
            findings.add("invalid", "BARRIER_COMMIT_POST_EVENTS_INVALID", "post_events must be a list when supplied.", barrier_id=barrier_id)
            post_events = []
        post_event_ids: set[str] = set()
        for post_index, event in enumerate(post_events, start=1):
            if not isinstance(event, dict):
                findings.add("invalid", "BARRIER_COMMIT_POST_EVENT_NOT_OBJECT", "Atomic post-event must be a JSON object.", barrier_id=barrier_id, index=post_index)
                continue
            check_runtime_label_fields(event, findings, kind="event", index=post_index)
            event_hash = record_hash_valid(event, "event_sha256", findings, kind="event", index=post_index)
            validate_record_manifest_binding(event, manifest, findings, kind="event", index=post_index)
            event_id = as_nonempty_string(event.get("event_id"))
            if event_id is None or event_id in event_ids or event_id in post_event_ids:
                findings.add("rejected", "BARRIER_COMMIT_POST_EVENT_DUPLICATE", "Atomic post-events must have unique IDs distinct from main events.", barrier_id=barrier_id, event_id=event_id)
                continue
            post_event_ids.add(event_id)
            if event.get("barrier_id") != barrier_id or event.get("generation_kind") != "verifier":
                findings.add("rejected", "BARRIER_COMMIT_POST_EVENT_SCOPE_INVALID", "Atomic post-event must be a verifier event at its enclosing barrier.", barrier_id=barrier_id, event_id=event_id)
            _ = event_hash
        hashes = payload.get("event_hashes")
        if not isinstance(hashes, dict) or not hashes:
            findings.add("invalid", "BARRIER_COMMIT_EVENT_HASHES_MISSING", "Atomic barrier commit must include event_hashes.", barrier_id=barrier_id)
        elif not all(isinstance(key, str) and is_sha256(value) for key, value in hashes.items()):
            findings.add("invalid", "BARRIER_COMMIT_EVENT_HASHES_INVALID", "Atomic commit event_hashes must contain string keys and SHA-256 values.", barrier_id=barrier_id)
        elif set(hashes) != event_ids and set(hashes) != event_aliases:
            findings.add("rejected", "BARRIER_COMMIT_EVENT_HASH_COVERAGE_MISMATCH", "Atomic commit event_hashes must cover exactly its events by ID or alias.", barrier_id=barrier_id)
        else:
            for event in events:
                if not isinstance(event, dict):
                    continue
                key = event.get("event_id") if set(hashes) == event_ids else event.get("model_alias")
                if hashes.get(key) != event.get("event_sha256"):
                    findings.add("rejected", "BARRIER_COMMIT_EVENT_HASH_MISMATCH", "Atomic commit event hash does not match its event record.", barrier_id=barrier_id, event_id=event.get("event_id"))
        post_hashes = payload.get("post_event_hashes", {})
        if post_events or manifest.verifier_spec is not None:
            if not isinstance(post_hashes, dict) or set(post_hashes) != post_event_ids:
                findings.add("rejected", "BARRIER_COMMIT_POST_EVENT_HASH_COVERAGE_MISMATCH", "post_event_hashes must cover exactly the atomic verifier post-events.", barrier_id=barrier_id)
            else:
                for event in post_events:
                    if isinstance(event, dict) and post_hashes.get(event.get("event_id")) != event.get("event_sha256"):
                        findings.add("rejected", "BARRIER_COMMIT_POST_EVENT_HASH_MISMATCH", "Atomic post-event hash does not match its verifier event.", barrier_id=barrier_id, event_id=event.get("event_id"))
        check_runtime_label_fields(barrier, findings, kind="barrier", index=index)
        check_runtime_label_fields(decision, findings, kind="decision", index=index)
        # The hash is otherwise unused here; retaining it makes an explicit
        # payload integrity failure visible even if a caller only asks for
        # derived-ledger verification.
        _ = payload_hash
        commits[barrier_id] = payload
    if manifest.require_commit_source and not commits and manifest.status == "complete":
        findings.add("rejected", "BARRIER_COMMITS_EMPTY", "Terminal committee protocol has no atomic barrier commits.")
    return commits


def _records_by_identifier(
    records: Sequence[Mapping[str, Any]],
    identifier: str,
    findings: Findings,
    *,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        value = as_nonempty_string(record.get(identifier))
        if value is None:
            continue
        if value in result:
            findings.add("rejected", f"DERIVED_{label.upper()}_ID_DUPLICATE", f"Derived {label} ledger contains duplicate {identifier}.", identifier=value, index=index)
            continue
        result[value] = record
    return result


def validate_derived_ledgers_against_commits(
    raw_events: Sequence[Mapping[str, Any]],
    raw_barriers: Sequence[Mapping[str, Any]],
    raw_decisions: Sequence[Mapping[str, Any]],
    commits: Mapping[str, Mapping[str, Any]],
    manifest: ManifestState,
    findings: Findings,
) -> None:
    """Require derived JSONLs to be exact, lossless views of atomic commits."""

    if not commits:
        return
    ledger_events = _records_by_identifier(raw_events, "event_id", findings, label="events")
    ledger_barriers = _records_by_identifier(raw_barriers, "barrier_id", findings, label="barriers")
    ledger_decisions = _records_by_identifier(raw_decisions, "decision_id", findings, label="decisions")
    committed_events: dict[str, Mapping[str, Any]] = {}
    committed_decisions: dict[str, Mapping[str, Any]] = {}
    for barrier_id, commit in commits.items():
        barrier = commit.get("barrier")
        decision = commit.get("decision")
        events = commit.get("events")
        post_events = commit.get("post_events", [])
        if isinstance(barrier, Mapping):
            observed = ledger_barriers.get(barrier_id)
            if observed is None or canonical_json_bytes(observed) != canonical_json_bytes(barrier):
                findings.add("rejected", "DERIVED_BARRIER_LEDGER_MISMATCH", "barriers.jsonl is not an exact derivative of its atomic barrier commit.", barrier_id=barrier_id)
        if isinstance(decision, Mapping):
            identifier = as_nonempty_string(decision.get("decision_id"))
            if identifier is not None:
                committed_decisions[identifier] = decision
        if isinstance(events, list):
            for event in events:
                if isinstance(event, Mapping):
                    identifier = as_nonempty_string(event.get("event_id"))
                    if identifier is not None:
                        committed_events[identifier] = event
        if isinstance(post_events, list):
            for event in post_events:
                if isinstance(event, Mapping):
                    identifier = as_nonempty_string(event.get("event_id"))
                    if identifier is not None:
                        committed_events[identifier] = event
    if set(ledger_events) != set(committed_events):
        findings.add("rejected", "DERIVED_EVENT_LEDGER_COVERAGE_MISMATCH", "events.jsonl IDs differ from atomic barrier commits.", ledger_count=len(ledger_events), commit_count=len(committed_events))
    for identifier, committed in committed_events.items():
        observed = ledger_events.get(identifier)
        if observed is None or canonical_json_bytes(observed) != canonical_json_bytes(committed):
            findings.add("rejected", "DERIVED_EVENT_LEDGER_MISMATCH", "events.jsonl record differs from its atomic commit event.", event_id=identifier)
    if set(ledger_decisions) != set(committed_decisions):
        findings.add("rejected", "DERIVED_DECISION_LEDGER_COVERAGE_MISMATCH", "decisions.jsonl IDs differ from atomic barrier commits.", ledger_count=len(ledger_decisions), commit_count=len(committed_decisions))
    for identifier, committed in committed_decisions.items():
        observed = ledger_decisions.get(identifier)
        if observed is None or canonical_json_bytes(observed) != canonical_json_bytes(committed):
            findings.add("rejected", "DERIVED_DECISION_LEDGER_MISMATCH", "decisions.jsonl record differs from its atomic commit decision.", decision_id=identifier)


def validate_verifier_prompt_provenance(
    verifier_event: EventRecord,
    barrier: BarrierRecord,
    barrier_events: Sequence[EventRecord],
    manifest: ManifestState,
    findings: Findings,
) -> None:
    """Rebuild the frozen verifier prompt and bind its recorded hash.

    A template hash alone is insufficient: without rebuilding the request, a
    collector could inject an answer, an alias, or arbitrary unregistered text
    and merely record the hash of that different string.  The public task
    prompt and committed current-barrier thoughts are enough to reproduce both
    registered variants exactly.
    """

    spec = manifest.verifier_spec
    task = manifest.task_records.get(barrier.task_id)
    if spec is None or task is None or task.public_prompt is None:
        findings.add(
            "rejected",
            "VERIFIER_PUBLIC_PROMPT_UNAVAILABLE",
            "Verifier-enabled protocol must retain a public prompt whose hash matches the task record.",
            barrier_id=barrier.barrier_id,
            event_id=verifier_event.event_id,
        )
        return
    variant = verifier_event.verifier_variant
    try:
        if variant == PROMPT_OPTIONS_ONLY_V1:
            rendered = render_verifier_prompt(task.public_prompt, variant=variant)
        elif variant == ANONYMOUS_RATIONALE_V1:
            seed_material = "|".join(
                [
                    manifest.protocol_id,
                    barrier.task_hash,
                    barrier.replicate_id,
                    str(barrier.step),
                    verifier_event.model_alias,
                    variant,
                ]
            )
            rendered = render_verifier_prompt(
                task.public_prompt,
                variant=variant,
                thoughts=[event.thought for event in barrier_events],
                seed_material=seed_material,
                rationale_max_chars=int(spec["rationale_max_chars"]),
            )
        else:
            raise ForcedChoiceVerifierError(f"Unsupported verifier variant {variant!r}.")
    except (ForcedChoiceVerifierError, KeyError, TypeError, ValueError) as error:
        findings.add(
            "rejected",
            "VERIFIER_PROMPT_RECONSTRUCTION_FAILED",
            "Auditor could not reconstruct the frozen verifier prompt from public/current-barrier inputs.",
            barrier_id=barrier.barrier_id,
            event_id=verifier_event.event_id,
            error=str(error),
        )
        return
    expected_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    if verifier_event.verifier_prompt_sha256 != expected_hash:
        findings.add(
            "rejected",
            "VERIFIER_PROMPT_HASH_MISMATCH",
            "Verifier prompt hash differs from the exact registered public/current-barrier reconstruction.",
            barrier_id=barrier.barrier_id,
            event_id=verifier_event.event_id,
            expected=expected_hash,
            actual=verifier_event.verifier_prompt_sha256,
        )


def validate_barrier_links(
    barriers: Sequence[BarrierRecord],
    events: Mapping[str, EventRecord],
    raw_barriers: Sequence[Mapping[str, Any]],
    manifest: ManifestState,
    findings: Findings,
    *,
    commits_by_barrier: Mapping[str, Mapping[str, Any]] | None = None,
) -> set[str]:
    referenced_main_events: set[str] = set()
    raw_by_id = {str(raw.get("barrier_id")): raw for raw in raw_barriers if isinstance(raw, Mapping)}
    roster_aliases = set(manifest.roster_by_alias)
    seen_keys: set[tuple[str, str, int]] = set()
    for barrier in barriers:
        key = (barrier.task_id, barrier.replicate_id, barrier.step)
        if key in seen_keys:
            findings.add("rejected", "BARRIER_SCOPE_DUPLICATE", "Multiple barrier records cover the same task/replicate/step.", task_id=barrier.task_id, replicate_id=barrier.replicate_id, step=barrier.step)
        seen_keys.add(key)
        raw = raw_by_id.get(barrier.barrier_id, {})
        commit = (commits_by_barrier or {}).get(barrier.barrier_id)
        raw_hashes = raw.get("event_hashes") if isinstance(raw, Mapping) else None
        if raw_hashes is None and isinstance(commit, Mapping):
            raw_hashes = commit.get("event_hashes")
        if not isinstance(raw_hashes, Mapping):
            raw_hashes = {}
        missing_events = [event_id for event_id in barrier.main_event_ids if event_id not in events]
        if missing_events:
            findings.add("rejected", "BARRIER_REFERENCES_MISSING_EVENT", "Barrier references one or more missing main events.", barrier_id=barrier.barrier_id, event_ids=missing_events)
            continue
        barrier_events = [events[event_id] for event_id in barrier.main_event_ids]
        referenced_main_events.update(barrier.main_event_ids)
        if any(event.generation_kind != "main" for event in barrier_events):
            findings.add("rejected", "BARRIER_NON_MAIN_EVENT_REFERENCED", "Barrier main_event_ids may contain only main generation events.", barrier_id=barrier.barrier_id)
        observed_aliases = [event.model_alias for event in barrier_events]
        if set(observed_aliases) != roster_aliases or len(observed_aliases) != len(set(observed_aliases)):
            findings.add("rejected", "BARRIER_EVENT_ROSTER_MISMATCH", "Barrier main events do not contain exactly one event per frozen roster alias.", barrier_id=barrier.barrier_id, expected=sorted(roster_aliases), actual=sorted(observed_aliases))
        hash_keys = set(raw_hashes)
        if hash_keys != set(barrier.main_event_ids) and hash_keys != set(observed_aliases):
            findings.add(
                "rejected",
                "BARRIER_EVENT_HASH_COVERAGE_MISMATCH",
                "Barrier event_hashes must cover exactly its main event IDs or aliases.",
                barrier_id=barrier.barrier_id,
                keys=sorted(hash_keys),
                event_ids=sorted(barrier.main_event_ids),
                aliases=sorted(observed_aliases),
            )
        for event in barrier_events:
            if (event.barrier_id, event.task_id, event.task_hash, event.replicate_id, event.step) != (barrier.barrier_id, barrier.task_id, barrier.task_hash, barrier.replicate_id, barrier.step):
                findings.add("rejected", "BARRIER_EVENT_SCOPE_MISMATCH", "A main event differs from its barrier's task/replicate/step scope.", barrier_id=barrier.barrier_id, event_id=event.event_id)
            committed_hash = raw_hashes.get(event.event_id, raw_hashes.get(event.model_alias))
            if committed_hash != event.event_sha256:
                findings.add("rejected", "BARRIER_EVENT_HASH_MISMATCH", "Barrier event_hashes does not match the referenced event record.", barrier_id=barrier.barrier_id, event_id=event.event_id, expected=raw_hashes.get(event.event_id), actual=event.event_sha256)
            if event.started_at < barrier.opened_at or event.started_monotonic_ns < barrier.opened_monotonic_ns:
                findings.add("rejected", "EVENT_PRECEDES_BARRIER_OPEN", "A main event starts before its barrier opens.", barrier_id=barrier.barrier_id, event_id=event.event_id)
            if event.completed_at > barrier.closed_at or event.completed_monotonic_ns > barrier.closed_monotonic_ns:
                findings.add("rejected", "EVENT_FOLLOWS_BARRIER_CLOSE", "A main event completes after its barrier closes.", barrier_id=barrier.barrier_id, event_id=event.event_id)
        expected_prompt = sum(event.prompt_tokens for event in barrier_events)
        expected_completion = sum(event.completion_tokens for event in barrier_events)
        expected_total = sum(event.total_tokens for event in barrier_events)
        actual_tokens = (barrier.fleet_prompt_tokens, barrier.fleet_completion_tokens, barrier.fleet_total_tokens)
        if actual_tokens != (expected_prompt, expected_completion, expected_total):
            findings.add("rejected", "BARRIER_FLEET_TOKEN_MISMATCH", "Barrier fleet token totals do not equal the sum of its main events.", barrier_id=barrier.barrier_id, expected={"prompt": expected_prompt, "completion": expected_completion, "total": expected_total}, actual={"prompt": barrier.fleet_prompt_tokens, "completion": barrier.fleet_completion_tokens, "total": barrier.fleet_total_tokens})
        verifier_events = [event for event in events.values() if event.generation_kind == "verifier" and event.barrier_id == barrier.barrier_id]
        if manifest.verifier_spec is None:
            if verifier_events:
                findings.add("rejected", "VERIFIER_EVENTS_WITHOUT_SPEC", "Verifier events appear in a protocol without a frozen verifier_spec.", barrier_id=barrier.barrier_id)
        else:
            expected_variants = set(manifest.verifier_spec.get("variants", []))
            commit_post = (commit or {}).get("post_events", []) if isinstance(commit, Mapping) else []
            raw_variants: dict[str, Mapping[str, Any]] = {}
            if isinstance(commit_post, list):
                for raw_event in commit_post:
                    if isinstance(raw_event, Mapping):
                        raw_variant = as_nonempty_string(raw_event.get("verifier_variant"))
                        raw_id = as_nonempty_string(raw_event.get("event_id"))
                        if raw_variant is None or raw_id is None or raw_variant in raw_variants:
                            findings.add("rejected", "VERIFIER_VARIANT_DUPLICATE_OR_MISSING", "Every barrier must have one uniquely identified verifier event per frozen variant.", barrier_id=barrier.barrier_id)
                            continue
                        raw_variants[raw_variant] = raw_event
            observed_variants = set(raw_variants)
            if observed_variants != expected_variants or len(verifier_events) != len(expected_variants):
                findings.add(
                    "rejected",
                    "VERIFIER_VARIANT_COVERAGE_MISMATCH",
                    "A verifier-enabled barrier must commit exactly one post-event for every frozen variant.",
                    barrier_id=barrier.barrier_id,
                    expected=sorted(expected_variants),
                    observed=sorted(observed_variants),
                )
            for event in verifier_events:
                if (
                    (event.task_id, event.task_hash, event.replicate_id, event.step)
                    != (barrier.task_id, barrier.task_hash, barrier.replicate_id, barrier.step)
                    or event.model_alias != manifest.verifier_spec.get("ledger_alias")
                ):
                    findings.add("rejected", "VERIFIER_EVENT_SCOPE_MISMATCH", "Verifier post-event scope or alias differs from its frozen main barrier/spec.", barrier_id=barrier.barrier_id, event_id=event.event_id)
                if event.started_at <= barrier.closed_at or event.started_monotonic_ns <= barrier.closed_monotonic_ns:
                    findings.add("rejected", "VERIFIER_EVENT_PRECEDES_BARRIER_CLOSE", "Verifier must start strictly after the complete main barrier closed.", barrier_id=barrier.barrier_id, event_id=event.event_id)
                if event.triggered_by_decision_id is not None:
                    findings.add("rejected", "VERIFIER_EVENT_UNEXPECTED_TRIGGER", "Mandatory collect-only verifier events must not claim a pre-query decision trigger.", barrier_id=barrier.barrier_id, event_id=event.event_id)
                validate_verifier_prompt_provenance(event, barrier, barrier_events, manifest, findings)
    main_event_ids = {event_id for event_id, event in events.items() if event.generation_kind == "main"}
    unreferenced = sorted(main_event_ids - referenced_main_events)
    if unreferenced:
        findings.add("rejected", "MAIN_EVENTS_WITHOUT_BARRIER", "Main events exist without a completed barrier ledger record.", count=len(unreferenced), examples=unreferenced[:10])
    return referenced_main_events


def validate_decision_links(
    decisions: Sequence[DecisionRecord],
    events: Mapping[str, EventRecord],
    barriers: Mapping[str, BarrierRecord],
    manifest: ManifestState,
    findings: Findings,
) -> set[tuple[str, str, int]]:
    final_scopes: set[tuple[str, str, int]] = set()
    decision_by_id = {decision.decision_id: decision for decision in decisions}
    for decision in decisions:
        barrier = barriers.get(decision.barrier_id)
        if barrier is None:
            findings.add("rejected", "DECISION_REFERENCES_MISSING_BARRIER", "Decision references a barrier absent from the completed barrier ledger.", decision_id=decision.decision_id, barrier_id=decision.barrier_id)
            continue
        if (decision.task_id, decision.task_hash, decision.replicate_id, decision.step) != (barrier.task_id, barrier.task_hash, barrier.replicate_id, barrier.step):
            findings.add("rejected", "DECISION_BARRIER_SCOPE_MISMATCH", "Decision scope differs from its referenced barrier.", decision_id=decision.decision_id, barrier_id=decision.barrier_id)
        if decision.decision_at < barrier.closed_at or decision.decision_monotonic_ns < barrier.closed_monotonic_ns:
            findings.add("rejected", "DECISION_PRECEDES_BARRIER_CLOSE", "Decision occurs before all frozen roster main events completed the barrier.", decision_id=decision.decision_id, barrier_id=decision.barrier_id, decision_at=utc_display(decision.decision_at), barrier_closed_at=utc_display(barrier.closed_at))
        missing_events = [event_id for event_id in decision.consumed_event_ids if event_id not in events]
        if missing_events:
            findings.add("rejected", "DECISION_CONSUMES_MISSING_EVENT", "Decision consumes event IDs absent from event ledger.", decision_id=decision.decision_id, event_ids=missing_events)
            continue
        consumed_events = [events[event_id] for event_id in decision.consumed_event_ids]
        selected_event = events.get(decision.selected_event_id)
        if selected_event is None:
            findings.add("rejected", "DECISION_SELECTED_EVENT_MISSING", "Decision selected_event_id is absent from the event ledger.", decision_id=decision.decision_id, selected_event_id=decision.selected_event_id)
        elif selected_event.barrier_id != decision.barrier_id or selected_event.model_alias != decision.selected_alias:
            findings.add("rejected", "DECISION_SELECTED_EVENT_MISMATCH", "Decision selected event must belong to its barrier and selected frozen alias.", decision_id=decision.decision_id, selected_event_id=decision.selected_event_id, selected_alias=decision.selected_alias)
        if any(event.barrier_id != decision.barrier_id for event in consumed_events):
            findings.add("rejected", "DECISION_CONSUMES_OTHER_BARRIER", "Decision consumes an event from another barrier, which would leak future/foreign information.", decision_id=decision.decision_id)
        if any(event.completed_at > decision.decision_at or event.completed_monotonic_ns > decision.decision_monotonic_ns for event in consumed_events):
            findings.add("rejected", "DECISION_CONSUMES_UNAVAILABLE_EVENT", "Decision consumes an event that completed after the decision timestamp.", decision_id=decision.decision_id)
        expected_prompt = sum(event.prompt_tokens for event in consumed_events)
        expected_completion = sum(event.completion_tokens for event in consumed_events)
        expected_total = sum(event.total_tokens for event in consumed_events)
        if (decision.consumed_prompt_tokens, decision.consumed_completion_tokens, decision.consumed_total_tokens) != (expected_prompt, expected_completion, expected_total):
            findings.add("rejected", "DECISION_TOKEN_ACCOUNTING_MISMATCH", "Decision token totals do not equal its unique consumed event ledger entries.", decision_id=decision.decision_id, expected={"prompt": expected_prompt, "completion": expected_completion, "total": expected_total}, actual={"prompt": decision.consumed_prompt_tokens, "completion": decision.consumed_completion_tokens, "total": decision.consumed_total_tokens})
        consumed_main = {event.event_id for event in consumed_events if event.generation_kind == "main"}
        if consumed_main != set(barrier.main_event_ids):
            findings.add("rejected", "DECISION_FULL_ROSTER_NOT_CONSUMED", "Strict full-roster decision must consume exactly every main event at its barrier.", decision_id=decision.decision_id, expected=sorted(barrier.main_event_ids), actual=sorted(consumed_main))
        if decision.is_stopping_decision:
            columns = manifest.feature_contracts.get(decision.feature_contract_id, ())
            consumed_kinds = {event.generation_kind for event in consumed_events}
            required_post_kinds = {kind for prefix, kind in POST_QUERY_PREFIXES.items() if any(column.startswith(prefix) for column in columns)}
            missing_post = sorted(required_post_kinds - consumed_kinds)
            if missing_post:
                findings.add("rejected", "DECISION_FREE_POST_QUERY_SIGNAL", "Feature contract uses K2/verifier signal without consuming a separately timestamped and costed post-query event.", decision_id=decision.decision_id, missing_generation_kinds=missing_post)
        if decision.is_stopping_decision and decision.stage == "final":
            scope = (decision.task_id, decision.replicate_id, decision.step)
            if scope in final_scopes:
                findings.add("rejected", "DECISION_FINAL_SCOPE_DUPLICATE", "Multiple final decisions exist for the same task/replicate/step.", task_id=decision.task_id, replicate_id=decision.replicate_id, step=decision.step)
            final_scopes.add(scope)
            if decision.action not in TERMINAL_ACTIONS:
                findings.add("rejected", "DECISION_FINAL_ACTION_NOT_TERMINAL", "Final decision action must be stop or continue.", decision_id=decision.decision_id, action=decision.action)
        elif not decision.is_stopping_decision:
            scope = (decision.task_id, decision.replicate_id, decision.step)
            if scope in final_scopes:
                findings.add("rejected", "OBSERVATIONAL_SELECTION_SCOPE_DUPLICATE", "Multiple answer-selection records exist for the same task/replicate/step.", task_id=decision.task_id, replicate_id=decision.replicate_id, step=decision.step)
            final_scopes.add(scope)
    for event in events.values():
        if event.triggered_by_decision_id is None:
            continue
        parent = decision_by_id.get(event.triggered_by_decision_id)
        if parent is None:
            findings.add("rejected", "POST_EVENT_PARENT_DECISION_MISSING", "Post-query event references a missing trigger decision.", event_id=event.event_id, parent_decision_id=event.triggered_by_decision_id)
            continue
        if event.started_at < parent.decision_at or event.started_monotonic_ns < parent.decision_monotonic_ns:
            findings.add("rejected", "POST_EVENT_PRECEDES_TRIGGER", "K2/verifier event starts before its triggering decision.", event_id=event.event_id, parent_decision_id=parent.decision_id)
    return final_scopes


def expected_scopes(manifest: ManifestState) -> set[tuple[str, str, int]]:
    return {
        (task_id, replicate_id, step)
        for task_id in manifest.task_records
        for replicate_id in manifest.replicate_ids
        for step in range(1, manifest.max_steps + 1)
    }


def audit_protocol(
    output_dir: Path,
    *,
    manifest_path: str | Path | None = None,
    events_path: str | Path | None = None,
    barriers_path: str | Path | None = None,
    decisions_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit one prospective protocol directory without modifying it."""

    root = output_dir.resolve()
    findings = Findings()
    report: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "output_dir": str(root),
        "read_only": True,
        "artifacts": {},
    }
    if not root.is_dir():
        findings.add("invalid", "OUTPUT_DIRECTORY_MISSING", "Audited output directory does not exist.", path=str(root))
        report["findings"] = findings.as_dicts()
        report.update({"assessment": "invalid", "exit_code": EXIT_INVALID})
        return report

    manifest_file = discover_file(root, manifest_path, MANIFEST_CANDIDATES)
    if manifest_file is None:
        findings.add("invalid", "MANIFEST_PATH_INVALID", "Requested manifest path escapes the audited output directory.")
        report["findings"] = findings.as_dicts()
        report.update({"assessment": "invalid", "exit_code": EXIT_INVALID})
        return report
    report["artifacts"]["manifest"] = {"path": display_path(manifest_file, root), "exists": manifest_file.is_file()}
    manifest = validate_manifest(root, manifest_file, findings)
    if manifest is None:
        report["findings"] = findings.as_dicts()
        report.update({"assessment": "invalid", "exit_code": EXIT_INVALID})
        return report
    report["manifest"] = {
        "schema_version": manifest.raw.get("schema_version"),
        "protocol_id": manifest.protocol_id,
        "status": manifest.status,
        "canonical_sha256": manifest.canonical_sha256,
        "file_sha256": manifest.file_sha256,
        "roster_aliases": [member.alias for member in manifest.roster],
        "replicate_ids": sorted(manifest.replicate_ids),
        "seed_scope": manifest.seed_scope,
        "max_steps": manifest.max_steps,
        "task_count": len(manifest.task_records),
        "sealed_gold_labels": manifest.sealed_gold_labels,
        "policy_spec_sha256": manifest.policy_spec_sha256,
        "feature_contracts": {key: list(value) for key, value in manifest.feature_contracts.items()},
        "confirmation_eligible": manifest.confirmation_eligible,
        "require_commit_source": manifest.require_commit_source,
        "policy_type": manifest.policy_type,
    }

    commits = read_barrier_commits(root, manifest, findings)
    report["artifacts"]["barrier_commits"] = {
        "path": COMMIT_DIRECTORY_NAME,
        "exists": (root / COMMIT_DIRECTORY_NAME).is_dir(),
        "count": len(commits),
    }

    event_file = discover_file(root, events_path, EVENT_CANDIDATES)
    barrier_file = discover_file(root, barriers_path, BARRIER_CANDIDATES)
    decision_file = discover_file(root, decisions_path, DECISION_CANDIDATES)
    if event_file is None or barrier_file is None or decision_file is None:
        findings.add("invalid", "LEDGER_PATH_INVALID", "A requested ledger path escapes the audited output directory.")
        report["findings"] = findings.as_dicts()
        report.update({"assessment": "invalid", "exit_code": EXIT_INVALID})
        return report
    report["artifacts"].update(
        {
            "events": {"path": display_path(event_file, root), "exists": event_file.is_file()},
            "barriers": {"path": display_path(barrier_file, root), "exists": barrier_file.is_file()},
            "decisions": {"path": display_path(decision_file, root), "exists": decision_file.is_file()},
        }
    )
    terminal = manifest.status == "complete"
    ledgers_required = terminal
    raw_events = read_json_records(event_file, root, findings, label="event_ledger", required=ledgers_required)
    raw_barriers = read_json_records(barrier_file, root, findings, label="barrier_ledger", required=ledgers_required)
    decisions_required = terminal or bool(commits)
    raw_decisions = read_json_records(decision_file, root, findings, label="decision_ledger", required=decisions_required)
    if raw_events is None:
        if not terminal:
            findings.add("incomplete", "EVENT_LEDGER_NOT_YET_PRESENT", "Collection is non-terminal and no derived event ledger is present yet.")
            raw_events = []
    if raw_barriers is None:
        if not terminal:
            findings.add("incomplete", "BARRIER_LEDGER_NOT_YET_PRESENT", "Collection is non-terminal and no derived barrier ledger is present yet.")
            raw_barriers = []
    if raw_decisions is None and not decisions_required:
        findings.add("incomplete", "DECISION_LEDGER_NOT_YET_PRESENT", "Collection is non-terminal and no decision ledger is present yet.")
        raw_decisions = []
    if raw_events is None or raw_barriers is None or raw_decisions is None:
        report["findings"] = findings.as_dicts()
        report.update({"assessment": "invalid", "exit_code": EXIT_INVALID})
        return report

    validate_derived_ledgers_against_commits(raw_events, raw_barriers, raw_decisions, commits, manifest, findings)

    events: dict[str, EventRecord] = {}
    seed_index: dict[int, str] = {}
    for index, raw in enumerate(raw_events, start=1):
        event = validate_event(raw, index, manifest, findings)
        if event is None:
            continue
        if event.event_id in events:
            findings.add("rejected", "EVENT_ID_DUPLICATE", "Event ledger contains duplicate event_id values.", event_id=event.event_id)
            continue
        if event.generation_seed in seed_index:
            findings.add("rejected", "EVENT_GENERATION_SEED_REUSED", "Per-event generation seed is reused; independent rollouts are not auditable.", event_id=event.event_id, prior_event_id=seed_index[event.generation_seed], generation_seed=event.generation_seed)
        else:
            seed_index[event.generation_seed] = event.event_id
        events[event.event_id] = event

    barriers: list[BarrierRecord] = []
    barrier_by_id: dict[str, BarrierRecord] = {}
    for index, raw in enumerate(raw_barriers, start=1):
        commit = commits.get(str(raw.get("barrier_id", "")))
        barrier = validate_barrier(raw, index, manifest, findings, commit=commit)
        if barrier is None:
            continue
        if barrier.barrier_id in barrier_by_id:
            findings.add("rejected", "BARRIER_ID_DUPLICATE", "Barrier ledger contains duplicate barrier_id values.", barrier_id=barrier.barrier_id)
            continue
        barriers.append(barrier)
        barrier_by_id[barrier.barrier_id] = barrier
    validate_barrier_links(barriers, events, raw_barriers, manifest, findings, commits_by_barrier=commits)

    decisions: list[DecisionRecord] = []
    decision_ids: set[str] = set()
    for index, raw in enumerate(raw_decisions or [], start=1):
        decision = validate_decision(raw, index, manifest, findings)
        if decision is None:
            continue
        if decision.decision_id in decision_ids:
            findings.add("rejected", "DECISION_ID_DUPLICATE", "Decision ledger contains duplicate decision_id values.", decision_id=decision.decision_id)
            continue
        decision_ids.add(decision.decision_id)
        decisions.append(decision)
    final_decision_scopes = validate_decision_links(decisions, events, barrier_by_id, manifest, findings)

    observed_barrier_scopes = {(barrier.task_id, barrier.replicate_id, barrier.step) for barrier in barriers}
    expected = expected_scopes(manifest)
    missing_barriers = sorted(expected - observed_barrier_scopes)
    extra_barriers = sorted(observed_barrier_scopes - expected)
    if extra_barriers:
        findings.add("rejected", "BARRIER_SCOPE_OUTSIDE_TASK_MANIFEST", "Barrier ledger contains scope not declared in frozen task/replicate/step manifest.", count=len(extra_barriers), examples=extra_barriers[:10])
    if missing_barriers:
        findings.add("incomplete", "BARRIER_COVERAGE_INCOMPLETE", "Not every frozen task/replicate/step has a complete barrier record.", expected=len(expected), actual=len(observed_barrier_scopes), missing_count=len(missing_barriers), examples=missing_barriers[:10])
    missing_final = sorted(observed_barrier_scopes - final_decision_scopes)
    extra_final = sorted(final_decision_scopes - observed_barrier_scopes)
    if extra_final:
        findings.add("rejected", "DECISION_SCOPE_WITHOUT_BARRIER", "Final decision exists for a scope without a completed barrier.", count=len(extra_final), examples=extra_final[:10])
    if missing_final:
        category = "rejected" if manifest.status == "complete" else "incomplete"
        findings.add(category, "FINAL_DECISION_COVERAGE_INCOMPLETE", "Not every completed barrier has exactly one final stop/continue decision.", completed_barriers=len(observed_barrier_scopes), final_decisions=len(final_decision_scopes), missing_count=len(missing_final), examples=missing_final[:10])
    if manifest.status == "complete" and missing_barriers:
        findings.add("rejected", "TERMINAL_MANIFEST_WITH_INCOMPLETE_BARRIERS", "Manifest is terminal but required frozen barrier coverage is incomplete.")
    if manifest.status not in {"complete", "collecting", "initialized", "failed"}:
        findings.add("invalid", "MANIFEST_STATUS_UNRECOGNIZED", "Manifest status must be initialized, collecting, complete, or failed.", status=manifest.status)
    if manifest.status != "complete":
        findings.add("incomplete", "MANIFEST_NOT_TERMINAL", "Manifest is not terminal; no final deployable claim is available.", status=manifest.status)
    if not manifest.confirmation_eligible:
        findings.add(
            "incomplete",
            "OBSERVATIONAL_PROTOCOL_NOT_CONFIRMATORY",
            "This is a structurally audited full-horizon observational collector, not a frozen deployed stopping-policy evaluation; it cannot substantiate a prospective stopping claim.",
            collection_mode=manifest.raw.get("collection_mode"),
            stopping_policy=manifest.raw.get("stopping_policy"),
        )

    report["ledgers"] = {
        "events": len(events),
        "barriers": len(barriers),
        "decisions": len(decisions),
        "final_decisions": len(final_decision_scopes),
        "expected_barriers": len(expected),
        "unique_generation_seeds": len(seed_index),
        "atomic_barrier_commits": len(commits),
    }
    report["findings"] = findings.as_dicts()
    if findings.has("invalid"):
        report["assessment"] = "invalid"
        report["exit_code"] = EXIT_INVALID
    elif findings.has("rejected"):
        report["assessment"] = "rejected"
        report["exit_code"] = EXIT_REJECTED
    elif findings.has("incomplete"):
        report["assessment"] = "observational" if not manifest.confirmation_eligible else "incomplete"
        report["exit_code"] = EXIT_INCOMPLETE
    else:
        report["assessment"] = "verified"
        report["exit_code"] = EXIT_VERIFIED
    return report


def human_report(report: Mapping[str, Any]) -> str:
    lines = [
        "PROSPECTIVE BARRIER PROTOCOL AUDIT (read-only)",
        f"Directory: {report.get('output_dir')}",
        f"Assessment: {report.get('assessment')} (exit {report.get('exit_code')})",
    ]
    manifest = report.get("manifest")
    if isinstance(manifest, Mapping):
        lines.append(
            "Protocol: "
            f"{manifest.get('protocol_id')} | status={manifest.get('status')} | "
            f"roster={len(manifest.get('roster_aliases', []))} | "
            f"tasks={manifest.get('task_count')} | replicas={manifest.get('replicate_ids')} | seed_scope={manifest.get('seed_scope')} | "
            f"max_steps={manifest.get('max_steps')}"
        )
    ledgers = report.get("ledgers")
    if isinstance(ledgers, Mapping):
        lines.append(
            "Ledgers: "
            f"events={ledgers.get('events')}, barriers={ledgers.get('barriers')}/{ledgers.get('expected_barriers')}, "
            f"decisions={ledgers.get('decisions')} (final={ledgers.get('final_decisions')}), "
            f"unique_seeds={ledgers.get('unique_generation_seeds')}"
        )
    lines.append("Findings:")
    findings = report.get("findings")
    if not isinstance(findings, list) or not findings:
        lines.append("  [INFO] No findings.")
    else:
        for finding in findings:
            if not isinstance(finding, Mapping):
                continue
            lines.append(f"  [{str(finding.get('severity', 'warning')).upper()}] {finding.get('code')}: {finding.get('message')}")
            evidence = finding.get("evidence")
            if evidence:
                lines.append(f"    Evidence: {json.dumps(evidence, sort_keys=True, default=str)}")
    return "\n".join(lines)


def add_hash(record: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(record)
    result[field] = canonical_json_hash(result)
    return result


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def self_test_report() -> dict[str, Any]:
    """Exercise a verified fixture and a token-accounting rejection fixture."""

    with tempfile.TemporaryDirectory(prefix="prospective_protocol_audit_") as temp_dir:
        root = Path(temp_dir)
        task = {"task_id": "task-001", "task_hash": hashlib.sha256(b"task-001").hexdigest(), "prompt_hash": hashlib.sha256(b"prompt").hexdigest()}
        write_jsonl(root / "prospective_tasks.jsonl", [task])
        task_hash = sha256_file(root / "prospective_tasks.jsonl")
        write_jsonl(root / "sealed_gold_labels.jsonl", [{"task_id": task["task_id"], "gold": "42"}])
        sealed_labels_hash = sha256_file(root / "sealed_gold_labels.jsonl")
        roster = [
            {"alias": "alpha", "model_source": "org/alpha", "model_revision": "a" * 40, "tokenizer_revision": "a" * 40},
            {"alias": "beta", "model_source": "org/beta", "model_revision": "b" * 40, "tokenizer_revision": "b" * 40},
        ]
        normalized = tuple(RosterMember(item["alias"], item["model_source"], item["model_revision"], item["tokenizer_revision"]) for item in roster)
        features = ["step", "confidence", "independent_vote_fraction"]
        manifest = {
            "schema_version": "prospective-barrier-v1",
            "protocol_id": "self-test-protocol",
            "status": "complete",
            "created_at_utc": "2026-07-19T12:00:00Z",
            "code_sha256": "c" * 64,
            "rng_scheme_version": "blake2b-v1",
            "seed_scope": "per_event",
            "replicate_ids": [1],
            "roster": roster,
            "roster_sha256": canonical_roster_hash(normalized),
            "task_manifest": {"path": "prospective_tasks.jsonl", "sha256": task_hash, "count": 1},
            "sealed_gold_labels": {"path": "sealed_gold_labels.jsonl", "sha256": sealed_labels_hash, "count": 1},
            "generation": {"max_steps": 1, "temperature": 0.6},
            "policy": {
                "policy_type": "fixed_stopping",
                "uses_gold_or_correctness": False,
                "peer_visibility": "post_complete_barrier_only",
                "requires_full_roster": True,
                "selection_rule_id": "fixed-leader-alpha-v1",
                "feature_columns": features,
            },
            "hardware": {"gpu": {"name": "test-gpu", "uuid": "GPU-test", "driver_version": "1.0", "vram_mib": 98304}},
        }
        manifest["policy_sha256"] = canonical_json_hash(manifest["policy"])
        (root / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest_sha = canonical_json_hash(manifest)
        events: list[dict[str, Any]] = []
        for index, alias in enumerate(("alpha", "beta"), start=1):
            event = {
                "protocol_id": manifest["protocol_id"], "manifest_sha256": manifest_sha,
                "event_id": f"event-{alias}", "barrier_id": "barrier-1", "task_id": task["task_id"], "task_hash": task["task_hash"],
                "replicate_id": 1, "step": 1, "model_alias": alias,
                "model_revision": roster[index - 1]["model_revision"], "generation_kind": "main", "generation_seed": index, "seed_scope": "per_event",
                "started_at_utc": f"2026-07-19T12:00:0{index}Z", "completed_at_utc": f"2026-07-19T12:00:1{index}Z",
                "started_monotonic_ns": index * 100, "completed_monotonic_ns": index * 100 + 50,
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            }
            events.append(add_hash(event, "event_sha256"))
        write_jsonl(root / "events.jsonl", events)
        barrier = {
            "protocol_id": manifest["protocol_id"], "manifest_sha256": manifest_sha,
            "barrier_id": "barrier-1", "task_id": task["task_id"], "task_hash": task["task_hash"], "replicate_id": 1, "step": 1,
            "main_event_ids": [event["event_id"] for event in events], "expected_aliases": ["alpha", "beta"], "completed_aliases": ["alpha", "beta"],
            "event_hashes": {event["event_id"]: event["event_sha256"] for event in events}, "barrier_complete": True, "status": "complete",
            "opened_at_utc": "2026-07-19T12:00:00Z", "closed_at_utc": "2026-07-19T12:00:20Z", "opened_monotonic_ns": 1, "closed_monotonic_ns": 300,
            "fleet_prompt_tokens": 20, "fleet_completion_tokens": 10, "fleet_total_tokens": 30,
        }
        write_jsonl(root / "barriers.jsonl", [add_hash(barrier, "barrier_sha256")])
        decision = {
            "protocol_id": manifest["protocol_id"], "manifest_sha256": manifest_sha,
            "decision_id": "decision-1", "barrier_id": "barrier-1", "task_id": task["task_id"], "task_hash": task["task_hash"], "replicate_id": 1, "step": 1,
            "decision_stage": "final", "action": "stop", "selected_alias": "alpha", "selected_event_id": "event-alpha", "selected_answer_hash": hashlib.sha256(b"42").hexdigest(),
            "selection_rule_id": manifest["policy"]["selection_rule_id"], "barrier_complete": True,
            "decision_at_utc": "2026-07-19T12:00:21Z", "decision_monotonic_ns": 400,
            "consumed_event_ids": [event["event_id"] for event in events], "consumed_prompt_tokens": 20, "consumed_completion_tokens": 10, "consumed_total_tokens": 30,
            "feature_contract_id": "default", "feature_contract_sha256": canonical_json_hash(features),
        }
        write_jsonl(root / "decisions.jsonl", [add_hash(decision, "decision_sha256")])
        valid = audit_protocol(root)
        if valid["exit_code"] != EXIT_VERIFIED:
            raise AssertionError(f"valid fixture unexpectedly failed: {valid}")
        broken = dict(decision)
        broken["consumed_total_tokens"] = 29
        write_jsonl(root / "decisions.jsonl", [add_hash(broken, "decision_sha256")])
        rejected = audit_protocol(root)
        if rejected["exit_code"] != EXIT_REJECTED:
            raise AssertionError(f"token mismatch fixture was not rejected: {rejected}")
        return {"valid_exit_code": valid["exit_code"], "token_mismatch_exit_code": rejected["exit_code"], "passed": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("research/outputs/prospective_committee"), help="Protocol artifact directory to audit.")
    parser.add_argument("--manifest", help="Relative manifest path below --output-dir; default discovers protocol_manifest.json.")
    parser.add_argument("--events", help="Relative event ledger path below --output-dir.")
    parser.add_argument("--barriers", help="Relative barrier ledger path below --output-dir.")
    parser.add_argument("--decisions", help="Relative decision ledger path below --output-dir.")
    parser.add_argument("--json", action="store_true", help="Emit full machine-readable report as JSON on stdout.")
    parser.add_argument("--self-test", action="store_true", help="Run built-in synthetic valid/rejected ledger checks in a temporary directory.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        try:
            report = self_test_report()
        except Exception as error:  # A self-test failure should be machine-detectable.
            report = {"passed": False, "error": str(error)}
            if args.json:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                print(f"Prospective protocol audit self-test failed: {error}", file=sys.stderr)
            return EXIT_INVALID
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print("Prospective protocol audit self-test passed: valid fixture verified; token mismatch rejected.")
        return EXIT_VERIFIED
    report = audit_protocol(
        args.output_dir,
        manifest_path=args.manifest,
        events_path=args.events,
        barriers_path=args.barriers,
        decisions_path=args.decisions,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print(human_report(report))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
