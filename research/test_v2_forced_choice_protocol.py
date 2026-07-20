#!/usr/bin/env python3
"""Synthetic no-model integration test for the prospective verifier protocol.

This test builds one immutable GPQA-style v2 barrier commit using the real
collector serializers, then audits the resulting derived ledgers.  It does not
load model weights or use an expected answer while constructing runtime events.
The expected audit result is intentionally ``EXIT_INCOMPLETE``: v2's current
``collect_only`` verifier is observational until a separately frozen override
policy is developed and confirmed.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from audit_prospective_protocol import EXIT_INCOMPLETE, EXIT_REJECTED, audit_protocol
from forced_choice_verifier import (
    ANONYMOUS_RATIONALE_V1,
    PROMPT_OPTIONS_ONLY_V1,
    ForcedChoiceScores,
    posterior_from_logprobs,
    render_verifier_prompt,
)
from real_trace_experiments import MODEL_CATALOG, TaskSpec
from run_prospective_committee_protocol import (
    COMMITS_DIR_NAME,
    PUBLIC_TASKS_NAME,
    SEALED_LABELS_NAME,
    barrier_id,
    canonical_json_bytes,
    create_or_validate_manifest,
    derived_seed,
    event_id,
    existing_commits,
    make_verifier_event_row,
    prepare_closed_main_barrier,
    prompt_hash,
    rebuild_derived_ledgers,
    schema_version_for_args,
    sha256_bytes,
    sha256_file,
    utc_strictly_after,
    verifier_model_spec_from_args,
    verifier_record,
    verifier_spec_from_args,
    write_barrier_commit,
    write_status,
)


def write_canonical_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def main_event(
    *,
    protocol_id: str,
    manifest_sha: str,
    file_sha: str,
    task: TaskSpec,
    alias: str,
    revision: str,
    ordinal: int,
) -> dict[str, object]:
    """Make a minimal valid main-row payload for serializer/auditor testing."""

    barrier = barrier_id(protocol_id, prompt_hash(task.prompt), 0, 1)
    seed = derived_seed(protocol_id, prompt_hash(task.prompt), 0, 1, alias, "main")
    row: dict[str, object] = {
        "event_id": event_id(barrier, alias),
        "barrier_id": barrier,
        "protocol_id": protocol_id,
        "protocol_manifest_sha256": file_sha,
        "manifest_sha256": manifest_sha,
        "generation_kind": "main",
        "task_id": task.task_id,
        "task_prompt_sha256": prompt_hash(task.prompt),
        "task_hash": prompt_hash(task.prompt),
        "domain": task.domain,
        "difficulty": task.difficulty,
        "answer_type": task.answer_type,
        "replica_id": 0,
        "step": 1,
        "model_alias": alias,
        "model_name": MODEL_CATALOG[alias].hf_name,
        "model_revision": revision,
        "event_seed": seed,
        "seed_scope": "per_event",
        "effective_rng_seed": seed,
        "batch_id": f"main-{ordinal}",
        "batch_index": 1,
        "batch_size": 1,
        "started_at_utc": f"2020-01-01T00:00:0{ordinal}Z",
        "ended_at_utc": f"2020-01-01T00:00:1{ordinal}Z",
        "started_monotonic_ns": ordinal * 1_000,
        "ended_monotonic_ns": ordinal * 1_000 + 500,
        "wall_clock_seconds": 0.0000005,
        "prompt_tokens": 11,
        "completion_tokens": 2,
        "total_tokens": 13,
        "thought": f"synthetic rationale {ordinal}",
        "answer": "A",
        "answer_normalized": "a",
        "confidence": 50,
        "model_stop_flag": 0,
        "parse_success": 1,
        "output_format_type": "synthetic",
        "answer_extraction_source": "synthetic",
    }
    row["event_sha256"] = sha256_bytes(canonical_json_bytes(row))
    return row


def run() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="prospective_verifier_v2_") as temporary:
        root = Path(temporary)
        prompt = (
            "Which option is the synthetic test answer?\n\nOptions:\n"
            "A. The first synthetic option\n"
            "B. The second synthetic option\n"
            "C. The third synthetic option\n"
            "D. The fourth synthetic option\n\n"
            "Respond with the letter of the single correct option."
        )
        task = TaskSpec(
            task_id="task_v2_forced_choice",
            domain="gpqa",
            difficulty="unit",
            prompt=prompt,
            answer_type="mcq",
            expected_answer="A",
            notes="Synthetic protocol-contract test only.",
            source="unit_test",
        )
        public_row = {
            "task_id": task.task_id,
            "domain": task.domain,
            "difficulty": task.difficulty,
            "prompt": task.prompt,
            "prompt_sha256": prompt_hash(task.prompt),
            "answer_type": task.answer_type,
            "notes": task.notes,
            "source": task.source,
            "dataset_revision": "f" * 40,
        }
        sealed_row = {
            "task_id": task.task_id,
            "prompt_sha256": prompt_hash(task.prompt),
            "expected_answer": task.expected_answer,
        }
        write_canonical_jsonl(root / PUBLIC_TASKS_NAME, [public_row])
        write_canonical_jsonl(root / SEALED_LABELS_NAME, [sealed_row])

        args = argparse.Namespace(
            protocol_id="v2_forced_choice_self_test",
            phase="development",
            roster=["qwen2p5_0p5b", "qwen2p5_3b"],
            replicas=[0],
            max_steps=1,
            max_new_tokens=8,
            temperature=0.0,
            batch_size=1,
            seed_mode="per_event",
            device="cpu",
            quantization="none",
            device_map=None,
            attn_implementation="sdpa",
            prompt_mode="minimal_json",
            system_prompt_mode="none",
            extended_observables=False,
            verifier_mode="collect_only",
            verifier_model="qwen2p5_0p5b",
            verifier_ledger_alias="gpqa_fc_qwen2p5_0p5b_v1",
            verifier_model_revision="d" * 40,
            verifier_variants=[PROMPT_OPTIONS_ONLY_V1, ANONYMOUS_RATIONALE_V1],
            verifier_active_variant=PROMPT_OPTIONS_ONLY_V1,
            verifier_batch_size=1,
            verifier_rationale_max_chars=128,
            policy="consensus",
            leader="qwen2p5_0p5b",
        )
        roster = tuple(args.roster)
        revisions = {"qwen2p5_0p5b": "a" * 40, "qwen2p5_3b": "b" * 40}
        verifier_model = verifier_model_spec_from_args(args)
        assert verifier_model is not None
        verifier_config = verifier_spec_from_args(args, verifier_model)
        assert verifier_config is not None
        manifest, manifest_sha, manifest_file_sha = create_or_validate_manifest(
            args,
            root,
            roster,
            revisions,
            sha256_file(root / PUBLIC_TASKS_NAME),
            sha256_file(root / SEALED_LABELS_NAME),
            1,
            {"gpu_name": "synthetic-gpu", "gpu_total_vram_bytes": 1024 * 1024 * 1024},
            verifier_record(verifier_model, "d" * 40),
            verifier_config,
        )
        main_events = [
            main_event(
                protocol_id=args.protocol_id,
                manifest_sha=manifest_sha,
                file_sha=manifest_file_sha,
                task=task,
                alias=alias,
                revision=revisions[alias],
                ordinal=index,
            )
            for index, alias in enumerate(roster, start=1)
        ]
        ordered_events, barrier = prepare_closed_main_barrier(
            protocol_id=args.protocol_id,
            manifest_sha=manifest_sha,
            protocol_manifest_file_sha=manifest_file_sha,
            task=task,
            replica_id=0,
            step=1,
            events=main_events,
            roster=roster,
        )
        post_events: list[dict[str, object]] = []
        logprobs = {"A": 0.0, "B": -1.0, "C": -2.0, "D": -3.0}
        posteriors, argmax, margin, entropy = posterior_from_logprobs(logprobs)
        scores = ForcedChoiceScores(
            option_logprobs=logprobs,
            option_posteriors=posteriors,
            argmax_option=argmax,
            top1_margin=margin,
            entropy=entropy,
            base_prompt_tokens=17,
            option_scoring_tokens={"A": 1, "B": 1, "C": 1, "D": 1},
        )
        for variant in verifier_config["variants"]:
            thoughts = [str(event["thought"]) for event in ordered_events] if variant == ANONYMOUS_RATIONALE_V1 else None
            seed_material = (
                f"{args.protocol_id}|{prompt_hash(task.prompt)}|0|1|{verifier_model.alias}|{variant}"
                if thoughts is not None
                else None
            )
            verifier_prompt = render_verifier_prompt(
                task.prompt,
                variant=variant,
                thoughts=thoughts,
                seed_material=seed_material,
                rationale_max_chars=128,
            )
            started_at = utc_strictly_after([str(barrier["closed_at_utc"])])
            started_ns = int(barrier["closed_monotonic_ns"]) + 10
            post_events.append(
                make_verifier_event_row(
                    protocol_id=args.protocol_id,
                    manifest_sha=manifest_sha,
                    protocol_manifest_file_sha=manifest_file_sha,
                    task=task,
                    replica_id=0,
                    step=1,
                    barrier=str(barrier["barrier_id"]),
                    barrier_record=barrier,
                    verifier_model=verifier_model,
                    verifier_revision="d" * 40,
                    verifier_config=verifier_config,
                    variant=variant,
                    seed=derived_seed(
                        args.protocol_id,
                        prompt_hash(task.prompt),
                        0,
                        1,
                        verifier_model.alias,
                        f"verifier:{variant}",
                    ),
                    started_at_utc=started_at,
                    ended_at_utc=utc_strictly_after([started_at]),
                    started_monotonic_ns=started_ns,
                    ended_monotonic_ns=started_ns + 1,
                    scores=scores,
                    prompt=verifier_prompt,
                    batch_id=f"verifier-{variant}",
                    batch_index=1,
                    batch_size=1,
                )
            )
        commits_dir = root / COMMITS_DIR_NAME
        commits_dir.mkdir()
        commit = write_barrier_commit(
            commits_dir=commits_dir,
            protocol_id=args.protocol_id,
            manifest_sha=manifest_sha,
            protocol_manifest_file_sha=manifest_file_sha,
            task=task,
            replica_id=0,
            step=1,
            events=ordered_events,
            roster=roster,
            policy=manifest["policy"],
            schema_version=schema_version_for_args(args),
            barrier_record=barrier,
            post_events=post_events,
        )
        commits = {str(barrier["barrier_id"]): commit}
        resumed = existing_commits(
            commits_dir,
            manifest_sha,
            manifest_file_sha,
            roster,
            schema_version_for_args(args),
            verifier_config,
        )
        if set(resumed) != {str(barrier["barrier_id"])}:
            raise AssertionError(f"Resumed v2 commits were not indexed by barrier ID: {sorted(resumed)}")
        rebuild_derived_ledgers(root, commits, schema_version=schema_version_for_args(args))
        write_status(
            root,
            manifest_sha=manifest_sha,
            protocol_manifest_file_sha=manifest_file_sha,
            expected_barriers=1,
            commits=commits,
            schema_version=schema_version_for_args(args),
            state="complete",
        )
        report = audit_protocol(root)
        if report["exit_code"] != EXIT_INCOMPLETE:
            raise AssertionError(json.dumps(report, indent=2, sort_keys=True, default=str))
        rejected_or_invalid = [
            finding
            for finding in report["findings"]
            if finding["category"] in {"rejected", "invalid"}
        ]
        if rejected_or_invalid:
            raise AssertionError(json.dumps(rejected_or_invalid, indent=2, sort_keys=True))
        # Re-sign an internally consistent commit with an undercharged
        # verifier prompt ledger.  A hash-only audit would accept this; the v2
        # semantic contract must still reject it.
        commit_path = commits_dir / f"{barrier['barrier_id']}.json"
        tampered = json.loads(commit_path.read_text(encoding="utf-8"))
        tampered_event = dict(tampered["post_events"][0])
        tampered_event["prompt_tokens"] = int(tampered_event["prompt_tokens"]) - 1
        tampered_event["total_tokens"] = int(tampered_event["total_tokens"]) - 1
        tampered_event.pop("event_sha256")
        tampered_event["event_sha256"] = sha256_bytes(canonical_json_bytes(tampered_event))
        tampered["post_events"][0] = tampered_event
        tampered["post_event_hashes"][str(tampered_event["event_id"])] = tampered_event["event_sha256"]
        tampered.pop("payload_sha256")
        tampered["payload_sha256"] = sha256_bytes(canonical_json_bytes(tampered))
        commit_path.write_bytes(canonical_json_bytes(tampered) + b"\n")
        tampered_commits = {str(barrier["barrier_id"]): tampered}
        rebuild_derived_ledgers(root, tampered_commits, schema_version=schema_version_for_args(args))
        rejected = audit_protocol(root)
        if rejected["exit_code"] != EXIT_REJECTED:
            raise AssertionError(json.dumps(rejected, indent=2, sort_keys=True, default=str))
        # A separately re-signed unknown prompt hash must fail reconstruction,
        # demonstrating that the auditor verifies more than template/event
        # hashes and token arithmetic.
        prompt_tampered = json.loads(canonical_json_bytes(commit).decode("utf-8"))
        prompt_event = dict(prompt_tampered["post_events"][0])
        prompt_event["verifier_prompt_sha256"] = "0" * 64
        prompt_event.pop("event_sha256")
        prompt_event["event_sha256"] = sha256_bytes(canonical_json_bytes(prompt_event))
        prompt_tampered["post_events"][0] = prompt_event
        prompt_tampered["post_event_hashes"][str(prompt_event["event_id"])] = prompt_event["event_sha256"]
        prompt_tampered.pop("payload_sha256")
        prompt_tampered["payload_sha256"] = sha256_bytes(canonical_json_bytes(prompt_tampered))
        commit_path.write_bytes(canonical_json_bytes(prompt_tampered) + b"\n")
        rebuild_derived_ledgers(
            root,
            {str(barrier["barrier_id"]): prompt_tampered},
            schema_version=schema_version_for_args(args),
        )
        prompt_rejected = audit_protocol(root)
        if prompt_rejected["exit_code"] != EXIT_REJECTED:
            raise AssertionError(json.dumps(prompt_rejected, indent=2, sort_keys=True, default=str))
        return {
            "passed": True,
            "exit_code": report["exit_code"],
            "assessment": report["assessment"],
            "events": report["ledgers"]["events"],
            "atomic_barrier_commits": report["ledgers"]["atomic_barrier_commits"],
            "tampered_verifier_exit_code": rejected["exit_code"],
            "tampered_prompt_exit_code": prompt_rejected["exit_code"],
        }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
