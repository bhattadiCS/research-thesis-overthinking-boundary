"""Integrity behavior for the data freeze, using independent temporary corpora."""

from __future__ import annotations

import copy
import csv
from fractions import Fraction
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("freeze_research_data", ROOT / "tools/freeze_research_data.py")
assert SPEC and SPEC.loader
freeze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze)


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cell = "global_fixture_gsm8k"
        self.source = f"{freeze.DATA_ROOT}/{self.cell}/trace_steps.csv"
        self.metadata = f"{freeze.DATA_ROOT}/{self.cell}/metadata.json"
        self.rows = [
            {"run_id": f"run-{run}", "task_id": f"task-{run}", "step": str(step),
             "correct": str((run + step) % 2), "model_alias": "fixture", "task_source": "gsm8k"}
            for run in range(2) for step in range(1, 6)
        ]
        self.write_trace()
        self.write(self.metadata, json.dumps({"model": {"alias": "fixture"}, "task_source": "gsm8k",
                                              "dataset_split": "train", "max_steps": 5}) + "\n")
        self.refresh_authority()

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value.encode("utf-8") if isinstance(value, str) else value)
        return path

    def write_trace(self):
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=list(self.rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(self.rows)
        self.write(self.source, output.getvalue())

    def refresh_authority(self):
        record = freeze.file_record(self.root, self.source)
        files = [{"path": f"{self.cell}/trace_steps.csv", "bytes": record["canonical_lf_bytes"],
                  "sha256": record["canonical_lf_sha256"]}]
        self.authority = {"files": files, "dataset_fingerprint": freeze.stable_hash(files),
                          "selected_cell_count": 1, "rows": 10, "source_qualified_trajectories": 2,
                          "raw_run_ids": 2, "raw_run_id_cross_cell_collisions": 0,
                          "task_ids": 2, "sequence_length_distribution": {"5": 2},
                          "preflight": {"torch_version": "2.13.0+cu130", "cuda_runtime": "13.0"}}
        self.write(freeze.AUTHORITY, json.dumps(self.authority) + "\n")

    def manifest(self, selection=None):
        return freeze.build_manifest(self.root, selection=selection, created_utc="2026-10-02T00:00:00Z")

    def use_canonical_authority(self):
        record = freeze.file_record(self.root, self.source)
        entry = self.authority["files"][0]
        entry["canonical_lf_sha256"] = record["canonical_lf_sha256"]
        entry["raw_sha256"] = record["sha256"]
        self.authority["dataset_fingerprint"] = freeze.stable_hash([
            {"path": entry["path"], "canonical_lf_sha256": entry["canonical_lf_sha256"]}
        ])
        self.authority["raw_dataset_fingerprint"] = freeze.stable_hash([
            {"path": entry["path"], "raw_sha256": entry["raw_sha256"]}
        ])
        self.write(freeze.AUTHORITY, json.dumps(self.authority) + "\n")

    def test_clean_corpus_verifies_and_is_deterministic(self):
        manifest = self.manifest()
        self.assertEqual(manifest, self.manifest())
        result = freeze.verify_manifest(self.root, manifest)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["rows"], 10)
        self.assertEqual(manifest["corpus"]["rows_by_step"], {str(step): 2 for step in range(1, 6)})
        self.assertEqual(manifest["authority"]["recorded_dataset_fingerprint"], self.authority["dataset_fingerprint"])

    def test_content_mutation_cannot_be_silently_refrozen(self):
        manifest = self.manifest()
        self.rows[0]["correct"] = "0"
        self.write_trace()
        with self.assertRaisesRegex(freeze.IntegrityError, "archived tournament SHA256"):
            freeze.verify_manifest(self.root, manifest)
        with self.assertRaises(freeze.IntegrityError):
            self.manifest()

    def test_missing_or_added_cell_fails_membership(self):
        manifest = self.manifest()
        path = self.root / self.source
        data = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(freeze.IntegrityError, "membership"):
            freeze.verify_manifest(self.root, manifest)
        self.write(self.source, data)
        self.write(f"{freeze.DATA_ROOT}/global_extra_gsm8k/trace_steps.csv", data)
        with self.assertRaisesRegex(freeze.IntegrityError, "membership"):
            freeze.verify_manifest(self.root, manifest)

    def test_portable_audit_accepts_only_line_endings(self):
        manifest = self.manifest()
        for relative in [self.source, self.metadata, freeze.AUTHORITY]:
            path = self.root / relative
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        with self.assertRaisesRegex(freeze.IntegrityError, "Changed file bytes"):
            freeze.verify_manifest(self.root, manifest)
        result = freeze.verify_manifest(self.root, manifest, allow_line_ending_changes=True)
        self.assertEqual(len(result["line_ending_only_changes"]), 3)
        path = self.root / self.source
        path.write_bytes(path.read_bytes().replace(b"task-0", b"task-x", 1))
        with self.assertRaises(freeze.IntegrityError):
            freeze.verify_manifest(self.root, manifest, allow_line_ending_changes=True)

    def test_chunk_boundary_and_legacy_cr_are_normalized_exactly(self):
        body = b"a" * (1024 * 1024 - 1) + b"\r\nb\rc\n\r"
        self.write("boundary.txt", body)
        record = freeze.file_record(self.root, "boundary.txt")
        normalized = body.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        self.assertEqual(record["canonical_lf_bytes"], len(normalized))
        self.assertEqual(record["canonical_lf_sha256"], freeze.hashlib.sha256(normalized).hexdigest())
        self.write("archive.npz", body)
        self.assertNotIn("canonical_lf_sha256", freeze.file_record(self.root, "archive.npz"))

    def test_duplicate_steps_rejected_even_if_authority_is_updated(self):
        self.rows[1]["step"] = self.rows[0]["step"]
        self.write_trace()
        self.refresh_authority()
        with self.assertRaisesRegex(freeze.IntegrityError, "Duplicate trajectory step"):
            self.manifest()

    def test_trajectory_cannot_change_tasks(self):
        self.rows[1]["task_id"] = "other-task"
        self.write_trace()
        self.refresh_authority()
        with self.assertRaisesRegex(freeze.IntegrityError, "multiple task IDs"):
            self.manifest()

    def test_nonbinary_labels_and_noncontiguous_steps_rejected(self):
        self.rows[1]["correct"] = "2"
        self.write_trace()
        self.refresh_authority()
        with self.assertRaisesRegex(freeze.IntegrityError, "nonbinary"):
            self.manifest()
        self.rows[1]["correct"] = "0"
        self.rows[1]["step"] = "6"
        self.write_trace()
        self.refresh_authority()
        with self.assertRaisesRegex(freeze.IntegrityError, "Noncontiguous"):
            self.manifest()

    def test_metadata_identity_mismatch_rejected(self):
        self.write(self.metadata, json.dumps({"model": {"alias": "wrong"}, "task_source": "gsm8k"}))
        with self.assertRaisesRegex(freeze.IntegrityError, "disagrees with metadata"):
            self.manifest()

    def test_forged_summary_rejected_with_recomputed_manifest_hash(self):
        manifest = self.manifest()
        manifest["files"][0]["rows"] = 999
        manifest["content_fingerprint"] = freeze.content_fingerprint(manifest)
        with self.assertRaisesRegex(freeze.IntegrityError, "Rebuilt scope"):
            freeze.verify_manifest(self.root, manifest)

    def test_nested_provenance_fields_are_not_ignored(self):
        for portable in (False, True):
            for location, key in [("scope", "created_utc"), ("preflight", "content_fingerprint"),
                                  ("model", "created_utc")]:
                with self.subTest(portable=portable, location=location, key=key):
                    manifest = self.manifest()
                    parent = {"scope": manifest["scope"],
                              "preflight": manifest["authority"]["recorded_preflight"],
                              "model": manifest["generation_configurations"][0]["recorded_configuration"]["model"]}[location]
                    parent[key] = "forged source provenance"
                    manifest["content_fingerprint"] = freeze.content_fingerprint(manifest)
                    with self.assertRaisesRegex(freeze.IntegrityError, "Rebuilt scope"):
                        freeze.verify_manifest(self.root, manifest, allow_line_ending_changes=portable)

    def test_portable_hash_exclusions_apply_only_to_file_records(self):
        self.write(self.metadata, json.dumps({"model": {"alias": "fixture", "canonical_lf_sha256": "model field",
                                                       "sha256": "recorded model identity", "bytes": 12},
                                             "task_source": "gsm8k", "dataset_split": "train", "max_steps": 5}))
        manifest = self.manifest()
        model = manifest["generation_configurations"][0]["recorded_configuration"]["model"]
        model["sha256"] = "forged model identity"
        manifest["content_fingerprint"] = freeze.content_fingerprint(manifest)
        with self.assertRaisesRegex(freeze.IntegrityError, "Rebuilt scope"):
            freeze.verify_manifest(self.root, manifest, allow_line_ending_changes=True)

    def test_canonical_authority_verifies_across_line_endings(self):
        self.use_canonical_authority()
        manifest = self.manifest()
        self.assertEqual(freeze.verify_manifest(self.root, manifest)["status"], "verified")
        path = self.root / self.source
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        result = freeze.verify_manifest(self.root, manifest, allow_line_ending_changes=True)
        self.assertEqual(result["line_ending_only_changes"], [self.source])

    def test_canonical_authority_aggregate_must_match(self):
        self.use_canonical_authority()
        self.authority["dataset_fingerprint"] = "0" * 64
        self.write(freeze.AUTHORITY, json.dumps(self.authority) + "\n")
        with self.assertRaisesRegex(freeze.IntegrityError, "canonical dataset fingerprint"):
            self.manifest()

    def test_declared_raw_authority_aggregate_must_match(self):
        self.use_canonical_authority()
        self.authority["raw_dataset_fingerprint"] = "0" * 64
        self.write(freeze.AUTHORITY, json.dumps(self.authority) + "\n")
        with self.assertRaisesRegex(freeze.IntegrityError, "raw dataset fingerprint"):
            self.manifest()

    def test_duplicate_frozen_entries_rejected(self):
        manifest = self.manifest()
        manifest["files"].append(copy.deepcopy(manifest["files"][0]))
        manifest["content_fingerprint"] = freeze.content_fingerprint(manifest)
        with self.assertRaisesRegex(freeze.IntegrityError, "membership"):
            freeze.verify_manifest(self.root, manifest)

    def test_auxiliary_membership_and_content_are_audited(self):
        self.write("evidence/one.json", "{}\n")
        selection = {"profile": "fixture", "fixed": [self.metadata], "globs": ["evidence/*.json"]}
        manifest = self.manifest(selection)
        self.write("evidence/two.json", "{}\n")
        with self.assertRaisesRegex(freeze.IntegrityError, "membership"):
            freeze.verify_manifest(self.root, manifest)
        (self.root / "evidence/two.json").unlink()
        self.write("evidence/one.json", "{\"change\":1}\n")
        with self.assertRaisesRegex(freeze.IntegrityError, "Changed file bytes"):
            freeze.verify_manifest(self.root, manifest)

    def test_paths_cannot_escape_repository(self):
        for relative in ["../outside", "/absolute", "C:/absolute", "folder\\file", "folder//file", "./file"]:
            with self.subTest(relative=relative), self.assertRaises(freeze.IntegrityError):
                freeze.checked_path(self.root, relative)

    def test_declared_exclusions_ignore_only_generated_bytecode(self):
        self.write("evidence/locked_code/controller.py", "source\n")
        self.write("evidence/locked_code/__pycache__/controller.cpython-312.pyc", b"compiled cache")
        self.write("evidence/unrelated.pyc", b"another compiled cache")
        selection = {"profile": "fixture", "fixed": [], "globs": ["evidence/**/*"],
                     "exclude_globs": ["**/__pycache__/**", "**/*.pyc"]}
        manifest = freeze.build_manifest(self.root, selection=selection)
        self.write("evidence/locked_code/__pycache__/controller.cpython-312.pyc", b"regenerated cache")
        freeze.verify_manifest(self.root, manifest)
        self.write("evidence/locked_code/controller.py", "changed source\n")
        with self.assertRaisesRegex(freeze.IntegrityError, "Changed file bytes"):
            freeze.verify_manifest(self.root, manifest)

    def test_existing_output_is_preserved_without_replace(self):
        path = self.write("user.json", "user data\n")
        with self.assertRaisesRegex(freeze.IntegrityError, "Output exists"):
            freeze.write_output(path, "replacement", replace=False)
        self.assertEqual(path.read_text(), "user data\n")

    def test_cli_replace_cannot_overwrite_any_selected_source(self):
        originals = {relative: (self.root / relative).read_bytes()
                     for relative in (self.source, self.metadata, freeze.AUTHORITY)}
        for relative in originals:
            with self.subTest(output=relative):
                for source, content in originals.items():
                    self.write(source, content)
                with patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()) as error:
                    status = freeze.main(["--root", str(self.root), "freeze", "--profile", "tournament",
                                          "--output", relative, "--replace"])
                self.assertEqual(status, 1)
                self.assertIn("overwrite a selected input", error.getvalue())
                self.assertEqual({source: (self.root / source).read_bytes() for source in originals}, originals)

    def test_cli_output_cannot_add_a_tournament_cell(self):
        relative = f"{freeze.DATA_ROOT}/global_extra_gsm8k/trace_steps.csv"
        with patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()) as error:
            status = freeze.main(["--root", str(self.root), "freeze", "--profile", "tournament", "--output", relative])
        self.assertEqual(status, 1)
        self.assertIn("tournament file membership", error.getvalue())
        self.assertFalse((self.root / relative).exists())

    def test_cli_output_cannot_enter_its_own_glob_membership(self):
        for relative in ("evidence/review_manifest.json", "evidence/nested/review_manifest.json"):
            with self.subTest(output=relative), patch.object(freeze, "THESIS_FIXED", [self.metadata]), \
                    patch.object(freeze, "THESIS_GLOBS", ["evidence/**/*.json"]), \
                    patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()) as error:
                status = freeze.main(["--root", str(self.root), "freeze", "--output", relative])
                self.assertEqual(status, 1)
                self.assertIn("own evidence membership", error.getvalue())
                self.assertFalse((self.root / relative).exists())

    def test_separate_new_manifest_output_is_verifiable(self):
        relative = "review_manifest.json"
        with patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()):
            status = freeze.main(["--root", str(self.root), "freeze", "--profile", "tournament", "--output", relative])
        self.assertEqual(status, 0)
        manifest = freeze.read_object(self.root / relative)
        self.assertEqual(freeze.verify_manifest(self.root, manifest)["status"], "verified")

    def test_explicitly_excluded_manifest_output_is_verifiable(self):
        relative = "evidence/review_manifest.json"
        selection = {"profile": "fixture", "fixed": [self.metadata], "globs": ["evidence/**/*.json"],
                     "exclude_globs": [relative]}
        manifest = self.manifest(selection)
        output = freeze.checked_freeze_output(self.root, relative, manifest)
        freeze.write_output(output, json.dumps(manifest) + "\n", replace=False)
        self.assertEqual(freeze.verify_manifest(self.root, manifest)["status"], "verified")

    def test_directory_only_glob_does_not_block_a_manifest_file(self):
        relative = "evidence/review_manifest.json"
        selection = {"profile": "fixture", "fixed": [self.metadata], "globs": ["evidence/**"]}
        manifest = self.manifest(selection)
        output = freeze.checked_freeze_output(self.root, relative, manifest)
        freeze.write_output(output, json.dumps(manifest) + "\n", replace=False)
        self.assertEqual(freeze.verify_manifest(self.root, manifest)["status"], "verified")

    def test_local_and_historical_versions_are_separate(self):
        class Distribution:
            metadata = {"Name": "torch"}
            version = "2.11.0+cu128"

            def read_text(self, _):
                return None

        self.write("requirements-colab.txt", "transformers==4.53.3\n")
        with patch.object(freeze.importlib.metadata, "distributions", return_value=[Distribution()]):
            result = freeze.capture_environment(self.root, replace=False)
        provenance = json.loads((self.root / "software_provenance_v1.json").read_text())
        self.assertEqual(provenance["current_local_environment"]["relevant_packages"]["torch"], "2.11.0+cu128")
        self.assertEqual(provenance["archived_tournament_environment"]["recorded_preflight"]["torch_version"], "2.13.0+cu130")
        self.assertFalse(result["historical_lock_complete"])
        self.assertIn("torch==2.11.0+cu128", (self.root / "requirements.local-environment.lock.txt").read_text())
        self.assertIn("torch==2.13.0+cu130", (self.root / "requirements.historical-tournament.partial.txt").read_text())


class AdversarialGoldTests(unittest.TestCase):
    def test_public_gold_pair_and_independent_exact_answers(self):
        public_rows = [json.loads(line) for line in (ROOT / "research/adversarial_tasks_v1.jsonl").read_text().splitlines() if line.strip()]
        gold_rows = [json.loads(line) for line in (ROOT / "research/adversarial_gold_v1.jsonl").read_text().splitlines() if line.strip()]
        public = {row["task_id"]: row for row in public_rows}
        gold = {row["task_id"]: row for row in gold_rows}
        self.assertEqual(len(public_rows), 20)
        self.assertEqual(len(gold_rows), 20)
        self.assertEqual(len(public), len(public_rows))
        self.assertEqual(len(gold), len(gold_rows))
        self.assertEqual(public.keys(), gold.keys())
        # Calculations are independent of the stored derivation strings and use
        # exact fractions. The race contract presumes unchanged speeds between
        # races; the prompt was also read for semantic ambiguities during review.
        expected = {
            "trap_bat_ball": (Fraction(11, 10) - 1) / 2,
            "trap_machines": Fraction(100, 100 * Fraction(1, 5)),
            "trap_lily": Fraction(48 - 1),
            "trap_average_speed": Fraction(120, Fraction(60, 30) + Fraction(60, 60)),
            "trap_discount": 100 * Fraction(120, 100) * Fraction(80, 100),
            "trap_percent_base": (100 - 80) * Fraction(100, 100),
            "trap_head_start": Fraction(90, 100) * (100 + 10),
            "trap_conditional_dice": Fraction(1, 6 + 6 - 1),
            "trap_pairwise_handshakes": Fraction(10 * 9, 2),
            "trap_zero_product": Fraction(len({0, 2})),
            "trap_square_root": Fraction(7),
            "trap_exponent_sign": Fraction(-(3 ** 2)),
            "trap_negative_product": Fraction((-3) ** 2),
            "trap_without_replacement": Fraction(3, 5) * Fraction(2, 4),
            "trap_reciprocal_rate": 1 / (Fraction(1, 3) + Fraction(1, 6)),
            "trap_inclusive_count": Fraction(43 - 17 + 1),
            "trap_percent_points": Fraction(50 - 40, 40) * 100,
            "trap_weekday_cycle": Fraction(100 % 7),
            "trap_mixture": Fraction(2 * 10 + 3 * 20, 2 + 3),
            "trap_irrelevant_fact": Fraction(120 + 15),
        }
        self.assertEqual(expected.keys(), gold.keys())
        for task_id, answer in expected.items():
            with self.subTest(task_id=task_id):
                self.assertEqual(Fraction(gold[task_id]["expected_answer"]), answer)
                self.assertEqual(public[task_id]["answer_type"], gold[task_id]["answer_type"])
                self.assertEqual(public[task_id]["answer_type"], "number")
                self.assertFalse({"expected_answer", "correct", "answer", "gold"} & public[task_id].keys())


if __name__ == "__main__":
    unittest.main()
