"""Synthetic contract tests only; no actual rehearsal evidence is produced."""
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.record_defense_rehearsal import (
    ATTESTATION_FLAGS, EVENT_TYPES, LedgerError, append_event, audit_ledger,
    digest_bytes, interactive, main, new_ledger, read_ledger, validate_ledger,
)


def test_plan():
    return {"slide_count": 25, "talk_target_seconds": 1800, "qa_target_seconds": 1800,
            "timing_tolerance_seconds": 120, "sources": {}, "speaker_notes_markdown": "SYNTHETIC TEST ONLY",
            "slides": [{"number": i, "title": f"TEST ONLY slide {i}", "planned_seconds": 72,
                        "planned_start_seconds": (i-1)*72, "planned_end_seconds": i*72,
                        "speaker_notes": "SYNTHETIC TEST ONLY", "speaker_notes_utf8_sha256": digest_bytes(b"SYNTHETIC TEST ONLY")}
                       for i in range(1, 26)]}


class RehearsalContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.start = datetime(2020, 1, 1, 12, tzinfo=timezone.utc)
        self.ledger = new_ledger("peer_mock", "SYNTHETIC TEST OPERATOR", "TEST_ONLY", test_plan())

    def tearDown(self):
        self.temp.cleanup()

    def timestamp(self, offset):
        return (self.start + timedelta(seconds=offset)).isoformat()

    def add(self, typ, offset, data=None):
        self.ledger = append_event(self.ledger, typ, data, self.timestamp(offset), root=self.root)

    def roster(self, audience="peer", consent="granted"):
        self.add("participant", -20, {"id": "speaker", "name": "TEST ONLY speaker", "role": "presenter"})
        self.add("participant", -19, {"id": "audience", "name": "TEST ONLY audience", "role": audience})
        self.add("consent", -18, {"participant_id": "speaker", "scope": "recording", "status": consent, "reference": "TEST ONLY consent"})
        self.add("consent", -17, {"participant_id": "audience", "scope": "recording", "status": consent, "reference": "TEST ONLY consent"})

    def session(self, question=True, skip=None, talk_seconds=1800):
        self.add("talk_start", 0)
        for i in range(2, 26):
            if i != skip:
                self.add("slide", (i-1)*talk_seconds/25, {"number": i})
        self.add("talk_end", talk_seconds)
        self.add("qa_start", talk_seconds + 10)
        if question:
            self.add("question", talk_seconds + 20, {"id": "q1", "asker_id": "audience", "text": "SYNTHETIC TEST ONLY question?"})
            self.add("outcome", talk_seconds + 40, {"question_id": "q1", "disposition": "answered", "answer_summary": "SYNTHETIC TEST ONLY answer", "supporting_references": ["TEST ONLY proof reference"]})
        self.add("qa_end", talk_seconds + 1810)
        self.add("session_end", talk_seconds + 1811)

    def evidence(self, kind="recording", coverage_end=3610):
        file = self.root / "SYNTHETIC_TEST_ONLY_evidence.txt"
        file.write_text("SYNTHETIC TEST BYTES: this is not a meeting or a recording.", encoding="utf-8")
        self.add("evidence", max(4000, coverage_end + 1), {"id": "test_evidence", "kind": kind, "reference": str(file),
                 "coverage_start": self.timestamp(-10), "coverage_end": self.timestamp(coverage_end),
                 "reviewed_by_id": "audience", "description": "SYNTHETIC TEST ONLY", "reviewed": True})
        return file

    def attest(self, flag_values=True):
        self.add("attestation", 4100, {"witness_id": "audience", "evidence_ids": ["test_evidence"],
                                      **{flag: flag_values for flag in ATTESTATION_FLAGS}})

    def test_blank_template_is_incomplete_and_has_no_actual_duration(self):
        blank = new_ledger("peer_mock", plan=test_plan(), template=True)
        report = audit_ledger(blank, self.root)
        self.assertEqual(report["evidence_status"], "incomplete")
        self.assertEqual(report["observed_event_count"], 0)
        self.assertIsNone(report["timing"]["talk"]["actual_seconds"])
        self.assertTrue(all(s["actual_closed_duration_seconds"] is None for s in report["slide_timings"]))
        with self.assertRaises(LedgerError):
            append_event(blank, "talk_start")

    def test_actual_sequence_produces_measured_timing_not_planned_substitution(self):
        self.roster()
        self.session()
        self.evidence()
        self.attest()
        report = audit_ledger(self.ledger, self.root)
        self.assertEqual(report["evidence_status"], "complete")
        self.assertEqual(report["timing"]["talk"]["actual_seconds"], 1800)
        self.assertEqual(report["timing"]["qa"]["actual_seconds"], 1800)
        self.assertTrue(report["timing_targets_met"])
        self.assertTrue(all(s["actual_closed_duration_seconds"] == 72 for s in report["slide_timings"]))

    def test_time_order_future_and_timezone_guards(self):
        self.roster()
        with self.assertRaises(LedgerError):
            self.add("talk_start", -30)
        with self.assertRaises(LedgerError):
            append_event(self.ledger, "talk_start", occurred_at="2020-01-01T12:00:00")
        with self.assertRaises(LedgerError):
            append_event(self.ledger, "talk_start", occurred_at="2999-01-01T12:00:00Z")

    def test_phase_and_slide_invariants(self):
        with self.assertRaises(LedgerError):
            self.add("qa_start", 0)
        self.roster()
        self.add("talk_start", 0)
        for number in (0, 26, True, 1):
            with self.assertRaises(LedgerError):
                self.add("slide", 1, {"number": number})
        with self.assertRaises(LedgerError):
            self.add("question", 1, {"id": "q", "asker_id": "audience", "text": "TEST ONLY"})

    def test_source_or_event_tampering_is_detected(self):
        self.roster()
        changed = copy.deepcopy(self.ledger)
        changed["events"][0]["data"]["name"] = "CHANGED"
        with self.assertRaises(LedgerError):
            validate_ledger(changed)
        changed = copy.deepcopy(self.ledger)
        changed["plan"]["slides"][0]["speaker_notes"] = "CHANGED"
        with self.assertRaises(LedgerError):
            validate_ledger(changed)

    def test_known_consent_cannot_be_inferred_and_decline_blocks_recording(self):
        self.roster(consent="declined")
        self.session()
        self.evidence()
        self.attest()
        report = audit_ledger(self.ledger, self.root)
        self.assertEqual(report["evidence_status"], "incomplete")
        self.assertTrue(any("consent" in reason.lower() for e in report["evidence_checks"] for reason in e["reasons"]))

    def test_late_grant_does_not_retroactively_cover_recording(self):
        self.roster(consent="unknown")
        self.add("consent", -5, {"participant_id": "speaker", "scope": "recording", "status": "granted", "reference": "TEST ONLY"})
        self.add("consent", -4, {"participant_id": "audience", "scope": "recording", "status": "granted", "reference": "TEST ONLY"})
        self.session()
        self.evidence()
        self.attest()
        self.assertEqual(audit_ledger(self.ledger, self.root)["evidence_status"], "incomplete")

    def test_skipped_slide_and_no_question_remain_incomplete(self):
        self.roster()
        self.session(question=False, skip=7)
        self.evidence()
        self.attest()
        report = audit_ledger(self.ledger, self.root)
        self.assertEqual(report["evidence_status"], "incomplete")
        self.assertIsNone(report["slide_timings"][6]["actual_closed_duration_seconds"])
        self.assertIn("No actual verbatim Q&A question recorded.", report["blockers"])

    def test_recording_reference_and_witness_are_both_required(self):
        self.roster()
        self.session()
        self.assertEqual(audit_ledger(self.ledger, self.root)["evidence_status"], "incomplete")
        self.evidence()
        self.attest(False)
        self.assertEqual(audit_ledger(self.ledger, self.root)["evidence_status"], "incomplete")

    def test_changed_evidence_bytes_reopen_incomplete_status(self):
        self.roster()
        self.session()
        file = self.evidence()
        self.attest()
        self.assertEqual(audit_ledger(self.ledger, self.root)["evidence_status"], "complete")
        file.write_text("DIFFERENT SYNTHETIC TEST BYTES", encoding="utf-8")
        self.assertEqual(audit_ledger(self.ledger, self.root)["evidence_status"], "incomplete")

    def test_dry_run_requires_adviser_and_accepts_unrecorded_dated_record(self):
        self.ledger = new_ledger("adviser_dry_run", "TEST OPERATOR", "TEST_ONLY", test_plan())
        self.roster(audience="adviser", consent="not_requested")
        self.session()
        self.evidence(kind="dated_record")
        self.attest()
        self.assertEqual(audit_ledger(self.ledger, self.root)["evidence_status"], "complete")
        changed = copy.deepcopy(self.ledger)
        changed["kind"] = "peer_mock"
        self.assertEqual(audit_ledger(changed, self.root)["evidence_status"], "incomplete")

    def test_complete_evidence_is_separate_from_timing_target(self):
        self.roster()
        self.session(talk_seconds=2000)
        self.evidence(coverage_end=3810)
        self.attest()
        report = audit_ledger(self.ledger, self.root)
        self.assertEqual(report["evidence_status"], "complete")
        self.assertFalse(report["timing_targets_met"])
        self.assertEqual(report["timing"]["talk"]["deviation_seconds"], 200)

    def test_unasked_or_unsupported_answer_is_rejected(self):
        self.roster()
        self.add("talk_start", 0)
        self.add("talk_end", 1800)
        self.add("qa_start", 1810)
        with self.assertRaises(LedgerError):
            self.add("outcome", 1820, {"question_id": "missing", "disposition": "not_answered", "action": "TEST ONLY followup"})
        self.add("question", 1820, {"id": "q", "asker_id": "audience", "text": "TEST ONLY question"})
        with self.assertRaises(LedgerError):
            self.add("outcome", 1830, {"question_id": "q", "disposition": "answered", "answer_summary": "TEST ONLY"})

    def test_audit_output_cannot_overwrite_actual_ledger(self):
        path = self.root / "TEST_ONLY_ledger.json"
        path.write_text(json.dumps(self.ledger), encoding="utf-8")
        before = path.read_bytes()
        self.assertEqual(main(["audit", "--ledger", str(path), "--output", str(path)]), 1)
        self.assertEqual(before, path.read_bytes())

    def test_schema_lists_the_same_event_types(self):
        schema = json.loads((Path(__file__).parent / "rehearsal_schema_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(set(schema["$defs"]["event"]["properties"]["type"]["enum"]), set(EVENT_TYPES))

    def test_cli_manual_input_validates_and_reports_incomplete_without_filling_times(self):
        ledger_path = self.root / "SYNTHETIC_CLI_ONLY.json"
        event_path = self.root / "SYNTHETIC_EVENT_ONLY.json"
        audit_path = self.root / "SYNTHETIC_AUDIT_ONLY.json"
        self.assertEqual(main(["init", "--ledger", str(ledger_path), "--kind", "peer_mock",
                               "--operator", "SYNTHETIC CLI TEST ONLY"]), 0)
        self.assertEqual(main(["init", "--ledger", str(ledger_path), "--kind", "peer_mock",
                               "--operator", "SYNTHETIC CLI TEST ONLY"]), 1)
        event_path.write_text(json.dumps({"type": "participant", "occurred_at": self.timestamp(0),
                               "data": {"id": "test_speaker", "name": "TEST ONLY", "role": "presenter"}}), encoding="utf-8")
        self.assertEqual(main(["append", "--ledger", str(ledger_path), "--event-file", str(event_path)]), 0)
        self.assertEqual(main(["validate", "--ledger", str(ledger_path)]), 0)
        self.assertEqual(main(["audit", "--ledger", str(ledger_path), "--output", str(audit_path)]), 2)
        report = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(report["observed_event_count"], 1)
        self.assertIsNone(report["timing"]["talk"]["actual_seconds"])

    def test_interactive_quit_preserves_an_empty_ledger(self):
        path = self.root / "SYNTHETIC_INTERACTIVE_ONLY.json"
        path.write_text(json.dumps(self.ledger), encoding="utf-8")
        with patch("builtins.input", side_effect=["quit"]):
            interactive(path)
        self.assertEqual(read_ledger(path)["events"], [])

    def test_revisits_accumulate_actual_time_and_keep_first_start(self):
        self.roster()
        self.add("talk_start", 0)
        self.add("slide", 70, {"number": 2})
        self.add("slide", 100, {"number": 1})
        self.add("slide", 150, {"number": 2})
        for i in range(3, 26):
            self.add("slide", 230 + (i-3)*70, {"number": i})
        self.add("talk_end", 1800)
        self.add("qa_start", 1810)
        self.add("question", 1820, {"id": "q", "asker_id": "audience", "text": "TEST ONLY question"})
        self.add("outcome", 1830, {"question_id": "q", "disposition": "needs_followup", "action": "TEST ONLY: inspect proof before next practice"})
        self.add("qa_end", 3610)
        self.add("session_end", 3611)
        self.evidence()
        self.attest()
        report = audit_ledger(self.ledger, self.root)
        self.assertEqual(report["evidence_status"], "complete")
        self.assertEqual(report["slide_timings"][0]["visit_count"], 2)
        self.assertEqual(report["slide_timings"][0]["actual_closed_duration_seconds"], 120)
        self.assertEqual(report["slide_timings"][0]["actual_first_start_seconds"], 0)
        self.assertEqual(report["slide_timings"][1]["actual_closed_duration_seconds"], 110)
        self.assertEqual(len(report["pending_question_actions"]), 1)

    def test_prior_peer_link_requires_complete_evidence_and_preserves_question_ids(self):
        path = self.root / "SYNTHETIC_PEER_ONLY.json"
        path.write_text(json.dumps(self.ledger), encoding="utf-8")
        with self.assertRaises(LedgerError):
            new_ledger("adviser_dry_run", "TEST ONLY", plan=test_plan(), prior_peer_ledger=path)
        self.roster()
        self.session()
        self.evidence()
        self.attest()
        path.write_text(json.dumps(self.ledger), encoding="utf-8")
        adviser = new_ledger("adviser_dry_run", "TEST ONLY", plan=test_plan(), prior_peer_ledger=path)
        self.assertEqual(adviser["prior_peer_mock"]["sha256"], digest_bytes(path.read_bytes()))
        self.assertEqual(adviser["prior_peer_mock"]["questions"], ["q1"])
        self.assertEqual(adviser["events"], [])


if __name__ == "__main__":
    unittest.main()
