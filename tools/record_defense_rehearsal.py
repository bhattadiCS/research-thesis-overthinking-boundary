"""Manually observed defense rehearsals; no recording or communications APIs.

Completion means a complete, human-attested evidence record, not independent
verification that an event occurred, academic approval, or meeting a timing target.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "defense-rehearsal-ledger-v1"
KINDS = ("peer_mock", "adviser_dry_run")
ROLES = ("presenter", "observer", "peer", "adviser")
EVENT_TYPES = ("participant", "consent", "talk_start", "slide", "talk_end",
               "qa_start", "question", "outcome", "qa_end", "session_end",
               "evidence", "attestation")
SOURCE_FILES = {
    "deck_json": "ThesisDocs/defense/defense_slides.json",
    "timing_plan": "ThesisDocs/defense/timing_plan.md",
    "speaker_notes": "ThesisDocs/defense/speaker_notes_30_minutes.md",
    "question_bank": "ThesisDocs/defense/defense_question_bank.md",
    "review_procedure": "ThesisDocs/committee_review_package_v1.md",
    "deck_pptx": "output/presentation/Thesis_Defense_v1_Aditya_Bhatt.pptx",
}
ATTESTATION_FLAGS = ("observed_session", "complete_attendee_roster",
                     "reviewed_evidence", "no_fabricated_events")


class LedgerError(ValueError):
    """An inconsistent event or evidence contract."""


def require(condition, message):
    if not condition:
        raise LedgerError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def instant(value):
    require(isinstance(value, str), "Timestamp must be an ISO-8601 string.")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LedgerError("Invalid timestamp: " + value) from exc
    require(result.tzinfo is not None and result.utcoffset() is not None,
            "Timestamp needs an explicit UTC offset, e.g. Z or -05:00.")
    return result.astimezone(timezone.utc)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical_digest(data):
    return digest_bytes(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode("utf-8"))


def logger_source_hash():
    return digest_bytes(Path(__file__).read_bytes())


def text_field(data, key):
    require(isinstance(data.get(key), str) and bool(data[key].strip()),
            f"{key} must be nonempty text.")
    return data[key]


def source_plan(root=ROOT, tolerance_seconds=120):
    """Freeze exact source bytes/notes and check the published cumulative plan."""
    require(type(tolerance_seconds) is int and tolerance_seconds >= 0,
            "Timing tolerance must be a nonnegative integer.")
    sources = {}
    for key, rel in SOURCE_FILES.items():
        path = Path(root) / rel
        require(path.is_file(), f"Required source is missing: {rel}")
        sources[key] = {"path": rel, "sha256": digest_bytes(path.read_bytes()),
                        "bytes": path.stat().st_size}
    deck = json.loads((Path(root) / SOURCE_FILES["deck_json"]).read_text(encoding="utf-8"))
    slides = deck["slides"]
    require(len(slides) == deck["slide_count"] == 25, "The source must have 25 slides.")
    require(sum(s["seconds"] for s in slides) == deck["total_seconds"] == 1800,
            "The source talk duration must be 1800 seconds.")
    timing_text = (Path(root) / SOURCE_FILES["timing_plan"]).read_text(encoding="utf-8")
    timing_rows = []
    for line in timing_text.splitlines():
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0].isdigit():
            timing_rows.append(cells)
    require(len(timing_rows) == 25, "The timing plan needs 25 numbered rows.")
    notes_text = (Path(root) / SOURCE_FILES["speaker_notes"]).read_text(encoding="utf-8")
    plan_slides, elapsed = [], 0
    for number, (slide, timing) in enumerate(zip(slides, timing_rows), 1):
        require(slide["number"] == number, "Deck slide numbers must be consecutive.")
        minutes, seconds = map(int, timing[1].split(":"))
        require(int(timing[0]) == number and minutes * 60 + seconds == elapsed
                and timing[2] == f'{slide["seconds"]} s' and timing[3] == slide["title"],
                f"Timing-plan/source disagreement at slide {number}.")
        require(slide["speaker_notes"] in notes_text,
                f"Exact slide {number} notes are missing from the notes Markdown.")
        plan_slides.append({"number": number, "title": slide["title"],
                            "planned_seconds": slide["seconds"],
                            "planned_start_seconds": elapsed,
                            "planned_end_seconds": elapsed + slide["seconds"],
                            "speaker_notes": slide["speaker_notes"],
                            "speaker_notes_utf8_sha256": digest_bytes(slide["speaker_notes"].encode("utf-8"))})
        elapsed += slide["seconds"]
    return {"slide_count": 25, "talk_target_seconds": 1800, "qa_target_seconds": 1800,
            "timing_tolerance_seconds": tolerance_seconds, "sources": sources,
            "speaker_notes_markdown": notes_text, "slides": plan_slides}


def new_ledger(kind, operator=None, session_id=None, plan=None, template=False,
               prior_peer_ledger=None):
    require(kind in KINDS, "Choose peer_mock or adviser_dry_run.")
    if not template:
        require(isinstance(operator, str) and bool(operator.strip()), "Name the human logging operator.")
    plan = copy.deepcopy(plan if plan is not None else source_plan())
    prior = None
    if prior_peer_ledger is not None:
        require(kind == "adviser_dry_run", "Only an adviser dry-run can link a prior peer mock.")
        p = Path(prior_peer_ledger).resolve()
        previous = read_ledger(p)
        previous_audit = audit_ledger(previous)
        require(previous["kind"] == "peer_mock" and previous_audit["evidence_status"] == "complete",
                "A linked prior peer ledger must have a complete rehearsal evidence record.")
        prior = {"path": str(p), "sha256": digest_bytes(p.read_bytes()),
                 "session_id": previous["session_id"],
                 "questions": [e["data"]["id"] for e in previous["events"] if e["type"] == "question"]}
    ledger = {"schema": SCHEMA, "template": template,
              "session_id": None if template else (session_id or str(uuid.uuid4())),
              "kind": kind, "operator": None if template else operator,
              "created_at": None if template else utc_now(),
              "logger_source": {"path": "tools/record_defense_rehearsal.py",
                                "sha256": logger_source_hash(), "python_version": sys.version.split()[0]},
              "plan": plan, "plan_sha256": canonical_digest(plan),
              "prior_peer_mock": prior, "events": []}
    validate_ledger(ledger)
    return ledger


def _data_contract(event):
    typ, data = event["type"], event["data"]
    require(isinstance(data, dict), "Event data must be an object.")
    if typ == "participant":
        for k in ("id", "name", "role"):
            text_field(data, k)
        require(data["role"] in ROLES, "Unknown participant role.")
    elif typ == "consent":
        text_field(data, "participant_id")
        require(data.get("scope") == "recording", "Consent scope is recording.")
        require(data.get("status") in ("unknown", "granted", "declined", "not_requested"),
                "Record the actual consent status.")
        if data["status"] == "granted":
            text_field(data, "reference")
    elif typ == "slide":
        require(type(data.get("number")) is int and 1 <= data["number"] <= 25,
                "Slide number must be an integer from 1 through 25.")
    elif typ == "question":
        for k in ("id", "asker_id", "text"):
            text_field(data, k)
        if "from_peer_question_id" in data:
            text_field(data, "from_peer_question_id")
    elif typ == "outcome":
        text_field(data, "question_id")
        require(data.get("stage", "qa") in ("qa", "follow_up"), "Outcome stage must be qa or follow_up.")
        require(data.get("disposition") in ("answered", "needs_followup", "deferred", "not_answered"),
                "Unknown question disposition.")
        if data["disposition"] == "answered":
            text_field(data, "answer_summary")
            require(isinstance(data.get("supporting_references"), list)
                    and bool(data["supporting_references"])
                    and all(isinstance(x, str) and bool(x.strip()) for x in data["supporting_references"]),
                    "An answered question needs the actual answer and supporting proof/data references.")
        else:
            text_field(data, "action")
    elif typ == "evidence":
        for k in ("id", "kind", "reference", "reviewed_by_id", "description"):
            text_field(data, k)
        require(data["kind"] in ("recording", "dated_record"), "Evidence must be recording or dated_record.")
        require(type(data.get("reviewed")) is bool, "Specify whether a human actually reviewed this evidence.")
        instant(data.get("coverage_start"))
        instant(data.get("coverage_end"))
        require(instant(data["coverage_end"]) >= instant(data["coverage_start"]),
                "Evidence coverage ends before it begins.")
        require(instant(data["coverage_end"]) <= instant(event["occurred_at"]),
                "Do not claim to have reviewed evidence from the future.")
    elif typ == "attestation":
        text_field(data, "witness_id")
        for flag in ATTESTATION_FLAGS:
            require(type(data.get(flag)) is bool, "Explicitly answer attestation flag: " + flag)
        require(isinstance(data.get("evidence_ids"), list)
                and all(isinstance(x, str) for x in data["evidence_ids"]), "List the evidence IDs reviewed.")
    else:
        require(not data, f"{typ} does not accept extra event data.")


def validate_ledger(ledger):
    require(isinstance(ledger, dict), "Ledger JSON must be an object.")
    require(ledger.get("schema") == SCHEMA, "Unsupported ledger schema.")
    require(ledger.get("kind") in KINDS and type(ledger.get("template")) is bool,
            "Invalid rehearsal kind/template flag.")
    if ledger["template"]:
        require(ledger.get("session_id") is None and ledger.get("operator") is None
                and ledger.get("created_at") is None, "A template cannot assert an actual session/operator.")
    else:
        text_field(ledger, "session_id")
        text_field(ledger, "operator")
        instant(ledger.get("created_at"))
    require(isinstance(ledger.get("events"), list), "Events must be an explicit list.")
    logger = ledger.get("logger_source", {})
    require(isinstance(logger, dict) and isinstance(logger.get("sha256"), str) and len(logger["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in logger["sha256"]), "Missing initial logger source hash.")
    prior = ledger.get("prior_peer_mock")
    if prior is not None:
        require(isinstance(prior, dict) and ledger["kind"] == "adviser_dry_run",
                "A prior peer link is an object belonging to an adviser dry-run.")
        for key in ("path", "session_id", "sha256"):
            text_field(prior, key)
        require(len(prior["sha256"]) == 64 and all(c in "0123456789abcdef" for c in prior["sha256"]),
                "Invalid prior peer ledger hash.")
        require(isinstance(prior.get("questions"), list)
                and all(isinstance(q, str) and bool(q.strip()) for q in prior["questions"])
                and len(set(prior["questions"])) == len(prior["questions"]),
                "Prior peer question IDs must be unique nonempty strings.")
    plan = ledger.get("plan")
    require(isinstance(plan, dict) and canonical_digest(plan) == ledger.get("plan_sha256"),
            "Frozen source/notes plan hash changed.")
    require(isinstance(plan.get("slides"), list) and isinstance(plan.get("sources"), dict),
            "Frozen plan needs explicit slide and source collections.")
    require(len(plan.get("slides", [])) == plan.get("slide_count") == 25
            and sum(s["planned_seconds"] for s in plan["slides"]) == plan["talk_target_seconds"] == 1800
            and plan.get("qa_target_seconds") == 1800,
            "The frozen contract is a 25-slide/30-minute talk plus 30-minute Q&A.")
    elapsed = 0
    for number, s in enumerate(plan["slides"], 1):
        require(isinstance(s, dict) and type(s.get("number")) is int and s["number"] == number
                and type(s["planned_seconds"]) is int and s["planned_seconds"] > 0
                and s["planned_start_seconds"] == elapsed
                and s["planned_end_seconds"] == elapsed + s["planned_seconds"]
                and digest_bytes(s["speaker_notes"].encode("utf-8")) == s["speaker_notes_utf8_sha256"],
                "Frozen slide/notes contract is inconsistent.")
        elapsed += s["planned_seconds"]
    require(type(plan.get("timing_tolerance_seconds")) is int and plan["timing_tolerance_seconds"] >= 0,
            "Invalid predeclared timing tolerance.")
    state = {"phase": "setup", "participants": {}, "consents": [], "visits": [],
             "questions": {}, "qa_outcomes": {}, "follow_up_outcomes": [],
             "evidence": {}, "attestations": [], "boundaries": {}}
    previous_hash, previous_time = None, None
    for index, event in enumerate(ledger.get("events", []), 1):
        require(isinstance(event, dict), "Each observed event must be an object.")
        require(not ledger["template"], "A blank template cannot contain observed events; initialize a session.")
        require(event.get("sequence") == index and event.get("type") in EVENT_TYPES,
                "Event sequence/type is invalid.")
        require(event.get("previous_event_sha256") == previous_hash, "Broken event hash chain.")
        unsealed = {k: v for k, v in event.items() if k != "event_sha256"}
        require(canonical_digest(unsealed) == event.get("event_sha256"), "An event changed after it was recorded.")
        t = instant(event["occurred_at"])
        require(t <= instant(event["recorded_at"]), "Observed event is dated after its logging time.")
        require(previous_time is None or t >= previous_time, "Observed event times must be chronological.")
        require(event.get("entry_mode") in ("observed_now", "manual_timestamp"), "Invalid entry mode.")
        require(isinstance(event.get("logger_source_sha256"), str) and len(event["logger_source_sha256"]) == 64
                and all(c in "0123456789abcdef" for c in event["logger_source_sha256"]), "Missing event logger source hash.")
        _data_contract(event)
        typ, data, phase = event["type"], event["data"], state["phase"]
        people, boundaries = state["participants"], state["boundaries"]
        if typ == "participant":
            require(phase != "ended" and data["id"] not in people, "Duplicate/late attendee entry.")
            people[data["id"]] = {**data, "observed_at": event["occurred_at"]}
        elif typ == "consent":
            require(data["participant_id"] in people, "Consent needs an observed attendee.")
            state["consents"].append({**data, "observed_at": event["occurred_at"]})
        elif typ == "talk_start":
            require(phase == "setup", "Talk can start only once, before Q&A.")
            require(any(p["role"] == "presenter" for p in people.values()), "Record the actual presenter first.")
            boundaries[typ] = event["occurred_at"]
            state["visits"].append({"number": 1, "start": event["occurred_at"], "end": None})
            state["phase"] = "talk"
        elif typ == "slide":
            require(phase == "talk", "A slide transition must occur during the talk.")
            require(data["number"] != state["visits"][-1]["number"], "Adjacent duplicate slide transition.")
            state["visits"][-1]["end"] = event["occurred_at"]
            state["visits"].append({"number": data["number"], "start": event["occurred_at"], "end": None})
        elif typ == "talk_end":
            require(phase == "talk", "Talk end needs an open talk.")
            state["visits"][-1]["end"] = event["occurred_at"]
            boundaries[typ] = event["occurred_at"]
            state["phase"] = "between"
        elif typ == "qa_start":
            require(phase == "between", "Q&A starts only after the talk ends.")
            boundaries[typ] = event["occurred_at"]
            state["phase"] = "qa"
        elif typ == "question":
            require(phase == "qa", "Record questions only during actual Q&A.")
            require(data["asker_id"] in people and data["id"] not in state["questions"],
                    "Question needs an observed asker and a unique ID.")
            if data.get("from_peer_question_id"):
                prior = ledger.get("prior_peer_mock")
                require(prior is not None and data["from_peer_question_id"] in prior["questions"],
                        "Follow-up question is not in the linked peer-mock ledger.")
            state["questions"][data["id"]] = {**data, "observed_at": event["occurred_at"]}
        elif typ == "outcome":
            require(data["question_id"] in state["questions"], "Outcome references an unasked question.")
            if data.get("stage", "qa") == "qa":
                require(phase == "qa", "A Q&A outcome must be observed before Q&A ends.")
                state["qa_outcomes"][data["question_id"]] = {**data, "observed_at": event["occurred_at"]}
            else:
                require(phase in ("after_qa", "ended"), "Follow-up work must remain distinct from Q&A responses.")
                state["follow_up_outcomes"].append({**data, "observed_at": event["occurred_at"]})
        elif typ == "qa_end":
            require(phase == "qa", "Q&A end needs open Q&A.")
            boundaries[typ] = event["occurred_at"]
            state["phase"] = "after_qa"
        elif typ == "session_end":
            require(phase == "after_qa", "Session end follows actual talk and Q&A ends.")
            boundaries[typ] = event["occurred_at"]
            state["phase"] = "ended"
        elif typ == "evidence":
            require(data["id"] not in state["evidence"] and data["reviewed_by_id"] in people,
                    "Evidence needs a unique ID and an observed human reviewer.")
            state["evidence"][data["id"]] = data
        elif typ == "attestation":
            require(phase == "ended" and data["witness_id"] in people,
                    "A real witness attests only after the session ends.")
            require(all(x in state["evidence"] for x in data["evidence_ids"]), "Attestation references unknown evidence.")
            state["attestations"].append(data)
        previous_hash, previous_time = event["event_sha256"], t
    return state


def local_reference(reference, root=ROOT):
    require(isinstance(reference, str) and bool(reference.strip()), "Evidence reference must be nonempty text.")
    try:
        parsed = urlparse(reference)
        if parsed.scheme in ("http", "https"):
            require(bool(parsed.netloc) and bool(parsed.hostname)
                    and not any(c.isspace() for c in reference),
                    "An HTTP(S) evidence reference needs a host and no whitespace.")
            parsed.port  # Reject malformed ports without opening or accessing the URL.
            return None
    except ValueError as exc:
        raise LedgerError("Invalid evidence reference: " + reference) from exc
    p = Path(reference).expanduser()
    return p if p.is_absolute() else Path(root) / p


def append_event(ledger, typ, data=None, occurred_at=None, root=ROOT, entry_mode=None):
    require(not ledger["template"], "Initialize a session; never populate the blank template.")
    validate_ledger(ledger)
    updated = copy.deepcopy(ledger)
    recorded = utc_now()
    require(data is None or isinstance(data, dict), "Event data must be an object.")
    payload = copy.deepcopy({} if data is None else data)
    observed = recorded if occurred_at is None else occurred_at
    instant(observed)
    if typ == "evidence":
        ref = local_reference(payload.get("reference", ""), root)
        if ref is not None:
            payload["resolved_local_path"] = str(ref.resolve())
            payload["file_sha256"] = digest_bytes(ref.read_bytes()) if ref.is_file() else None
            payload["file_bytes"] = ref.stat().st_size if ref.is_file() else None
    event = {"sequence": len(updated["events"]) + 1, "type": typ,
             "occurred_at": observed, "recorded_at": recorded,
             "entry_mode": entry_mode if entry_mode is not None else ("manual_timestamp" if occurred_at is not None else "observed_now"),
             "logger_source_sha256": logger_source_hash(),
             "data": payload, "previous_event_sha256": updated["events"][-1]["event_sha256"] if updated["events"] else None}
    event["event_sha256"] = canonical_digest(event)
    updated["events"].append(event)
    validate_ledger(updated)
    return updated


def _seconds(a, b):
    return round((instant(b) - instant(a)).total_seconds(), 6)


def prior_peer_check(ledger, root=ROOT):
    """Check optional lineage without rewriting either rehearsal record."""
    prior = ledger.get("prior_peer_mock")
    if prior is None:
        return {"status": "not_linked", "reason": None}
    path = Path(prior["path"])
    if not path.is_absolute():
        path = Path(root) / path
    try:
        if not path.is_file():
            return {"status": "missing", "reason": "Linked prior peer ledger is missing."}
        if digest_bytes(path.read_bytes()) != prior["sha256"]:
            return {"status": "changed", "reason": "Linked prior peer ledger bytes changed."}
        peer = read_ledger(path)
        require(peer["kind"] == "peer_mock" and peer["session_id"] == prior["session_id"],
                "Linked prior peer session identity does not match its snapshot.")
        questions = [e["data"]["id"] for e in peer["events"] if e["type"] == "question"]
        require(questions == prior["questions"], "Linked prior peer question snapshot does not match its record.")
        peer_audit = audit_ledger(peer, root)
        require(peer_audit["evidence_status"] == "complete", "Linked prior peer evidence is no longer complete.")
        return {"status": "matched", "reason": None, "session_end": peer_audit["actual_boundaries"]["session_end"]}
    except (LedgerError, OSError, KeyError, TypeError, ValueError) as exc:
        return {"status": "invalid", "reason": "Linked prior peer record cannot be verified: " + str(exc)}


def audit_ledger(ledger, root=ROOT):
    state = validate_ledger(ledger)
    blockers = []
    boundaries, people = state["boundaries"], state["participants"]
    for marker in ("talk_start", "talk_end", "qa_start", "qa_end", "session_end"):
        if marker not in boundaries:
            blockers.append("Missing observed boundary: " + marker)
    slides = []
    talk_start = boundaries.get("talk_start")
    prior_check = prior_peer_check(ledger, root)
    if prior_check["reason"] is not None:
        blockers.append(prior_check["reason"])
    elif prior_check["status"] == "matched" and talk_start and instant(talk_start) < instant(prior_check["session_end"]):
        blockers.append("The adviser talk begins before its linked prior peer session ends.")
    for planned in ledger["plan"]["slides"]:
        visits = [v for v in state["visits"] if v["number"] == planned["number"]]
        closed = [v for v in visits if v["end"] is not None]
        actual = round(sum(_seconds(v["start"], v["end"]) for v in closed), 6) if visits else None
        complete = bool(visits) and len(closed) == len(visits)
        if not complete or actual is None or actual <= 0:
            blockers.append(f'Slide {planned["number"]} has no complete positive-duration observation.')
        first = _seconds(talk_start, visits[0]["start"]) if visits and talk_start else None
        slides.append({"slide": planned["number"], "title": planned["title"],
                       "planned_start_seconds": planned["planned_start_seconds"],
                       "planned_duration_seconds": planned["planned_seconds"],
                       "actual_first_start_seconds": first, "visit_count": len(visits),
                       "actual_closed_duration_seconds": actual,
                       "start_deviation_seconds": round(first - planned["planned_start_seconds"], 6) if first is not None else None,
                       "duration_deviation_seconds": round(actual - planned["planned_seconds"], 6) if complete else None,
                       "observation_complete": complete})
    timing = {}
    for phase in ("talk", "qa"):
        actual = _seconds(boundaries[phase + "_start"], boundaries[phase + "_end"]) if all(phase + x in boundaries for x in ("_start", "_end")) else None
        target = ledger["plan"][phase + "_target_seconds"]
        timing[phase] = {"target_seconds": target, "actual_seconds": actual,
                         "deviation_seconds": round(actual - target, 6) if actual is not None else None,
                         "within_predeclared_tolerance": abs(actual - target) <= ledger["plan"]["timing_tolerance_seconds"] if actual is not None else None}
    if timing["talk"]["actual_seconds"] == 0 or timing["qa"]["actual_seconds"] == 0:
        blockers.append("Talk and Q&A must each have positive observed duration.")
    required_role = "peer" if ledger["kind"] == "peer_mock" else "adviser"
    if not any(p["role"] == required_role and talk_start and instant(p["observed_at"]) <= instant(talk_start) for p in people.values()):
        blockers.append("Missing actual " + required_role + " attendee before the talk.")
    for pid in people:
        consents = [c for c in state["consents"] if c["participant_id"] == pid]
        if not consents or consents[-1]["status"] == "unknown":
            blockers.append("Unknown recording-consent status: " + pid)
    if not state["questions"]:
        blockers.append("No actual verbatim Q&A question recorded.")
    unanswered = sorted(set(state["questions"]) - set(state["qa_outcomes"]))
    for qid in unanswered:
        blockers.append("Missing actual Q&A outcome: " + qid)
    evidence_checks = []
    for eid, data in state["evidence"].items():
        reasons = []
        if not data["reviewed"]:
            reasons.append("A human has not marked this evidence reviewed.")
        if not (talk_start and boundaries.get("qa_end") and
                instant(data["coverage_start"]) <= instant(talk_start) and
                instant(data["coverage_end"]) >= instant(boundaries["qa_end"])):
            reasons.append("Evidence does not cover the observed talk and Q&A.")
        ref = local_reference(data["reference"], root)
        if ref is not None:
            stored = Path(data.get("resolved_local_path", str(ref)))
            if not stored.is_file() or stored.stat().st_size == 0:
                reasons.append("Local evidence is missing or empty.")
            elif digest_bytes(stored.read_bytes()) != data.get("file_sha256"):
                reasons.append("Local evidence bytes changed since the reference was logged.")
        if data["kind"] == "recording":
            coverage_end = instant(data["coverage_end"])
            for pid, participant in people.items():
                begins = max(instant(participant["observed_at"]), instant(data["coverage_start"]))
                eligible = [c for c in state["consents"] if c["participant_id"] == pid and instant(c["observed_at"]) <= begins]
                revoked = any(c["participant_id"] == pid and begins < instant(c["observed_at"]) <= coverage_end and c["status"] != "granted" for c in state["consents"])
                if not eligible or eligible[-1]["status"] != "granted" or revoked:
                    reasons.append("Recording lacks prior uninterrupted consent: " + pid)
        evidence_checks.append({"id": eid, "kind": data["kind"], "reference": data["reference"],
                                "reference_verification": "local bytes checked; content review is human-attested" if ref is not None else "external reference; access/content review is human-attested",
                                "acceptable": not reasons, "reasons": reasons})
    acceptable = {e["id"] for e in evidence_checks if e["acceptable"] and
                  (ledger["kind"] != "peer_mock" or e["kind"] == "recording")}
    if not acceptable:
        blockers.append("No reviewed recording with coverage/consent." if ledger["kind"] == "peer_mock"
                        else "No reviewed dated record or recording covering the dry-run.")
    witnessed = any(all(a[x] for x in ATTESTATION_FLAGS) and bool(set(a["evidence_ids"]) & acceptable)
                    and talk_start and instant(people[a["witness_id"]]["observed_at"]) <= instant(talk_start)
                    for a in state["attestations"])
    if not witnessed:
        blockers.append("Missing explicit post-session witness attestation for acceptable evidence.")
    source_checks = []
    for name, source in ledger["plan"]["sources"].items():
        p = Path(root) / source["path"]
        current = digest_bytes(p.read_bytes()) if p.is_file() else None
        source_checks.append({"source": name, "path": source["path"], "frozen_sha256": source["sha256"],
                              "current_sha256": current, "matches": current == source["sha256"]})
    current_outcomes = dict(state["qa_outcomes"])
    for outcome in state["follow_up_outcomes"]:
        current_outcomes[outcome["question_id"]] = outcome
    pending_actions = [{"question_id": qid, "disposition": d["disposition"], "action": d.get("action", "")}
                       for qid, d in current_outcomes.items() if d["disposition"] != "answered"]
    return {"schema": "defense-rehearsal-audit-v1", "session_id": ledger["session_id"],
            "kind": ledger["kind"], "is_template": ledger["template"],
            "audit_generated_at": utc_now(), "ledger_content_sha256": canonical_digest(ledger),
            "plan_sha256": ledger["plan_sha256"], "observed_event_count": len(ledger["events"]),
            "initial_logger_source": ledger["logger_source"], "auditing_logger_sha256": logger_source_hash(),
            "evidence_status": "incomplete" if blockers else "complete",
            "completion_scope": "Consistency/completeness of manually entered, human-attested rehearsal evidence; not independent event verification or academic approval.",
            "blockers": blockers, "actual_boundaries": boundaries, "timing": timing,
            "timing_tolerance_seconds": ledger["plan"]["timing_tolerance_seconds"],
            "timing_targets_met": all(t["within_predeclared_tolerance"] is True for t in timing.values()),
            "slide_timings": slides, "participants": list(people.values()),
            "consent_history": state["consents"], "question_count": len(state["questions"]),
            "questions": [{**q, "qa_outcome": state["qa_outcomes"].get(qid), "current_outcome": current_outcomes.get(qid)} for qid, q in state["questions"].items()],
            "follow_up_outcomes": state["follow_up_outcomes"], "pending_question_actions": pending_actions,
            "evidence_checks": evidence_checks, "witness_attestation_count": len(state["attestations"]),
            "prior_peer_mock": ledger.get("prior_peer_mock"),
            "prior_peer_mock_check": prior_check,
            "peer_questions_revisited": [q["from_peer_question_id"] for q in state["questions"].values() if q.get("from_peer_question_id")],
            "current_source_checks": source_checks,
            "institutional_status": "Defense scheduling, announcement, committee approval, signatures and degree clearance are not established by this log."}


def read_ledger(path):
    ledger = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_ledger(ledger)
    return ledger


def write_json(path, data, new=False):
    path = Path(path)
    require(not new or not path.exists(), "Refusing to overwrite an existing ledger/template: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".writing")
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_timing_csv(path, report):
    fields = list(report["slide_timings"][0])
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["slide_timings"])


def _prompt(message, default=None):
    value = input(message + (f" [{default}]" if default is not None else "") + ": ").strip()
    return value if value else default


def _yes(message):
    value = _prompt(message + " (yes/no; no default)")
    require(value in ("yes", "no"), "Answer yes or no explicitly.")
    return value == "yes"


def interactive(ledger_path):
    print("Manual observations only. No audio/video capture, invites, or messages are sent.")
    print("Commands: participant, consent, talk-start, next [Enter], slide N, talk-end,")
    print("qa-start, question, outcome, qa-end, end, evidence, attest, audit, quit.")
    print("Use append --event-file for retrospective timestamps in chronological order.")
    while True:
        ledger = read_ledger(ledger_path)
        state = validate_ledger(ledger)
        try:
            line = input(f'{state["phase"]}> ').strip()
        except EOFError:
            print("Input ended; existing events remain unchanged. Completion is not inferred.")
            return
        command_time = utc_now()
        if line == "quit":
            return
        try:
            data = {}
            if line in ("", "next"):
                require(state["phase"] == "talk", "Enter/next is available only during the talk.")
                typ, data = "slide", {"number": state["visits"][-1]["number"] + 1}
            elif line.startswith("slide "):
                typ, data = "slide", {"number": int(line.split()[1])}
            elif line in ("talk-start", "talk-end", "qa-start", "qa-end", "end"):
                typ = {"end": "session_end"}.get(line, line.replace("-", "_"))
            elif line == "participant":
                typ, data = "participant", {"id": _prompt("Unique attendee ID"), "name": _prompt("Actual name/alias"), "role": _prompt("Actual role: presenter/observer/peer/adviser")}
            elif line == "consent":
                typ, data = "consent", {"participant_id": _prompt("Attendee ID"), "scope": "recording", "status": _prompt("Actual status: granted/declined/not_requested/unknown")}
                if data["status"] == "granted":
                    data["reference"] = _prompt("Actual consent record, or dated verbal-consent description")
            elif line == "question":
                typ, data = "question", {"id": _prompt("Unique question ID"), "asker_id": _prompt("Actual asker ID"), "text": _prompt("Question verbatim")}
                if ledger.get("prior_peer_mock"):
                    prior = _prompt("Prior peer question ID, only if revisiting one", "")
                    if prior:
                        data["from_peer_question_id"] = prior
            elif line == "outcome":
                typ, data = "outcome", {"question_id": _prompt("Question ID"), "stage": _prompt("Stage: qa/follow_up", "qa"), "disposition": _prompt("Actual outcome: answered/needs_followup/deferred/not_answered")}
                data["answer_summary"] = _prompt("Actual answer summary (blank if no answer)", "")
                if data["disposition"] == "answered":
                    data["supporting_references"] = _prompt("Actual proof/data references, separated by |", "").split("|")
                else:
                    data["action"] = _prompt("Actual follow-up action/owner")
            elif line == "evidence":
                typ, data = "evidence", {"id": _prompt("Unique evidence ID"), "kind": _prompt("Kind: recording/dated_record"), "reference": _prompt("Actual existing local file or external http(s) reference"), "coverage_start": _prompt("Actual evidence coverage start, ISO timestamp with offset"), "coverage_end": _prompt("Actual evidence coverage end, ISO timestamp with offset"), "reviewed_by_id": _prompt("Actual reviewer attendee ID"), "description": _prompt("What the actual evidence contains"), "reviewed": _yes("Has that human actually reviewed it")}
            elif line == "attest":
                typ, data = "attestation", {"witness_id": _prompt("Actual witness attendee ID"), "evidence_ids": _prompt("Evidence IDs actually reviewed, separated by |", "").split("|")}
                for flag in ATTESTATION_FLAGS:
                    data[flag] = _yes(flag.replace("_", " "))
            elif line == "audit":
                report = audit_ledger(ledger)
                print(json.dumps({k: report[k] for k in ("evidence_status", "blockers", "timing", "pending_question_actions")}, indent=2))
                continue
            else:
                raise LedgerError("Unknown command; no event saved.")
            observed_time = command_time if typ in ("talk_start", "slide", "talk_end", "qa_start", "qa_end", "session_end", "question") else None
            updated = append_event(ledger, typ, data, occurred_at=observed_time, entry_mode="observed_now")
            write_json(ledger_path, updated)
            print(f'Saved observed event {len(updated["events"])} at {updated["events"][-1]["occurred_at"]}.')
        except EOFError:
            print("Input ended during entry; no partial event was saved.")
            return
        except (LedgerError, ValueError) as exc:
            print("Not saved: " + str(exc), file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create an empty actual-session ledger; no rehearsal is asserted.")
    init.add_argument("--ledger", required=True, type=Path)
    init.add_argument("--kind", choices=KINDS, required=True)
    init.add_argument("--operator", required=True)
    init.add_argument("--session-id")
    init.add_argument("--tolerance-seconds", type=int, default=120)
    init.add_argument("--prior-peer-ledger", type=Path)
    templates = sub.add_parser("templates", help="Create both blank source-bound templates.")
    templates.add_argument("--directory", required=True, type=Path)
    for name in ("interactive", "validate", "append", "audit"):
        p = sub.add_parser(name)
        p.add_argument("--ledger", required=True, type=Path)
        if name == "append":
            p.add_argument("--event-file", required=True, type=Path,
                           help='JSON {"type":...,"data":...,"occurred_at":...}; omit timestamp only for a current observation.')
        if name == "audit":
            p.add_argument("--output", type=Path)
            p.add_argument("--timing-csv", type=Path)
            p.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            ledger = new_ledger(args.kind, args.operator, args.session_id,
                                plan=source_plan(tolerance_seconds=args.tolerance_seconds),
                                prior_peer_ledger=args.prior_peer_ledger)
            write_json(args.ledger, ledger, new=True)
            print("Empty session created; rehearsal evidence remains incomplete.")
        elif args.command == "templates":
            plan = source_plan()
            for kind in KINDS:
                write_json(args.directory / (kind + "_blank.json"), new_ledger(kind, plan=plan, template=True), new=True)
            print("Two blank templates created with zero observed events.")
        elif args.command == "interactive":
            interactive(args.ledger)
        elif args.command == "validate":
            validate_ledger(read_ledger(args.ledger))
            print("Event/plan invariants passed. This is not a completed rehearsal assertion.")
        elif args.command == "append":
            event = json.loads(args.event_file.read_text(encoding="utf-8"))
            require(isinstance(event, dict) and {"type", "data"} <= event.keys()
                    and event.keys() <= {"type", "data", "occurred_at"},
                    "Manual event JSON needs type/data and only an optional occurred_at field.")
            require(isinstance(event["data"], dict), "Manual event data must be an object.")
            if "occurred_at" in event:
                instant(event["occurred_at"])
            ledger = append_event(read_ledger(args.ledger), event["type"], event["data"], event.get("occurred_at"))
            write_json(args.ledger, ledger)
            print("Observed event appended; original earlier events preserved.")
        elif args.command == "audit":
            output_paths = [p.resolve() for p in (args.output, args.timing_csv) if p is not None]
            require(len(set(output_paths)) == len(output_paths) and args.ledger.resolve() not in output_paths,
                    "Audit outputs must be distinct from each other and from the observed ledger.")
            report = audit_ledger(read_ledger(args.ledger))
            if args.output:
                write_json(args.output, report)
            if args.timing_csv:
                write_timing_csv(args.timing_csv, report)
            print(json.dumps({k: report[k] for k in ("evidence_status", "observed_event_count", "blockers", "timing", "timing_targets_met")}, indent=2))
            return 0 if report["evidence_status"] == "complete" or args.allow_incomplete else 2
        return 0
    except KeyboardInterrupt:
        print("Logging interrupted; existing events are preserved and completion is not inferred.", file=sys.stderr)
        return 130
    except (LedgerError, OSError, KeyError, TypeError, ValueError) as exc:
        print("Rehearsal logger: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
