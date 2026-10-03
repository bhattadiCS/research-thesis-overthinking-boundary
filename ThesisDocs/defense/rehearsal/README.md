# Actual rehearsal evidence

This directory prepares W10's recorded peer mock and W11's adviser dry-run. **No rehearsal has been observed or recorded here.** Both supplied ledgers are blank, contain zero observed events, and audit as incomplete. Document creation dates and planned durations are not rehearsal dates or measurements.

The logger is `tools/record_defense_rehearsal.py`. It uses Python's standard library, accepts manual observations, and neither captures audio/video nor contacts anyone. The source deck has 25 slides and a planned 1,800-second talk; Q&A has a separate 1,800-second target. Each session freezes the exact notes, slide titles and durations, deck/notes/timing/procedure/question-bank/PPTX byte hashes, and a canonical plan hash. The initial ledger and each appended event also record the exact logger-source hash; the initial Python version is retained. It does not edit those sources.

## Begin an actual session

Run from the repository root, using a new path for each rehearsal:

```powershell
python tools/record_defense_rehearsal.py init --ledger ThesisDocs/defense/rehearsal/sessions/peer_mock_actual.json --kind peer_mock --operator "ACTUAL OBSERVER NAME"
python tools/record_defense_rehearsal.py interactive --ledger ThesisDocs/defense/rehearsal/sessions/peer_mock_actual.json
```

Replace the operator placeholder with the human actually entering observations. Initializing creates an empty session, not a completed practice. The blank templates are reference examples, cannot accept observed events, and should remain blank. `init` checks the current JSON, cumulative timing plan, and exact notes before freezing them. It refuses to overwrite an existing ledger. A predeclared timing tolerance defaults to 120 seconds per phase; change it at initialization with `--tolerance-seconds`, before observing results.

For the adviser run, use `--kind adviser_dry_run` and another ledger. Add `--prior-peer-ledger PATH` to bind a complete actual peer-mock ledger and its question IDs. When recording a revisited question, supply that earlier question ID. At each audit, a supplied link must still match the prior ledger's bytes, session identity and question IDs, and the prior evidence must remain complete. The adviser talk cannot precede that peer session's end. Leave the link absent when there is no prior record to cite. This helps document the harder questions from the first rehearsal. Linking a prior log or an adviser participant does not establish defense logistics, an announcement, manuscript approval or an institutional review cycle.

## Observe the session

1. Enter `participant` for each actual attendee with a unique ID, name or agreed alias, and actual role (`presenter`, `observer`, `peer`, `adviser`). The presenter and peer/adviser must be observed before the talk. The final witness must attest that the roster is complete; an omitted attendee cannot be detected automatically.
2. Enter `consent` for each attendee. Record the actual status: `granted`, `declined`, `not_requested`, or `unknown`. Granted recording consent requires its actual written reference or dated verbal-consent description. Record it before recording begins. Unknown consent keeps the evidence incomplete. A peer mock requires recording consent from everyone captured. The logger never starts a recording; operate any external recording tool yourself with the actual participants' consent. An unrecorded adviser run can use a dated written record and an explicitly observed `not_requested` or `declined` recording status.
3. Enter `talk-start` at the actual start of slide 1. Press Enter or type `next` at each actual transition; type `slide N` for a jump or revisit. At slide 25, use `talk-end` at the actual end. Adjacent duplicate slides, invalid indices, and slide transitions outside the talk are rejected. Skipped slides and revisits remain visible in the audit.
4. Enter `qa-start` and `qa-end` at the actual Q&A boundaries. Between them, use `question` to record the actual asker's ID and verbatim question, and `outcome` to record the actual answer or unresolved outcome. An `answered` outcome requires the answer summary and supporting proof/data references. `needs_followup`, `deferred`, and `not_answered` require a real action/owner. Unresolved questions do not make an honestly documented session fictional; the audit retains their actions separately. Use outcome stage `follow_up` only for later work, after Q&A, so it cannot silently replace what happened during the session.
5. Enter `end` after the actual talk and Q&A finish. Add `evidence`: an existing local recording/dated-record file or an actual external HTTP(S) reference, its observed coverage start/end, reviewer ID and description. The human must explicitly state whether the reference was reviewed. The tool hashes existing local bytes; it cannot identify the people/content of a recording. External access and content review are entirely human-attested. W10's evidence must be a recording; W11 accepts a recording or dated record. Both must cover the talk and Q&A.
6. Use `attest` only after the session. A real witness explicitly answers each flag: actually observed the session, complete attendee roster, actually reviewed the cited evidence, and no fabricated events. No affirmative answer is inferred or filled by default.

Boundary and slide timestamps are captured as soon as the human enters the command. A question command likewise captures that observation before transcription, so typing the question does not shift its start time. Consent, outcomes, evidence review and attestation are timestamped when their entry is finished. These are human logging timestamps, not automatic measurements of speech or slides. `quit`, end of input, or an interrupted entry preserve saved events without filling remaining timestamps or saving a partial event.

## Transcribe a real dated record

For retrospective entry, prepare one event JSON at a time and append it **in actual chronological order**:

```json
{"type":"slide","occurred_at":"ACTUAL ISO-8601 TIME WITH UTC OFFSET","data":{"number":2}}
```

```powershell
python tools/record_defense_rehearsal.py append --ledger PATH_TO_ACTUAL_LEDGER --event-file PATH_TO_ACTUAL_EVENT_JSON
```

Replace the placeholder with the actual observed timestamp, for example a timestamp transcribed from the real recording; never a planned time. The schema is `rehearsal_schema_v1.json`. Timestamps require an explicit UTC offset. The manual event file must contain `type` and an object-valued `data`, with only an optional `occurred_at` field. Omit that field only for a current observation; an explicit empty, null, or malformed timestamp is rejected rather than replaced by the current time. Future observed events, backwards times, impossible phase order, unasked question outcomes, unknown attendees, and broken event/plan hashes are rejected. Appending preserves earlier events in a hash chain. The hashes detect accidental alteration; they are not authentication or independent proof that a meeting occurred. Use one logging process per ledger.

The fields for the twelve event types are:

| Event | Data fields |
|---|---|
| `participant` | `id`, `name`, `role` |
| `consent` | `participant_id`, `scope`=`recording`, `status`, and `reference` when granted |
| `talk_start`, `talk_end`, `qa_start`, `qa_end`, `session_end` | Empty object; boundaries are actual event timestamps |
| `slide` | `number` from 1 through 25 |
| `question` | `id`, `asker_id`, `text`; optional `from_peer_question_id` with a linked prior ledger |
| `outcome` | `question_id`, `stage`=`qa` or `follow_up`, `disposition`; `answer_summary` and nonempty `supporting_references` for answered, otherwise `action` |
| `evidence` | `id`, `kind`=`recording` or `dated_record`, `reference`, `coverage_start`, `coverage_end`, `reviewed_by_id`, `description`, `reviewed` boolean; local-file hash/path/size added by the CLI |
| `attestation` | `witness_id`, `evidence_ids`, and explicit booleans `observed_session`, `complete_attendee_roster`, `reviewed_evidence`, `no_fabricated_events` |

## Audit actual evidence and timing

```powershell
python tools/record_defense_rehearsal.py audit --ledger PATH_TO_ACTUAL_LEDGER --output PATH_TO_NEW_AUDIT_JSON --timing-csv PATH_TO_NEW_TIMING_CSV
```

The audit reports first slide-start deviations, actual accumulated time and duration deviations for all 25 slides, revisits, missing observations, talk/Q&A durations and target deviations, attendee/consent history, questions and original Q&A outcomes, later follow-ups, recording/record checks and source drift. A missing measurement is null, never a planned duration substituted as actual.

`evidence_status: complete` requires closed observed talk/Q&A/session boundaries, all 25 slides with positive observed durations, an actual peer or adviser of the correct kind before the talk, explicit known consent statuses, real questions and their actual Q&A outcomes, acceptable reviewed evidence covering both phases, and explicit post-session witness attestation. Recording consent must precede covered attendance and remain granted throughout recording. Missing or changed local evidence bytes block completion. Completion means consistency/completeness of **manually entered, human-attested rehearsal evidence**, not independent verification that the meeting occurred, mastery, accuracy, scheduling, a committee approval, or graduation.

Timing-target success is a separate field. A fully documented session can be complete while its talk or Q&A timing needs another practice. Source changes after a rehearsal remain visible; the frozen notes preserve what was practiced, and humans decide whether the changed deck needs another run. The audit preserves each original Q&A outcome and the separate follow-up history. Its `current_outcome` and pending actions use the latest observed follow-up, so a later answered question clears its pending action without rewriting the Q&A record. The optional prior-peer link and revisited-question IDs document W11 preparation; an otherwise complete dry-run does not imply that the linked first rehearsal or defense logistics happened without their own evidence.

Audit exit codes: 0 for complete evidence, 2 for incomplete evidence, 1 for invalid input/invariants. `--allow-incomplete` writes an honest incomplete audit with exit 0, useful for blank templates. `validate` checks structure without asserting completion. Audit output paths cannot overwrite the observed ledger.

## Verify the logger

```powershell
python -m unittest discover -s ThesisDocs/defense/rehearsal -p test_rehearsal_logger.py -v
python tools/record_defense_rehearsal.py validate --ledger ThesisDocs/defense/rehearsal/peer_mock_blank.json
python tools/record_defense_rehearsal.py audit --ledger ThesisDocs/defense/rehearsal/peer_mock_blank.json --allow-incomplete
```

The tests use **synthetic events and temporary evidence bytes only**. They are software verification and do not count as either mock defense. No microphone, external messaging, PDF, or real meeting is created by these commands.

## Historical blank-record provenance

The supplied blank ledgers, blank audits, timing CSVs, `test_output.txt` and `verification.json` are preserved historical artifacts. Their 19-test receipt and initial logger binding refer to Git revision `09225c95c676ab3cae1c4b9586d74f7f803f519f`, whose logger SHA-256 is `3ad5977bfdf9ca4a9fb38d5065c1c1be12f62903180bb384a99482143bd08edf`. They retain zero actual events and null observed times. Use that revision to reproduce the historical logger and receipt.

The current logger and test source include regression fixes made after that receipt. Consequently, the historical verification's hashes for the logger, test source and this README differ from current files; this is a source revision, not new rehearsal evidence. Do not refresh the old ledgers or audits to conceal that distinction. Run the current tests for current-code validation, and use `init` with a new path for future observations. A fresh audit records the auditing logger's current hash while retaining the ledger's initial logger hash.
