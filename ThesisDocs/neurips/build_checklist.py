"""Fill the unmodified official checklist questions and guidelines."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANSWERS = [
    ("Yes", r"Sections \ref{sec:intro}--\ref{sec:limits} distinguish conditional theory, retrospective diagnostics, and actual development generation; the abstract explicitly reports fixed-two behavior and inconclusive accuracy."),
    ("Yes", r"Section \ref{sec:limits} and Appendices \ref{app:calibration}--\ref{app:uncertainty} discuss filtration, calibration, parser/prompt shift, low accuracy, dependence, and incomplete historical provenance."),
    ("Yes", r"Section \ref{sec:theory} states the decision assumptions and gives proof sketches; Appendices \ref{app:proofs}--\ref{app:calibration} provide full proofs and common-reference counterexamples. Finite verification checks supplement rather than replace the proofs."),
    ("No", r"Appendix \ref{app:protocol} fully specifies the new fitted controller and live protocol, and saved records permit numerical reanalysis. Complete historical generation/training environments and all development choices are unavailable, so full independent reexecution of every historical result is not established."),
    ("No", r"The documented anonymous review capsule provides selected sources and evidence, but omits large raw archives and model weights (Appendix \ref{app:uncertainty}). Public open-access hosting and a complete license-reviewed release are not established."),
    ("No", r"Section \ref{sec:methods} and Appendix \ref{app:protocol} provide task roles, ordered features, optimization/calibration settings, cost, and decoding controls for the new controller. Some historical detector-training and generation metadata remain incomplete and are not inferred from current software."),
    ("Yes", r"Section \ref{sec:results} and Appendix \ref{app:uncertainty} define paired task/cell resampling units and exact binomial-reference bounds, including their independence and handpicked-bank limitations. No accuracy noninferiority or universal timing guarantee is claimed."),
    ("No", r"Appendix \ref{app:protocol} reports the live GPU, memory, software, decoding, model seconds, and token/padding ledgers. Full historical project compute, failed runs, and additional development CPU work were not comprehensively measured."),
    ("NA", r"A completed human author review of the NeurIPS Code of Ethics is not available for this agent-assisted preparation draft. Section \ref{sec:methods} discloses this verification boundary; license and release review remain pending."),
    ("Yes", r"Section \ref{sec:limits} discusses potential compute accessibility benefits and harms from premature stopping, low accuracy, and subgroup variation. It calls for confirmatory accuracy/subgroup evaluation and avoids unmeasured energy claims."),
    ("NA", r"The work does not introduce or release a new high-risk foundation model or scraped image dataset. It studies public reasoning benchmarks and a small probability controller; a broader licensed artifact release remains pending."),
    ("No", r"The benchmark, model, architecture, and theory creators are cited in the references. A complete version-by-version asset license/terms audit and release packaging are still pending, so compliance is not represented as fully verified."),
    ("Yes", r"Appendices \ref{app:protocol}--\ref{app:uncertainty} specify the new model, audits, decisions, tests, and scope limits. The accompanying anonymous review capsule contains the portable artifact and protocol/report/README documentation; final license review and public release remain pending."),
    ("NA", r"No new crowdsourcing or recruited human-subject experiment is conducted; evaluation uses existing public benchmarks and a handpicked mathematical task bank."),
    ("NA", r"No new crowdsourcing or recruited human-subject research is conducted. The manuscript does not invent an institutional approval or human verification attestation."),
    ("Yes", r"Section \ref{sec:methods} explicitly describes LLM coding agents' contributions to theory, auditing, implementation, label reconstruction, and manuscript preparation, and the remaining human verification. The response-generation LLM is also a core experimental component whose protocol is specified."),
]


def build() -> None:
    template_path = ROOT / "template" / "checklist.tex"
    raw = template_path.read_bytes()
    template = raw.decode("utf-8").replace("\r\n", "\n")
    body = template.split("%%% END INSTRUCTIONS %%%", 1)[1].lstrip()
    assert body.count(r"\answerTODO{}") == len(ANSWERS) == 16
    assert body.count(r"\justificationTODO{}") == 16
    original_questions = re.findall(r"Question: (.*)", body)
    for answer, justification in ANSWERS:
        body = body.replace(r"\answerTODO{}", "\\" + "answer" + answer + "{}", 1)
        body = body.replace(r"\justificationTODO{}", justification, 1)
    output = "\\section*{NeurIPS Paper Checklist}\n\n" + body
    assert re.findall(r"Question: (.*)", output) == original_questions
    assert r"\answerTODO{}" not in output and r"\justificationTODO{}" not in output
    # Questions, headings and guidelines are preserved; only answers change.
    (ROOT / "paper_checklist.tex").write_text(output, encoding="utf-8", newline="\n")
    meta = {
        "official_template_sha256": hashlib.sha256(raw).hexdigest(),
        "questions_preserved": 16,
        "instruction_block_removed": True,
        "answers": [{"item": i + 1, "answer": a, "justification_tex": j}
                    for i, (a, j) in enumerate(ANSWERS)],
        "scope": "Preparation draft; no human review or public release attestation invented",
    }
    (ROOT / "checklist_answers.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("Official checklist filled: 16 exact questions, 16 answers; no TODO.")


if __name__ == "__main__":
    build()
