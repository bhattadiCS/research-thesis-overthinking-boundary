"""Label-blind stopping decisions at completed reasoning-step boundaries.

The production API accepts one *new* completed observation, not a trace table.
It owns the prefix and has no field for gold, correctness, future steps, or a
retrospective ensemble score.  This first policy is a frozen heuristic, not the
Semester 1 full-sequence classifier and not a calibrated correctness predictor.
Peer agreement is usable only through a complete, timestamped, same-step panel.
No cross-model peer information is required by the default single-model policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence


SCHEMA_VERSION = "prefix-online-stopper-v1"
T_MIN = 2


def _integer_at_least(value: int, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")


@dataclass(frozen=True, slots=True)
class PublicTask:
    """Generation input.  Evaluator labels must be held in a separate ledger."""

    task_id: str
    prompt: str
    answer_type: str = "number"
    domain: str = "math"
    difficulty: str = "unspecified"


@dataclass(frozen=True, slots=True)
class Observation:
    step: int
    answer: str
    confidence: float | None
    parse_success: bool
    generated_tokens: int
    prompt_tokens: int = 0
    auxiliary_tokens: int = 0
    observed_ns: int = 0
    thought: str = ""
    raw_text: str = ""
    model_stop_flag: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 1:
            raise ValueError("step must be a positive integer")
        for name in ("generated_tokens", "prompt_tokens", "auxiliary_tokens", "observed_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.answer, str) or not isinstance(self.parse_success, bool):
            raise ValueError("answer must be text and parse_success must be a boolean")
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 100
        ):
            raise ValueError("explicit confidence must be finite and in [0, 100]")


@dataclass(frozen=True, slots=True)
class PeerVote:
    peer_id: str
    step: int
    answer: str
    completed_ns: int
    generated_tokens: int


@dataclass(frozen=True, slots=True)
class ClosedPeerPanel:
    """A validated barrier receipt; partial and retrospective panels are illegal.

    ``closed_ns`` and ``decision_ns`` share the observation's monotonic clock.
    Caller must account for generation in every peer, including disagreeing
    peers. A timestamp cannot itself prove provenance: preserve event receipts
    in the prospective protocol ledger when conducting an actual fleet study.
    """

    step: int
    roster: tuple[str, ...]
    votes: tuple[PeerVote, ...]
    closed_ns: int

    def validate(self, step: int, decision_ns: int) -> None:
        _integer_at_least(self.step, "peer panel step", 1)
        _integer_at_least(self.closed_ns, "peer barrier clock", 1)
        _integer_at_least(decision_ns, "decision clock", 1)
        if (not isinstance(self.roster, tuple) or not isinstance(self.votes, tuple)
                or not self.roster or not all(isinstance(peer, str) and peer.strip() for peer in self.roster)
                or len(self.roster) != len(set(self.roster))):
            raise ValueError("peer roster must be frozen, nonempty and unique")
        if any(not isinstance(vote, PeerVote) for vote in self.votes):
            raise ValueError("peer votes must be immutable PeerVote records")
        if self.step != step or {v.peer_id for v in self.votes} != set(self.roster):
            raise ValueError("a full same-step peer roster is required")
        if len(self.votes) != len(self.roster):
            raise ValueError("duplicate/missing peer votes")
        if self.closed_ns <= 0 or decision_ns < self.closed_ns:
            raise ValueError("peer barrier must close before the decision")
        for vote in self.votes:
            _integer_at_least(vote.step, "peer vote step", 1)
            _integer_at_least(vote.completed_ns, "peer completion clock", 1)
            _integer_at_least(vote.generated_tokens, "peer generated tokens", 0)
            if vote.step != step or not 0 < vote.completed_ns <= self.closed_ns:
                raise ValueError("future, stale, or uncompleted peer vote")
            if not isinstance(vote.answer, str) or not vote.answer.strip() or vote.generated_tokens < 0:
                raise ValueError("peer answer and token accounting are required")


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Freeze this object before opening evaluation outcomes.

    Confidence is model-reported and uncalibrated.  Stability can be confidently
    wrong.  ``min_peers=0`` makes no claim about fleet consensus.  Changing any
    parameter creates a different policy hash; Pareto variants are development
    comparisons, not an outer-test threshold search.
    """

    name: str = "confidence_stability_v1"
    mode: str = "heuristic"  # heuristic, never, or fixed
    max_steps: int = 5
    min_steps: int = T_MIN
    fixed_step: int = 2
    confidence_threshold: float = 90.0
    stable_steps: int = 2
    confidence_drop: float = 15.0
    wobble_changes: int = 2
    min_peers: int = 0
    peer_agreement_threshold: float = 1.0

    def __post_init__(self) -> None:
        for name in ("max_steps", "min_steps", "fixed_step", "stable_steps", "wobble_changes", "min_peers"):
            _integer_at_least(getattr(self, name), name, 0)
        if self.mode not in {"heuristic", "never", "fixed"}:
            raise ValueError("unsupported stopping mode")
        if self.min_steps < T_MIN or self.max_steps < self.min_steps:
            raise ValueError("the mandatory minimum is two completed steps")
        if not self.min_steps <= self.fixed_step <= self.max_steps:
            raise ValueError("fixed step must be within the valid horizon")
        if self.stable_steps < 2 or self.wobble_changes < 1 or self.min_peers < 0:
            raise ValueError("invalid stability/peer requirement")
        if not 0 <= self.confidence_threshold <= 100 or not 0 < self.confidence_drop <= 100:
            raise ValueError("invalid confidence thresholds")
        if not 0 <= self.peer_agreement_threshold <= 1:
            raise ValueError("invalid agreement threshold")

    @property
    def sha256(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Decision:
    step: int
    stop: bool
    reason: str
    selected_answer: str
    selected_step: int
    decision_ns: int
    latency_ns: int
    peer_tokens: int = 0
    peer_agreement: float | None = None


def answer_key(answer: str) -> str:
    """Exact, case-insensitive whitespace normalization; never label matching."""

    return " ".join(answer.casefold().split())


class OnlineStoppingController:
    def __init__(self, config: PolicyConfig = PolicyConfig()) -> None:
        self.config = config
        self._prefix: list[Observation] = []
        self._closed = False

    @property
    def prefix(self) -> tuple[Observation, ...]:
        return tuple(self._prefix)

    def observe(
        self, observation: Observation, *, peers: ClosedPeerPanel | None = None, decision_ns: int | None = None
    ) -> Decision:
        started = time.perf_counter_ns()
        now = time.monotonic_ns() if decision_ns is None else decision_ns
        _integer_at_least(now, "decision clock", 1)
        if self._closed:
            raise RuntimeError("cannot feed observations after a terminal STOP")
        if observation.step != len(self._prefix) + 1 or observation.step > self.config.max_steps:
            raise ValueError("observations must arrive once, consecutively, in prefix order")
        if observation.observed_ns > now:
            raise ValueError("future observation")
        peer_tokens = 0
        agreement = None
        if peers is not None:
            peers.validate(observation.step, now)
            peer_tokens = sum(v.generated_tokens for v in peers.votes)
            agreement = sum(answer_key(v.answer) == answer_key(observation.answer) for v in peers.votes) / len(peers.votes)
        self._prefix.append(observation)
        selected = next((o for o in reversed(self._prefix) if o.answer.strip()), observation)
        stop, reason = False, "continue"
        if observation.step == self.config.max_steps:
            stop, reason = True, "terminal_horizon"
        elif observation.step < self.config.min_steps:
            reason = "minimum_two_steps"
        elif self.config.mode == "fixed" and observation.step >= self.config.fixed_step:
            stop, reason = True, "fixed_budget"
        elif self.config.mode == "heuristic":
            valid = observation.parse_success and bool(observation.answer.strip()) and observation.confidence is not None
            previous = self._prefix[-2]
            peers_ready = self.config.min_peers == 0 or (
                peers is not None and len(peers.votes) >= self.config.min_peers
                and agreement is not None and agreement >= self.config.peer_agreement_threshold
            )
            recent = self._prefix[-self.config.stable_steps:]
            stable = len(recent) == self.config.stable_steps and all(
                o.parse_success and bool(o.answer.strip()) and answer_key(o.answer) == answer_key(observation.answer)
                for o in recent
            )
            if valid and stable and observation.confidence >= self.config.confidence_threshold and peers_ready:
                stop, reason = True, "stable_high_confidence"
            elif valid and observation.step >= max(3, self.config.min_steps) and peers_ready:
                if (previous.parse_success and previous.answer.strip() and previous.confidence is not None
                        and previous.confidence - observation.confidence >= self.config.confidence_drop):
                    stop, reason, selected = True, "confidence_drop_retain_previous", previous
                else:
                    valid_prefix = [o for o in self._prefix if o.parse_success and o.answer.strip()]
                    changes = sum(answer_key(a.answer) != answer_key(b.answer) for a, b in zip(valid_prefix, valid_prefix[1:]))
                    if changes >= self.config.wobble_changes:
                        # Deterministic label-blind selection among already observed answers.
                        selected = max(valid_prefix, key=lambda o: (o.confidence if o.confidence is not None else -1, -o.step))
                        stop, reason = True, "answer_wobble_retain_highest_confidence"
            if not valid:
                reason = "incomplete_or_untrusted_parse"
            elif not peers_ready:
                reason = "await_complete_peer_panel"
        self._closed = stop
        return Decision(observation.step, stop, reason, selected.answer, selected.step, now, time.perf_counter_ns() - started, peer_tokens, agreement)


class CancellationToken:
    """Thread-safe token checked by the decoder before each new token."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


class StepGenerator(Protocol):
    def generate_step(self, task: PublicTask, history: Sequence[Observation], step: int, cancellation: CancellationToken) -> Observation: ...
    def cancel(self) -> None: ...


@dataclass(frozen=True, slots=True)
class GenerationResult:
    task_id: str
    observations: tuple[Observation, ...]
    decisions: tuple[Decision, ...]
    answer: str
    selected_step: int
    stopped_at_step: int
    generated_tokens: int
    prompt_tokens: int
    auxiliary_tokens: int
    peer_generated_tokens: int
    elapsed_seconds: float
    cancelled: bool


def run_online_generation(
    task: PublicTask, generator: StepGenerator, config: PolicyConfig = PolicyConfig(), *, cancellation: CancellationToken | None = None
) -> GenerationResult:
    """Actually stop generation: never request or decode any future step.

    No evaluator/label callback is accepted. ``cancel`` also stops pending token
    work in a backend implementing cancellation; this loop itself schedules no
    speculative/background work. External cancellation before the second step
    is an aborted run, not an eligible early-stop decision.
    """

    token = cancellation if cancellation is not None else CancellationToken()
    controller = OnlineStoppingController(config)
    decisions: list[Decision] = []
    started = time.perf_counter()
    try:
        for step in range(1, config.max_steps + 1):
            if token.cancelled:
                raise InterruptedError("generation aborted by caller")
            observation = generator.generate_step(task, controller.prefix, step, token)
            if token.cancelled:
                raise InterruptedError("generation aborted by caller")
            decision = controller.observe(observation)
            decisions.append(decision)
            if decision.stop:
                token.cancel()
                generator.cancel()
                break
    except BaseException:
        token.cancel()
        generator.cancel()
        raise
    prefix = controller.prefix
    final = decisions[-1]
    return GenerationResult(task.task_id, prefix, tuple(decisions), final.selected_answer, final.selected_step,
                            final.step, sum(o.generated_tokens for o in prefix), sum(o.prompt_tokens for o in prefix),
                            sum(o.auxiliary_tokens for o in prefix), sum(d.peer_tokens for d in decisions),
                            time.perf_counter() - started, token.cancelled)
