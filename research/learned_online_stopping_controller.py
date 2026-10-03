"""Prefix-trained expected-drift stopping without changing frozen v1 runtime.

The unweighted, separately calibrated predictor is injected as an immutable
artifact. Runtime input is an observed prefix and public benchmark domain.
Training labels may estimate q_t and P(correct_{t+1}|prefix); none are accepted
by this controller. The expected gain is recomputed from those probabilities,
not from a supplied classifier decision or a retrospective final-sequence head.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

from online_stopping_controller import Decision, Observation, OnlineStoppingController, PolicyConfig


class PrefixPredictor(Protocol):
    def estimate(self, prefix: Sequence[Observation], domain: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class LearnedPolicy:
    predictor_sha256: str
    name: str = "learned_prefix_drift_v1"
    max_steps: int = 5
    min_steps: int = 2
    reward_value: float = 1.0
    wrong_penalty: float = 0.0
    step_cost: float = 0.05

    def __post_init__(self) -> None:
        for name in ("min_steps", "max_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.min_steps < 2 or self.max_steps < self.min_steps:
            raise ValueError("learned controller requires at least two completed steps")
        if len(self.predictor_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.predictor_sha256):
            raise ValueError("frozen predictor SHA256 is required")
        if any(not math.isfinite(v) or v < 0 for v in (self.reward_value, self.wrong_penalty, self.step_cost)):
            raise ValueError("utility constants must be finite and nonnegative")
        scale = self.reward_value + self.wrong_penalty
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("the reward/penalty scale must be finite and positive")
        if not math.isfinite(scale + self.step_cost):
            raise ValueError("utility constants must permit finite drift at every valid probability")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class LearnedDecision(Decision):
    q_current: float | None = None
    p_next: float | None = None
    estimated_drift: float | None = None
    predictor_sha256: str | None = None
    feature_contract: Any = None


class LearnedOnlineStoppingController:
    def __init__(self, predictor: PrefixPredictor, domain: str, policy: LearnedPolicy) -> None:
        self.predictor, self.domain, self.policy = predictor, domain, policy
        # Reuse the audited observation/terminal/answer-selection invariants.
        self._base = OnlineStoppingController(PolicyConfig(name="never", mode="never", max_steps=policy.max_steps, min_steps=policy.min_steps, fixed_step=policy.min_steps))
        self._closed = False

    @property
    def prefix(self) -> tuple[Observation, ...]:
        return self._base.prefix

    def observe(self, observation: Observation) -> LearnedDecision:
        started = time.perf_counter_ns()
        if self._closed:
            raise RuntimeError("cannot feed future observations after learned STOP")
        base = self._base.observe(observation)
        q, p_next, mu, contract = None, None, None, None
        stop, reason = base.stop, base.reason
        # The terminal branch precedes inference. There is no identifiable
        # archived next-step target at the horizon and no extrapolated p_next.
        if observation.step >= self.policy.min_steps and not base.stop:
            estimate = self.predictor.estimate(self.prefix, self.domain)
            if estimate.artifact_sha256 != self.policy.predictor_sha256:
                raise ValueError("predictor changed after policy freeze")
            q, p_next = float(estimate.q_current), float(estimate.p_next)
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (q, p_next)):
                raise ValueError("prefix estimator must return finite probabilities in [0,1]")
            mu = (p_next - q) * (self.policy.reward_value + self.policy.wrong_penalty) - self.policy.step_cost
            stop, reason = mu <= 0, "estimated_drift_nonpositive" if mu <= 0 else "estimated_drift_positive"
            contract = estimate.feature_contract
        self._closed = stop
        return LearnedDecision(
            step=base.step, stop=stop, reason=reason, selected_answer=base.selected_answer, selected_step=base.selected_step,
            decision_ns=time.monotonic_ns(), latency_ns=time.perf_counter_ns() - started,
            q_current=q, p_next=p_next, estimated_drift=mu, predictor_sha256=self.policy.predictor_sha256,
            feature_contract=contract,
        )
