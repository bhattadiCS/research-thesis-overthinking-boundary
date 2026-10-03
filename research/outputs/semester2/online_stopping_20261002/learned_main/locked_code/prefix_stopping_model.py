"""Portable, label-blind prefix probability model for the online controller.

Runtime requires only the Python standard library and the Observation schema.
Fitting, calibration and evaluation belong in train_prefix_stopping_model.py.
Probabilities are fitted estimates, not certified conditional posteriors.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from online_stopping_controller import Observation, answer_key


FEATURE_CONTRACT = "observation-prefix-probability-v1"
ARTIFACT_SCHEMA = "prefix-stopping-probabilities-v1"
FEATURE_NAMES = (
    "step", "parse_success", "answer_present", "confidence_centered",
    "confidence_missing", "confidence_delta", "prefix_confidence_mean",
    "log_generated_tokens", "log_cumulative_generated_tokens", "answer_changed",
    "answer_streak", "prefix_answer_changes", "log_answer_characters",
    "log_thought_characters", "thought_numeric_density", "thought_uncertainty_density",
    "thought_verification_density", "domain_gsm8k", "domain_math",
    "step_confidence_interaction", "streak_confidence_interaction",
)
SUPPORTED_DOMAINS = ("gsm8k", "math")
_UNCERTAINTY = frozenset(("maybe", "uncertain", "unsure", "however", "alternatively", "doubt"))
_VERIFY = frozenset(("check", "verify", "recheck", "calculate", "confirm", "verification"))


def _validate_prefix(prefix: Sequence[Observation]) -> None:
    if not prefix or len(prefix) > 5:
        raise ValueError("a nonempty prefix of at most five observations is required")
    if any(not isinstance(o, Observation) for o in prefix):
        raise TypeError("the estimator accepts Observation objects, not labeled trace rows")
    if [o.step for o in prefix] != list(range(1, len(prefix) + 1)):
        raise ValueError("observations must be consecutive and begin at step one")


def prefix_features(prefix: Sequence[Observation], domain: str) -> tuple[float, ...]:
    """Use only current/prior observations, with no fit-dependent transforms."""
    _validate_prefix(prefix)
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError(f"unsupported model domain: {domain!r}")
    current = prefix[-1]
    keys = [answer_key(o.answer) for o in prefix]
    changed = len(keys) > 1 and keys[-1] != keys[-2]
    streak = 1
    for key in reversed(keys[:-1]):
        if key != keys[-1]:
            break
        streak += 1
    trusted = [
        float(o.confidence) / 100 for o in prefix
        if o.parse_success and o.confidence is not None
    ]
    confidence = (
        (float(current.confidence) - 50) / 50
        if current.parse_success and current.confidence is not None else 0.0
    )
    delta = 0.0
    if len(prefix) > 1:
        previous = prefix[-2]
        if (previous.parse_success and current.parse_success
                and previous.confidence is not None and current.confidence is not None):
            delta = (float(current.confidence) - float(previous.confidence)) / 100
    words = re.findall(r"[a-z]+", current.thought.casefold())
    denominator = max(1, len(words))
    values = (
        float(current.step), float(current.parse_success), float(bool(current.answer.strip())),
        confidence, float(not (current.parse_success and current.confidence is not None)),
        delta, sum(trusted) / len(trusted) if trusted else 0.5,
        math.log1p(current.generated_tokens),
        math.log1p(sum(o.generated_tokens for o in prefix)),
        float(changed), float(streak),
        float(sum(a != b for a, b in zip(keys, keys[1:]))),
        math.log1p(len(current.answer)), math.log1p(len(current.thought)),
        sum(ch.isdigit() for ch in current.thought) / max(1, len(current.thought)),
        sum(word in _UNCERTAINTY for word in words) / denominator,
        sum(word in _VERIFY for word in words) / denominator,
        float(domain == "gsm8k"), float(domain == "math"),
        current.step * confidence, streak * confidence,
    )
    if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite or misaligned prefix features")
    return values


def _sigmoid(value: float) -> float:
    if value >= 0:
        tail = math.exp(-value)
        return 1 / (1 + tail)
    tail = math.exp(value)
    return tail / (1 + tail)


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True, slots=True)
class Estimate:
    q_current: float
    p_next: float
    gain: float
    artifact_sha256: str
    feature_contract: str = FEATURE_CONTRACT


class PrefixStoppingModel:
    """A frozen pair of serialized logistic models and Platt calibrators.

    The current target is correctness of the latest nonempty candidate
    available at the prefix. The next target uses that same selection rule
    after one more increment. The caller supplies the known public domain.
    The controller, rather than this estimator, enforces the stopping floor.
    """

    def __init__(self, artifact: dict, *, artifact_sha256: str | None = None):
        if artifact.get("schema") != ARTIFACT_SCHEMA:
            raise ValueError("unsupported prefix model artifact schema")
        if artifact.get("feature_contract") != FEATURE_CONTRACT:
            raise ValueError("feature contract mismatch")
        if artifact.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("feature names/order mismatch")
        if artifact.get("domains") != list(SUPPORTED_DOMAINS):
            raise ValueError("unsupported artifact domain contract")
        if artifact.get("min_steps") != 2 or artifact.get("max_steps") != 5:
            raise ValueError("model horizon must be min two and max five")
        self.step_cost = _finite_number(artifact.get("step_cost"), "step_cost")
        if self.step_cost < 0:
            raise ValueError("step cost must be nonnegative")
        self._components = {}
        for name in ("q_current", "p_next"):
            component = artifact.get("models", {}).get(name)
            if not isinstance(component, dict):
                raise ValueError(f"missing model component {name}")
            normalized = {}
            for key in ("mean", "scale", "coefficient"):
                values = component.get(key)
                if not isinstance(values, list) or len(values) != len(FEATURE_NAMES):
                    raise ValueError(f"{name}.{key} dimension mismatch")
                normalized[key] = tuple(_finite_number(v, f"{name}.{key}") for v in values)
            if any(value <= 0 for value in normalized["scale"]):
                raise ValueError("scaler denominators must be positive")
            normalized["intercept"] = _finite_number(component.get("intercept"), f"{name}.intercept")
            calibrator = component.get("calibration", {})
            normalized["slope"] = _finite_number(calibrator.get("slope"), f"{name}.calibration.slope")
            normalized["offset"] = _finite_number(calibrator.get("intercept"), f"{name}.calibration.intercept")
            self._components[name] = normalized
        self.artifact = artifact
        encoded = json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.artifact_sha256 = artifact_sha256 or hashlib.sha256(encoded).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "PrefixStoppingModel":
        raw = Path(path).read_bytes()
        return cls(json.loads(raw), artifact_sha256=hashlib.sha256(raw).hexdigest())

    def _score(self, component_name: str, features: tuple[float, ...], *, calibrated: bool = True) -> float:
        component = self._components[component_name]
        logit = component["intercept"] + sum(
            weight * ((value - mean) / scale)
            for value, mean, scale, weight in zip(
                features, component["mean"], component["scale"], component["coefficient"],
            )
        )
        if calibrated:
            logit = component["slope"] * logit + component["offset"]
        probability = _sigmoid(logit)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("model emitted an invalid probability")
        return probability

    def score_current(self, prefix: Sequence[Observation], domain: str) -> float:
        return self._score("q_current", prefix_features(prefix, domain))

    def estimate(self, prefix: Sequence[Observation], domain: str) -> Estimate:
        features = prefix_features(prefix, domain)
        if len(prefix) == 5:
            raise ValueError("no next-step target exists at the terminal horizon")
        q = self._score("q_current", features)
        next_q = self._score("p_next", features)
        return Estimate(q, next_q, next_q - q - self.step_cost, self.artifact_sha256)
