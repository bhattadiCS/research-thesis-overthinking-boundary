"""Deterministic completed-response generator fixtures; no model work."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research"))
from online_stopping_controller import Observation


def observation(step, answer="A", confidence=95, *, parsed=True, tokens=None):
    return Observation(
        step=step, answer=answer, confidence=confidence, parse_success=parsed,
        generated_tokens=[11, 13, 29, 31, 47][step - 1] if tokens is None else tokens,
        prompt_tokens=100 + step * step,
        thought=f"Check response {step}.", raw_text=f"raw response {step}: {answer}",
    )


class ScriptedGenerator:
    """Check exact observed prefixes and expose cancellation scheduling."""

    def __init__(self, observations=None, *, fail_at=None, block_at=None,
                 before_call=None, after_call=None):
        self.observations = observations or [observation(i) for i in range(1, 6)]
        self.fail_at = fail_at
        self.block_at = block_at
        self.before_call = before_call
        self.after_call = after_call
        self.calls = []
        self.cancel_count = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.current_token = None

    def generate_step(self, task, history, step, cancellation):
        if cancellation.cancelled:
            raise AssertionError("backend scheduled a cancelled generation")
        assert list(history) == self.observations[:step - 1], "generator received a changed/future prefix"
        if self.before_call:
            self.before_call(step, cancellation)
        self.calls.append((step, tuple(history), task))
        self.current_token = cancellation
        try:
            if self.fail_at == step:
                raise RuntimeError("deterministic decoder failure")
            if self.block_at == step:
                self.entered.set()
                while not self.release.wait(.01):
                    if cancellation.cancelled:
                        raise InterruptedError("deterministic decoding cancelled")
            if cancellation.cancelled:
                raise InterruptedError("cancelled before completed observation")
            result = self.observations[step - 1]
            if self.after_call:
                self.after_call(step, cancellation)
            return result
        finally:
            self.current_token = None

    def cancel(self):
        self.cancel_count += 1
        if self.current_token is not None:
            self.current_token.cancel()


class ScriptedPredictor:
    artifact_sha256 = "a" * 64

    def __init__(self, q=.8, next_probability=.7, *, values=None):
        self.q = q
        self.next_probability = next_probability
        self.values = values or {}
        self.calls = []

    def estimate(self, prefix, domain):
        assert len(prefix) < 5, "next-step head was called at terminal horizon"
        self.calls.append((tuple(prefix), domain))
        q, next_probability = self.values.get(len(prefix), (self.q, self.next_probability))
        return SimpleNamespace(
            q_current=q, p_next=next_probability, gain=999,
            artifact_sha256=self.artifact_sha256, feature_contract="deterministic-test-prefix",
        )
