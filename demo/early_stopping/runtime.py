"""Live, shared-prefix comparison. No training, grading, archive replay or writes.

Every response is actually generated once. Both policies receive the identical
prefix while the early arm is open. After its STOP, its prefix/answer/costs are
frozen, and only the full-horizon arm receives further generated responses.
Physical work is therefore the baseline's work, not the sum of logical arms.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
import hashlib
import json
import math
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / 'research'
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from learned_online_stopping_controller import LearnedOnlineStoppingController, LearnedPolicy
from online_stopping_controller import (CancellationToken, OnlineStoppingController,
                                        PolicyConfig, PublicTask, answer_key)
from prefix_stopping_model import PrefixStoppingModel

DEFAULT_MODEL = Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775'
DEFAULT_PREDICTOR = RESEARCH / 'outputs/semester2/prefix_model_v1/prefix_model.json'
MAX_STEPS = 5
TERMINAL = frozenset(('complete', 'cancelled', 'error'))
COMPARISON_NOTE = ('Both sides share the same generated prefix. After early STOP, only the full run continues. '
                   'Token savings compare the two policies; running this comparison still generates the full run. '
                   'Matching answers does not establish correctness.')


@dataclass(frozen=True)
class RunConfig:
    prompt: str
    domain: str = 'math'
    policy: str = 'learned'
    step_cost: float = .05

    def __post_init__(self):
        if not isinstance(self.prompt, str) or not self.prompt.strip() or len(self.prompt) > 3000:
            raise ValueError('Enter a prompt of 1 to 3,000 characters.')
        if self.domain not in ('math', 'gsm8k'):
            raise ValueError('Choose general math or arithmetic.')
        if self.policy not in ('learned', 'heuristic'):
            raise ValueError('Choose learned or heuristic stopping.')
        if (isinstance(self.step_cost, bool) or not isinstance(self.step_cost, (int, float))
                or not math.isfinite(self.step_cost) or not 0 <= self.step_cost <= 1):
            raise ValueError('Step cost must be a finite number between 0 and 1.')

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict) or set(payload) - {'prompt', 'domain', 'policy', 'step_cost'}:
            raise ValueError('Unexpected comparison settings.')
        return cls(**payload)


class ComparisonRunner:
    def __init__(self, generator, predictor=None):
        self.generator, self.predictor = generator, predictor

    def run(self, config: RunConfig, emit: Callable, cancellation: CancellationToken):
        if config.policy == 'learned':
            if self.predictor is None:
                raise ValueError('The frozen prefix predictor is unavailable.')
            early = LearnedOnlineStoppingController(self.predictor, config.domain,
                LearnedPolicy(self.predictor.artifact_sha256, step_cost=config.step_cost))
        else:
            early = OnlineStoppingController(PolicyConfig())
        baseline = OnlineStoppingController(PolicyConfig(name='never', mode='never'))
        task = PublicTask(uuid.uuid4().hex, config.prompt.strip(), 'number' if config.domain == 'gsm8k' else 'math', config.domain)
        opened = True
        started = time.perf_counter()
        early_final = baseline_final = None
        early_elapsed = baseline_elapsed = 0.

        def totals(controller):
            return dict(steps=len(controller.prefix),
                        completion_tokens=sum(o.generated_tokens for o in controller.prefix),
                        prompt_tokens=sum(o.prompt_tokens for o in controller.prefix))

        try:
            for step in range(1, MAX_STEPS+1):
                if cancellation.cancelled:
                    raise InterruptedError('Comparison cancelled')
                arms = ['early', 'baseline'] if opened else ['baseline']
                emit('response_start', step=step, arms=arms)
                if hasattr(self.generator, 'set_callback'):
                    self.generator.set_callback(lambda text, count, s=step, a=arms:
                        emit('token', step=s, arms=a, text=text, generated_tokens=count))
                step_start = time.perf_counter()
                observation = self.generator.generate_step(task, baseline.prefix, step, cancellation)
                if cancellation.cancelled:
                    raise InterruptedError('Comparison cancelled')
                emit('response_end', step=step, arms=arms, observation=asdict(observation),
                     step_seconds=time.perf_counter()-step_start)
                baseline_final = baseline.observe(observation)
                baseline_elapsed = time.perf_counter()-started
                emit('decision', arm='baseline', decision=asdict(baseline_final),
                     totals=totals(baseline), elapsed_seconds=baseline_elapsed)
                if opened:
                    early_final = early.observe(observation)
                    early_elapsed = time.perf_counter()-started
                    emit('decision', arm='early', decision=asdict(early_final),
                         totals=totals(early), elapsed_seconds=early_elapsed)
                    opened = not early_final.stop
            if early_final is None or baseline_final is None or not early_final.stop or not baseline_final.stop:
                raise RuntimeError('Comparison did not reach its terminal decisions.')

            def final(controller, decision, elapsed):
                t = totals(controller)
                return dict(answer=decision.selected_answer, selected_step=decision.selected_step,
                            stopped_at_step=decision.step, reason=decision.reason,
                            completion_tokens=t['completion_tokens'], prompt_tokens=t['prompt_tokens'],
                            elapsed_seconds=elapsed)
            left = final(early, early_final, early_elapsed)
            right = final(baseline, baseline_final, baseline_elapsed)
            saved = right['completion_tokens']-left['completion_tokens']
            return dict(early=left, baseline=right, completion_tokens_saved=saved,
                        completion_savings=saved/right['completion_tokens'] if right['completion_tokens'] else 0.,
                        physical_completion_tokens=right['completion_tokens'],
                        physical_prompt_tokens=right['prompt_tokens'],
                        answers_match=bool(left['answer'].strip() and right['answer'].strip()) and answer_key(left['answer']) == answer_key(right['answer']),
                        comparison_note=COMPARISON_NOTE)
        finally:
            if hasattr(self.generator, 'set_callback'):
                self.generator.set_callback(None)
            self.generator.cancel()


class RunState:
    def __init__(self, config: RunConfig):
        self.id = uuid.uuid4().hex
        self.config = config
        self.status = 'queued'
        self.events: list[dict[str, Any]] = []
        self.summary = None
        self.error = None
        self.cancellation = CancellationToken()
        self.condition = threading.Condition()

    def emit(self, event_type, **fields):
        with self.condition:
            if self.status in TERMINAL:
                return
            if event_type in TERMINAL:
                self.status = event_type
                self.summary = fields.get('summary')
                self.error = fields.get('message') if event_type == 'error' else None
            elif event_type == 'run_started':
                self.status = 'running'
            event = dict(seq=len(self.events)+1, type=event_type, run_id=self.id, **fields)
            self.events.append(event)
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return dict(id=self.id, status=self.status, prompt=self.config.prompt,
                        config=asdict(self.config), events=list(self.events),
                        summary=self.summary, error=self.error, last_seq=len(self.events))

    def events_since(self, seq: int, timeout: float = 10.):
        with self.condition:
            if len(self.events) <= seq and self.status not in TERMINAL:
                self.condition.wait(timeout)
            return list(self.events[seq:]), self.status in TERMINAL


class ComparisonManager:
    def __init__(self, model_path=DEFAULT_MODEL, device='cuda', predictor_path=DEFAULT_PREDICTOR,
                 generator_factory=None):
        self.model_path = Path(model_path).resolve()
        self.device = device
        self.predictor_path = Path(predictor_path).resolve()
        self.predictor = None
        self.predictor_error = None
        if self.predictor_path.is_file():
            try:
                self.predictor = PrefixStoppingModel.load(self.predictor_path)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self.predictor_error = str(exc)
        self.generator_factory = generator_factory
        self.generator = None
        self.runs: dict[str, RunState] = {}
        self.lock = threading.RLock()
        self.worker = None
        self.active_id = None
        self.latest_id = None

    def status(self):
        with self.lock:
            latest = self.runs.get(self.latest_id)
            return dict(model=dict(path=str(self.model_path), name='Qwen2.5-0.5B-Instruct' if 'Qwen2.5-0.5B' in str(self.model_path) else self.model_path.name,
                                   available=self.generator_factory is not None or (self.model_path/'config.json').is_file(),
                                   loaded=self.generator is not None, device=self.device),
                        predictor=dict(available=self.predictor is not None,
                                       sha256=self.predictor.artifact_sha256 if self.predictor else None,
                                       error=self.predictor_error),
                        busy=self.active_id is not None,
                        run=dict(id=latest.id, status=latest.status) if latest else None)

    def get(self, run_id):
        with self.lock:
            if run_id not in self.runs:
                raise KeyError('Comparison not found. It may have expired after restarting the server.')
            return self.runs[run_id]

    def start(self, payload):
        config = payload if isinstance(payload, RunConfig) else RunConfig.from_payload(payload)
        with self.lock:
            if self.active_id is not None:
                raise RuntimeError('A comparison is already running. Wait or cancel it first.')
            if not self.status()['model']['available']:
                raise FileNotFoundError('Local model snapshot is missing. Start the server with --model-path pointing to a complete local model.')
            if config.policy == 'learned' and self.predictor is None:
                raise ValueError('Frozen prefix predictor is missing or invalid. Choose the heuristic policy or check --predictor-path.')
            # Keep a bounded in-memory history; no prompt or research-output files.
            while len(self.runs) >= 30:
                self.runs.pop(next(iter(self.runs)))
            state = RunState(config)
            self.runs[state.id] = state
            self.active_id = self.latest_id = state.id
            self.worker = threading.Thread(target=self._work, args=(state,), daemon=True)
            self.worker.start()
            return state

    def cancel(self, run_id):
        state = self.get(run_id)
        state.cancellation.cancel()
        return state

    def _work(self, state):
        try:
            if self.generator is None:
                state.status = 'loading'
                state.emit('status', message='Loading the local model. The first run can take a moment.')
                if self.generator_factory is None:
                    if __package__:
                        from .streaming_generator import StreamingStepGenerator
                    else:
                        from streaming_generator import StreamingStepGenerator
                    self.generator = StreamingStepGenerator(self.model_path, device=self.device, max_steps=5, max_new_tokens=128)
                else:
                    self.generator = self.generator_factory()
            if state.cancellation.cancelled:
                raise InterruptedError('Comparison cancelled')
            state.emit('run_started', prompt=state.config.prompt, policy=state.config.policy,
                       domain=state.config.domain, max_steps=5, min_steps=2,
                       step_cost=state.config.step_cost, model_name=self.status()['model']['name'])
            summary = ComparisonRunner(self.generator, self.predictor).run(state.config, state.emit, state.cancellation)
            if state.cancellation.cancelled:
                raise InterruptedError('Comparison cancelled')
            state.emit('complete', summary=summary)
        except InterruptedError as exc:
            state.emit('cancelled', message=str(exc), summary=None)
        except Exception as exc:
            # Details stay in terminal; the UI receives an actionable error.
            import traceback
            traceback.print_exc()
            message = str(exc)
            if 'out of memory' in message.lower():
                message = 'GPU memory is full. Close other model apps or restart this demo with --device cpu.'
            state.emit('error', message=message or type(exc).__name__, summary=None)
        finally:
            with self.lock:
                self.active_id = None

    def close(self):
        with self.lock:
            if self.active_id:
                self.runs[self.active_id].cancellation.cancel()
