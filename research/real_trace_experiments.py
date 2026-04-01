from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "real_traces"
STEP_COST = 0.05
SYSTEM_PROMPT = (
    "You are participating in a research protocol about incremental reasoning. "
    "Reply with exactly four lines in this format: THOUGHT: ..., ANSWER: ..., CONFIDENCE: ..., STOP: .... "
    "THOUGHT must be one short sentence. ANSWER must be only the current best final answer. "
    "CONFIDENCE must be an integer from 0 to 100. STOP must be yes or no."
)
DAY_NAMES = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    domain: str
    difficulty: str
    prompt: str
    answer_type: str
    expected_answer: str
    notes: str


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    hf_name: str
    family: str
    parameter_count: str


TASKS = [
    TaskSpec(
        task_id="calendar_offset_easy",
        domain="calendar",
        difficulty="easy",
        prompt=(
            "If today is Tuesday, what day of the week will it be 14 days later? "
            "Answer with only the weekday."
        ),
        answer_type="day",
        expected_answer="tuesday",
        notes="Very easy task where extra verification can become redundant.",
    ),
    TaskSpec(
        task_id="fraction_remaining",
        domain="arithmetic",
        difficulty="medium",
        prompt=(
            "A tank is half full. Then one quarter of the whole tank is drained. "
            "What fraction of the tank is full now?"
        ),
        answer_type="fraction",
        expected_answer="1/4",
        notes="Easy fraction task with exact verification.",
    ),
    TaskSpec(
        task_id="linear_equation",
        domain="algebra",
        difficulty="medium",
        prompt="Solve 2x + 5 = 19. Answer with only the value of x.",
        answer_type="int",
        expected_answer="7",
        notes="Simple algebra task that should be solvable by small instruct models.",
    ),
    TaskSpec(
        task_id="python_trace",
        domain="code",
        difficulty="medium",
        prompt=(
            "What does this Python code print?\n"
            "total = 0\n"
            "for i in range(1, 5):\n"
            "    total += i\n"
            "print(total)\n"
            "Answer with only the printed value."
        ),
        answer_type="int",
        expected_answer="10",
        notes="Straightforward code tracing benchmark.",
    ),
]


MODEL_CATALOG = {
    "deepseek_r1_distill_1p5b": ModelSpec(
        alias="deepseek_r1_distill_1p5b",
        hf_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        family="DeepSeek-R1 distill",
        parameter_count="1.5B",
    ),
    "qwen2p5_0p5b": ModelSpec(
        alias="qwen2p5_0p5b",
        hf_name="Qwen/Qwen2.5-0.5B-Instruct",
        family="Qwen2.5 instruct",
        parameter_count="0.5B",
    ),
}


def normalize_answer(raw_answer: str, answer_type: str) -> str:
    text = raw_answer.strip().lower()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return ""

    if answer_type == "day":
        for day_name in DAY_NAMES:
            if day_name in text:
                return day_name
        return text.split()[0]

    if answer_type == "int":
        match = re.findall(r"-?\d+", text)
        return match[-1] if match else text

    if answer_type == "fraction":
        fraction_match = re.findall(r"-?\d+\s*/\s*-?\d+", text)
        if fraction_match:
            value = fraction_match[-1].replace(" ", "")
            try:
                return str(Fraction(value))
            except ZeroDivisionError:
                return value
        decimal_match = re.findall(r"-?\d+(?:\.\d+)?", text)
        if decimal_match:
            try:
                return str(Fraction(decimal_match[-1]).limit_denominator())
            except ValueError:
                return decimal_match[-1]
        return text

    return text


def verify_answer(task: TaskSpec, candidate_answer: str) -> bool:
    return normalize_answer(candidate_answer, task.answer_type) == normalize_answer(task.expected_answer, task.answer_type)


def utility(correct: bool, step: int) -> float:
    return float(correct) - STEP_COST * (step - 1)


def lexical_overlap(current_text: str, prior_text: str) -> float:
    current_tokens = set(re.findall(r"[a-z0-9]+", current_text.lower()))
    prior_tokens = set(re.findall(r"[a-z0-9]+", prior_text.lower()))
    if not current_tokens or not prior_tokens:
        return 0.0
    return len(current_tokens & prior_tokens) / len(current_tokens | prior_tokens)


def conversation_prompt(task: TaskSpec, history: list[dict[str, Any]], step: int, max_steps: int, prompt_mode: str) -> str:
    if history:
        rendered_history = "\n".join(
            (
                f"Step {item['step']}: thought={item['thought']} | answer={item['answer']} | "
                f"confidence={item['confidence']} | stop={item['model_stop_flag']}"
            )
            for item in history
        )
    else:
        rendered_history = "No previous steps."

    protocol_text = (
        f"Research protocol: you are at incremental reasoning step {step} of {max_steps}. "
        "Even if you already know the answer, continue with one short reasoning or verification step. "
        "Do not repeat the full solution. Update the answer field every time.\n\n"
        f"Previous steps:\n{rendered_history}\n\n"
    )

    if prompt_mode == "structured_four_line":
        format_instr = (
            "Return exactly four lines:\n"
            "THOUGHT: <one short sentence>\n"
            "ANSWER: <current best final answer only>\n"
            "CONFIDENCE: <integer 0-100>\n"
            "STOP: <yes or no>"
        )
    elif prompt_mode == "minimal_json":
        format_instr = (
            "Return a single JSON object with the following keys: 'thought' (string), 'answer' (string), "
            "'confidence' (integer 0-100), and 'stop' (boolean)."
        )
    else:
        format_instr = "Provide the final answer."

    return (
        f"Task id: {task.task_id}\n"
        f"Domain: {task.domain}\n"
        f"Difficulty band: {task.difficulty}\n"
        f"Task: {task.prompt}\n\n"
        f"{protocol_text if prompt_mode != 'answer_only' else ''}"
        f"{format_instr}"
    )


def render_prompt(tokenizer: Any, user_prompt: str, system_prompt_mode: str) -> str:
    if system_prompt_mode == "default":
        sys_msg = SYSTEM_PROMPT
    elif system_prompt_mode == "short":
        sys_msg = "You are a logical reasoning assistant."
    else:
        sys_msg = ""
        
    messages = []
    if sys_msg:
        messages.append({"role": "system", "content": sys_msg})
    messages.append({"role": "user", "content": user_prompt})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if sys_msg:
        return f"System: {sys_msg}\nUser: {user_prompt}\nAssistant:"
    return f"User: {user_prompt}\nAssistant:"


def parse_generation(raw_text: str, answer_type: str, prompt_mode: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```\w*\n", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n```$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    result = {
        "thought": "",
        "answer": "",
        "confidence": 50,
        "stop": False,
        "parse_success": 0,
        "output_format_type": "fallback_freeform",
        "answer_extraction_source": "fallback",
        "stop_extraction_source": "default",
        "confidence_extraction_source": "default",
    }
    
    if prompt_mode == "structured_four_line":
        line_thought = re.search(r"^THOUGHT\s*:\s*(.*)$", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        line_answer = re.search(r"^ANSWER\s*:\s*(.*)$", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        line_confidence = re.search(r"^CONFIDENCE\s*:\s*(-?\d+(?:\.\d+)?)", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        line_stop = re.search(r"^STOP\s*:\s*(yes|no|true|false)", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        
        if line_thought and line_answer and line_confidence and line_stop:
            result["parse_success"] = 1
            result["output_format_type"] = "structured_exact"
        elif line_thought or line_answer or line_confidence or line_stop:
            result["output_format_type"] = "structured_partial"
            
        if line_thought:
            result["thought"] = line_thought.group(1).strip()
        if line_answer:
            result["answer"] = line_answer.group(1).strip()
            result["answer_extraction_source"] = "ANSWER_line"
        if line_confidence:
            result["confidence"] = int(float(line_confidence.group(1)))
            result["confidence_extraction_source"] = "CONFIDENCE_line"
        if line_stop:
            result["stop"] = line_stop.group(1).lower() in {"yes", "true"}
            result["stop_extraction_source"] = "STOP_line"

    elif prompt_mode == "minimal_json":
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                parsed = json.loads(candidate)
                result["thought"] = str(parsed.get("thought", "")).strip()
                result["answer"] = str(parsed.get("answer", "")).strip()
                result["confidence"] = int(float(parsed.get("confidence", 50)))
                result["stop"] = bool(parsed.get("stop", False))
                result["parse_success"] = 1
                result["output_format_type"] = "json_exact"
                result["answer_extraction_source"] = "json_field"
                result["confidence_extraction_source"] = "json_field"
                result["stop_extraction_source"] = "json_field"
            except (json.JSONDecodeError, TypeError, ValueError):
                result["output_format_type"] = "json_partial"

    if not result["answer"]:
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if lines:
            result["answer"] = lines[-1]
            result["answer_extraction_source"] = "last_line_fallback"
            
    if not result["thought"]:
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if lines:
            result["thought"] = lines[0]

    return result


def load_model(model_spec: ModelSpec, device: str) -> tuple[Any, Any, str, str]:
    tokenizer = AutoTokenizer.from_pretrained(model_spec.hf_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "auto":
        actual_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        actual_device = device
    backend = "transformers+torch(cpu)"
    load_kwargs: dict[str, Any] = {"trust_remote_code": True}

    try:
        if actual_device == "cuda":
            load_kwargs["torch_dtype"] = torch.float16
            model = AutoModelForCausalLM.from_pretrained(model_spec.hf_name, **load_kwargs)
            model.to("cuda")
            backend = "transformers+torch(cuda)"
        else:
            model = AutoModelForCausalLM.from_pretrained(model_spec.hf_name, **load_kwargs)
            model.to("cpu")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_spec.hf_name, trust_remote_code=True)
        model.to("cpu")
        actual_device = "cpu"
        backend = "transformers+torch(cpu-fallback)"

    model.eval()
    return model, tokenizer, actual_device, backend


def generate_with_diagnostics(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    actual_device: str,
    temperature: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(actual_device)
    attention_mask = encoded["attention_mask"].to(actual_device)

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.02,
    }
    if temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.95})
    else:
        generation_kwargs.update({"do_sample": False})

    with torch.no_grad():
        generated_ids = model.generate(**generation_kwargs)
        prompt_length = input_ids.shape[1]
        completion_ids = generated_ids[0, prompt_length:]
        raw_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        forward_outputs = model(generated_ids, attention_mask=torch.ones_like(generated_ids), output_hidden_states=True)
        full_logits = forward_outputs.logits[0]
        generated_length = completion_ids.shape[0]
        if generated_length > 0:
            scoring_logits = full_logits[prompt_length - 1 : generated_ids.shape[1] - 1]
            log_probs = torch.log_softmax(scoring_logits, dim=-1)
            probs = torch.softmax(scoring_logits, dim=-1)
            token_logprobs = log_probs.gather(1, completion_ids.unsqueeze(1)).squeeze(1)
            token_entropies = -(probs * log_probs).sum(dim=-1)
            pooled_hidden = forward_outputs.hidden_states[-1][0, prompt_length:, :].mean(dim=0).float().cpu().numpy()
        else:
            token_logprobs = torch.empty(0)
            token_entropies = torch.empty(0)
            pooled_hidden = np.zeros((model.config.hidden_size,), dtype=np.float32)

    return {
        "raw_text": raw_text,
        "generated_tokens": int(generated_length),
        "mean_token_logprob": float(token_logprobs.mean().item()) if len(token_logprobs) else float("nan"),
        "mean_entropy": float(token_entropies.mean().item()) if len(token_entropies) else float("nan"),
        "entropy_std": float(token_entropies.std().item()) if len(token_entropies) > 1 else 0.0,
        "pooled_hidden": pooled_hidden,
    }


def run_single_trace(
    model: Any,
    tokenizer: Any,
    model_spec: ModelSpec,
    task: TaskSpec,
    actual_device: str,
    temperature: float,
    seed: int,
    max_steps: int,
    max_new_tokens: int,
    hidden_dir: Path,
    prompt_mode: str,
    system_prompt_mode: str,
    is_baseline: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    set_seed(seed)
    history: list[dict[str, Any]] = []
    hidden_vectors: list[np.ndarray] = []
    run_rows: list[dict[str, Any]] = []

    steps_to_run = 1 if is_baseline else max_steps

    for step in range(1, steps_to_run + 1):
        actual_prompt_mode = "answer_only" if is_baseline else prompt_mode
        user_prompt = conversation_prompt(
            task=task,
            history=history,
            step=step,
            max_steps=max_steps,
            prompt_mode=actual_prompt_mode
        )
        prompt_text = render_prompt(tokenizer, user_prompt, system_prompt_mode)
        started_at = time.perf_counter()
        generated = generate_with_diagnostics(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            actual_device=actual_device,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        elapsed_seconds = time.perf_counter() - started_at
        
        parsed = parse_generation(generated["raw_text"], task.answer_type, actual_prompt_mode)
        normalized_answer = normalize_answer(parsed["answer"], task.answer_type)
        is_correct = verify_answer(task, parsed["answer"])
        prior_answer = history[-1]["answer_normalized"] if history else ""
        answer_changed = bool(history) and normalized_answer != prior_answer and normalized_answer != ""
        prior_thought = " ".join(item["thought"] for item in history[-2:])
        
        raw_text_length_chars = len(generated["raw_text"])
        hit_max_new_tokens = int(generated["generated_tokens"] == max_new_tokens)
        truncated_output_suspected = int(hit_max_new_tokens and not parsed["parse_success"])
        
        hidden_vector = generated["pooled_hidden"]
        if hidden_vectors:
            previous_hidden = hidden_vectors[-1]
            denominator = float(np.linalg.norm(hidden_vector) * np.linalg.norm(previous_hidden))
            hidden_cosine_shift = 1.0 - float(np.dot(hidden_vector, previous_hidden) / denominator) if denominator else 0.0
            hidden_l2_shift = float(np.linalg.norm(hidden_vector - previous_hidden))
        else:
            hidden_cosine_shift = 0.0
            hidden_l2_shift = 0.0
        hidden_vectors.append(hidden_vector)

        row = {
            "run_id": f"{model_spec.alias}__{task.task_id}__temp{temperature:.2f}__seed{seed}",
            "model_alias": model_spec.alias,
            "model_name": model_spec.hf_name,
            "task_id": task.task_id,
            "domain": task.domain,
            "difficulty": task.difficulty,
            "step": step,
            "thought": parsed["thought"],
            "answer": parsed["answer"],
            "answer_normalized": normalized_answer,
            "correct": int(is_correct),
            "confidence": int(max(0, min(100, parsed["confidence"]))),
            "model_stop_flag": int(bool(parsed["stop"])),
            "answer_changed": int(answer_changed),
            "thought_token_count": int(len(re.findall(r"\w+", parsed["thought"]))),
            "raw_generation_tokens": generated["generated_tokens"],
            "mean_token_logprob": generated["mean_token_logprob"],
            "entropy_mean": generated["mean_entropy"],
            "entropy_std": generated["entropy_std"],
            "hidden_norm": float(np.linalg.norm(hidden_vector)),
            "hidden_l2_shift": hidden_l2_shift,
            "hidden_cosine_shift": hidden_cosine_shift,
            "lexical_echo": lexical_overlap(parsed["thought"], prior_thought),
            "verbose_confidence_proxy": float(parsed["confidence"]) / 100.0 + 0.01 * generated["generated_tokens"],
            "utility": utility(correct=is_correct, step=step),
            "elapsed_seconds": elapsed_seconds,
            "seed": seed,
            "temperature": temperature,
            "device": actual_device,
            "prompt_mode": actual_prompt_mode,
            "system_prompt_mode": system_prompt_mode,
            "is_baseline": int(is_baseline),
            "parse_success": parsed["parse_success"],
            "output_format_type": parsed["output_format_type"],
            "answer_extraction_source": parsed["answer_extraction_source"],
            "stop_extraction_source": parsed["stop_extraction_source"],
            "confidence_extraction_source": parsed["confidence_extraction_source"],
            "hit_max_new_tokens": hit_max_new_tokens,
            "truncated_output_suspected": truncated_output_suspected,
            "raw_text_length_chars": raw_text_length_chars,
            "raw_text_length_tokens": generated["generated_tokens"],
            "raw_text": generated["raw_text"],
        }
        history.append(row)
        run_rows.append(row)

    hidden_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(hidden_dir / f"{run_rows[0]['run_id']}.npz", hidden_states=np.stack(hidden_vectors))

    first_correct_step = next((row["step"] for row in run_rows if row["correct"] == 1), -1)
    first_model_stop_step = next((row["step"] for row in run_rows if row["model_stop_flag"] == 1), -1)
    oracle_stop = max(run_rows, key=lambda row: row["utility"])["step"]
    revision_count = int(sum(row["answer_changed"] for row in run_rows))
    
    run_summary = {
        "run_id": run_rows[0]["run_id"],
        "model_alias": model_spec.alias,
        "model_name": model_spec.hf_name,
        "task_id": task.task_id,
        "domain": task.domain,
        "difficulty": task.difficulty,
        "temperature": temperature,
        "seed": seed,
        "prompt_mode": actual_prompt_mode,
        "system_prompt_mode": system_prompt_mode,
        "is_baseline": int(is_baseline),
        "ever_correct": int(first_correct_step != -1),
        "correct_at_step_1": int(run_rows[0]["correct"] == 1) if run_rows else 0,
        "oracle_stop": oracle_stop,
        "first_correct_step": first_correct_step if first_correct_step != -1 else max_steps,
        "first_model_stop_step": first_model_stop_step if first_model_stop_step != -1 else max_steps,
        "revision_count": revision_count,
        "best_utility": max(row["utility"] for row in run_rows),
        "final_correct": run_rows[-1]["correct"],
        "device": actual_device,
    }
    return run_rows, run_summary


def summarize_transitions(step_frame: pd.DataFrame) -> pd.DataFrame:
    ordered = step_frame.sort_values(["run_id", "step"]).copy()
    ordered["next_correct"] = ordered.groupby("run_id")["correct"].shift(-1)
    ordered = ordered.dropna(subset=["next_correct"])
    ordered["next_correct"] = ordered["next_correct"].astype(int)
    ordered["repair"] = ((ordered["correct"] == 0) & (ordered["next_correct"] == 1)).astype(int)
    ordered["corruption"] = ((ordered["correct"] == 1) & (ordered["next_correct"] == 0)).astype(int)
    rows: list[dict[str, Any]] = []
    for step, group in ordered.groupby("step"):
        wrong_group = group[group["correct"] == 0]
        correct_group = group[group["correct"] == 1]
        n_repairs = wrong_group["repair"].sum()
        n_corruptions = correct_group["corruption"].sum()
        repair_rate = float(wrong_group["repair"].mean()) if len(wrong_group) else float("nan")
        corruption_rate = float(correct_group["corruption"].mean()) if len(correct_group) else float("nan")
        q_t = float(group["correct"].mean())
        if (n_repairs + n_corruptions) >= 3 and not math.isnan(repair_rate) and not math.isnan(corruption_rate):
            hazard_mu = (1.0 - q_t) * repair_rate - q_t * corruption_rate - STEP_COST
        else:
            hazard_mu = float("nan")
        rows.append(
            {
                "step": int(step),
                "q_t": q_t,
                "n_repairs": int(n_repairs),
                "n_corruptions": int(n_corruptions),
                "repair_rate": repair_rate,
                "corruption_rate": corruption_rate,
                "hazard_mu": hazard_mu,
                "entropy_mean": float(group["entropy_mean"].mean()),
                "confidence_mean": float(group["confidence"].mean()),
                "answer_changed_rate": float(group["answer_changed"].mean()),
                "hidden_shift_mean": float(group["hidden_l2_shift"].mean()),
                "n_transitions": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small real-trace pilot on an open-weight reasoning model.")
    parser.add_argument("--model", default="deepseek_r1_distill_1p5b", choices=sorted(MODEL_CATALOG.keys()))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.6])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-tasks", type=int, default=len(TASKS))
    parser.add_argument("--prompt-mode", default="structured_four_line", choices=["structured_four_line", "minimal_json", "answer_only"])
    parser.add_argument("--system-prompt-mode", default="default", choices=["default", "short", "none"])
    parser.add_argument("--run-baseline", action="store_true", help="Run competence baseline instead of iterative reasoning")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent generation threads")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir = output_dir / "hidden_states"
    model_spec = MODEL_CATALOG[args.model]
    model, tokenizer, actual_device, backend = load_model(model_spec=model_spec, device=args.device)

    all_rows: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []

    tasks_to_run = []
    for task in TASKS[: args.max_tasks]:
        for temperature in args.temperatures:
            for seed in args.seeds:
                tasks_to_run.append((task, temperature, seed))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for task, temperature, seed in tasks_to_run:
            kwargs = dict(
                model=model,
                tokenizer=tokenizer,
                model_spec=model_spec,
                task=task,
                actual_device=actual_device,
                temperature=temperature,
                seed=seed,
                max_steps=args.max_steps,
                max_new_tokens=args.max_new_tokens,
                hidden_dir=hidden_dir,
                prompt_mode=args.prompt_mode,
                system_prompt_mode=args.system_prompt_mode,
                is_baseline=args.run_baseline,
            )
            futures.append(executor.submit(run_single_trace, **kwargs))

        for future in concurrent.futures.as_completed(futures):
            try:
                rows, run_summary = future.result()
                all_rows.extend(rows)
                all_runs.append(run_summary)
            except Exception as exc:
                print(f"A trace generated an exception: {exc}", flush=True)

    step_frame = pd.DataFrame(all_rows)
    run_frame = pd.DataFrame(all_runs)
    
    if args.run_baseline:
        file_prefix = "baseline_"
        step_frame.to_csv(output_dir / f"{file_prefix}steps.csv", index=False)
        run_frame.to_csv(output_dir / f"{file_prefix}runs.csv", index=False)
        print(f"Wrote baseline artifacts to: {output_dir}")
        return

    transition_frame = summarize_transitions(step_frame)
    pilot_summary = pd.DataFrame(
        [
            {
                "model_alias": model_spec.alias,
                "model_name": model_spec.hf_name,
                "family": model_spec.family,
                "parameter_count": model_spec.parameter_count,
                "n_runs": int(len(run_frame)),
                "n_tasks": int(step_frame["task_id"].nunique()),
                "mean_oracle_stop": float(run_frame["oracle_stop"].mean()),
                "mean_first_correct_step": float(run_frame["first_correct_step"].mean()),
                "mean_model_stop_step": float(run_frame["first_model_stop_step"].mean()),
                "mean_revision_count": float(run_frame["revision_count"].mean()),
                "mean_entropy": float(step_frame["entropy_mean"].mean()),
                "mean_hidden_shift": float(step_frame["hidden_l2_shift"].mean()),
                "repair_rate_overall": float(transition_frame["repair_rate"].dropna().mean()),
                "corruption_rate_overall": float(transition_frame["corruption_rate"].dropna().mean()),
                "runs_ever_correct": int(run_frame["ever_correct"].sum()),
                "backend": backend,
                "device": actual_device,
            }
        ]
    )

    step_frame.to_csv(output_dir / "trace_steps.csv", index=False)
    run_frame.to_csv(output_dir / "trace_runs.csv", index=False)
    transition_frame.to_csv(output_dir / "hazard_by_step.csv", index=False)
    pilot_summary.to_csv(output_dir / "pilot_summary.csv", index=False)

    metadata = {
        "model": asdict(model_spec),
        "backend": backend,
        "device": actual_device,
        "temperatures": args.temperatures,
        "seeds": args.seeds,
        "max_steps": args.max_steps,
        "max_new_tokens": args.max_new_tokens,
        "step_cost": STEP_COST,
        "prompt_mode": args.prompt_mode,
        "system_prompt_mode": args.system_prompt_mode,
        "tasks": [asdict(task) for task in TASKS[: args.max_tasks]],
        "hidden_state_accessible": True,
        "token_logprobs_accessible": True,
        "reasoning_traces_accessible": True,
        "runtime_limit_notes": "Pilot sized for local Windows workstation; fixed-step protocol is used to observe post-answer revisions.",
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    for _, row in pilot_summary.iterrows():
        print(
            f"{row['model_alias']}: runs={int(row['n_runs'])}, "
            f"ever_correct={int(row['runs_ever_correct'])}, "
            f"oracle={row['mean_oracle_stop']:.2f}, first_correct={row['mean_first_correct_step']:.2f}, "
            f"repair={row['repair_rate_overall']:.3f}, corruption={row['corruption_rate_overall']:.3f}, "
            f"device={row['device']}"
        )
    print(f"Wrote real-trace artifacts to: {output_dir}")


if __name__ == "__main__":
    main()