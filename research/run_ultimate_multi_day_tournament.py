#!/usr/bin/env python3
"""Causal, task-grouped stopping-probe tournament for the ResearchThesis corpus.

This is deliberately a replacement for the older ``run_*tournament.py`` sketches.
Those scripts concatenate source cells and then group sequences by the raw ``run_id``;
raw IDs are reused across cells, so that operation joins incompatible traces.  This
runner makes the source cell part of every trajectory key and evaluates deployment
safe, prefix-causal probes only.

The scientific contract is:

* an observation at t is a function only of x_1, ..., x_t;
* outer and inner cross-validation are GroupKFold splits on raw ``task_id``;
* imputation/scaling, temperature calibration, and every learned component are fit
  only on the corresponding training task groups;
* correctness uncertainty (a Beta-Bernoulli mean/concentration) is distinct from
  transition hazards.  The deployed one-step stopping gain is

      mu_t = (1 - q_t) P(y_{t+1}=1 | y_t=0, x_<=t)
             - q_t P(y_{t+1}=0 | y_t=1, x_<=t) - c_step.

The code contains a pure-PyTorch selective diagonal SSM fallback rather than
silently requiring mamba_ssm, a causal spectral/Fourier operator, a real
Vietoris--Rips persistence calculation for short sliding windows, a five-expert
MoE, Optuna/SQLite resume support, and a Blackwell-oriented BF16 execution path.

The historical 0.8656 BiGRU score in ``advanced_tournament_results.log`` is not a
valid online stopping benchmark: the old BiGRU sees the full future trace and the
old loader merges source-cell collisions.  This runner intentionally does not try
to reproduce that number.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import gc
import hashlib
import itertools
import json
import logging
import math
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

try:  # Optuna is intentionally optional for --self-test and data audits.
    import optuna
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler
except ImportError:  # pragma: no cover - exercised only on minimally provisioned hosts.
    optuna = None
    HyperbandPruner = None
    TPESampler = None


LOG = logging.getLogger("ultimate_tournament")

T_MIN = 2
STEP_COST = 0.05
AVG_TOKENS_PER_STEP = 250.0
TOKEN_PRICE = STEP_COST / AVG_TOKENS_PER_STEP
EPS = 1.0e-8
PROJECTION_DIM = 64

BASE_NUMERIC_COLUMNS = [
    "step",
    "entropy_mean",
    "entropy_std",
    "confidence",
    "answer_changed",
    "thought_token_count",
    "raw_generation_tokens",
    "mean_token_logprob",
    "hidden_norm",
    "hidden_l2_shift",
    "hidden_cosine_shift",
    "lexical_echo",
    "verbose_confidence_proxy",
    "elapsed_seconds",
    "tokens_per_second",
    "k2_agreement",
    "k2_raw_generation_tokens",
    "answer_span_mean_logprob",
    "answer_span_min_logprob",
    "answer_span_mean_entropy",
    "answer_span_std_entropy",
    "hit_max_new_tokens",
    "truncated_output_suspected",
]


@dataclass
class SequenceStore:
    """Right-padded, source-qualified trajectory tensors held on host memory."""

    x: np.ndarray  # [runs, time, features], float32
    y: np.ndarray  # [runs, time], int64
    next_y: np.ndarray  # [runs, time], int64; ignored at final step
    lengths: np.ndarray  # [runs], int64
    row_ids: np.ndarray  # [runs, time], dataframe row IDs, -1 when padded
    steps: np.ndarray  # [runs, time]
    thought_tokens: np.ndarray  # [runs, time]
    k2_tokens: np.ndarray  # [runs, time]
    k2_agreement: np.ndarray  # [runs, time]
    trajectory_ids: np.ndarray  # [runs], object
    task_ids: np.ndarray  # [runs], object
    source_cells: np.ndarray  # [runs], object
    feature_names: list[str]

    @property
    def n_runs(self) -> int:
        return int(self.x.shape[0])

    @property
    def max_len(self) -> int:
        return int(self.x.shape[1])

    def valid_mask_np(self, indices: np.ndarray | None = None) -> np.ndarray:
        lengths = self.lengths if indices is None else self.lengths[indices]
        return np.arange(self.max_len)[None, :] < lengths[:, None]


@dataclass
class ModelConfig:
    d_model: int = 256
    dropout: float = 0.10
    tcn_blocks: int = 4
    transformer_layers: int = 2
    attention_heads: int = 8
    ssm_state: int = 24
    fno_modes: int = 8
    lr: float = 2.0e-3
    weight_decay: float = 1.0e-4
    hazard_weight: float = 0.50
    brier_weight: float = 0.10
    moe_balance_weight: float = 0.010
    concentration_weight: float = 0.002
    beta_variance_weight: float = 0.0
    mine_weight: float = 0.0
    gate_temperature: float = 1.0


class TelemetryDB:
    """Small durable SQLite ledger; unlike Optuna's DB it is human-queryable."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                ts REAL NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trials (
                fold INTEGER NOT NULL,
                trial_number INTEGER NOT NULL,
                state TEXT NOT NULL,
                value REAL,
                params TEXT NOT NULL,
                ts REAL NOT NULL,
                PRIMARY KEY (fold, trial_number)
            );
            CREATE TABLE IF NOT EXISTS folds (
                fold INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                ts REAL NOT NULL
            );
            """
        )
        self.conn.commit()

    def put_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO run_meta(key, value) VALUES (?, ?)",
            (key, json.dumps(value, sort_keys=True, default=str)),
        )
        self.conn.commit()

    def event(self, kind: str, payload: Any) -> None:
        self.conn.execute(
            "INSERT INTO events(ts, kind, payload) VALUES (?, ?, ?)",
            (time.time(), kind, json.dumps(payload, sort_keys=True, default=str)),
        )
        self.conn.commit()

    def trial(self, fold: int, number: int, state: str, value: float | None, params: Any) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO trials(fold, trial_number, state, value, params, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (fold, number, state, value, json.dumps(params, sort_keys=True), time.time()),
        )
        self.conn.commit()

    def fold(self, fold: int, status: str, payload: Any) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO folds(fold, status, payload, ts) VALUES (?, ?, ?, ?)",
            (fold, status, json.dumps(payload, sort_keys=True, default=str), time.time()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "ultimate_tournament_runtime.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOG.addHandler(console)
    LOG.addHandler(file_handler)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def safe_float_series(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Return numeric values plus an explicit missingness indicator.

    Missing telemetry must never be made indistinguishable from an observed zero.
    Values are provisionally zero-filled here; the paired missingness feature lets a
    fold-local learned model distinguish that sentinel after robust scaling.
    """

    if column not in frame.columns:
        return np.zeros(len(frame), dtype=np.float32), np.ones(len(frame), dtype=np.float32)
    numeric = pd.to_numeric(frame[column], errors="coerce")
    missing = (~np.isfinite(numeric.to_numpy(dtype=float))).astype(np.float32)
    values = numeric.fillna(0.0).to_numpy(dtype=np.float32)
    values[~np.isfinite(values)] = 0.0
    return values, missing


def parse_projection_column(frame: pd.DataFrame, column: str, dim: int = PROJECTION_DIM) -> tuple[np.ndarray, np.ndarray]:
    """Parse fixed-width CSV projection strings with an availability flag.

    A malformed or absent vector produces zeros *and* availability=0, rather than a
    silent pseudo-observation.  The canonical V2 cells use valid 64-dimensional
    vectors, while several legacy cells do not contain these columns.
    """

    result = np.zeros((len(frame), dim), dtype=np.float32)
    available = np.zeros(len(frame), dtype=np.float32)
    if column not in frame.columns:
        return result, available
    values = frame[column].to_numpy()
    for index, raw in enumerate(values):
        if not isinstance(raw, str) or not raw.strip():
            continue
        text = raw.strip().strip("[]")
        parsed = np.fromstring(text, sep=",", dtype=np.float32)
        if parsed.size != dim or not np.all(np.isfinite(parsed)):
            continue
        result[index] = parsed
        available[index] = 1.0
    return result, available


def _softmax_np(vector: np.ndarray) -> np.ndarray:
    shifted = vector.astype(np.float64) - float(np.max(vector))
    exp = np.exp(np.clip(shifted, -60.0, 60.0))
    return exp / max(float(exp.sum()), EPS)


def _vietoris_rips_barcodes(points: np.ndarray) -> dict[int, list[tuple[float, float]]]:
    """Compute H0/H1 Z2 barcodes for a tiny Vietoris--Rips complex.

    Trace windows are short (normally five points), so an exact reduction of the
    boundary matrix is cheaper and more reproducible than an undeclared dependency
    on ripser/gudhi.  Vertices, edges, and triangles suffice to obtain H0/H1 bars.
    """

    count = int(points.shape[0])
    if count == 0:
        return {0: [], 1: []}
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    simplices: list[tuple[float, int, tuple[int, ...]]] = []
    for vertex in range(count):
        simplices.append((0.0, 0, (vertex,)))
    for left, right in itertools.combinations(range(count), 2):
        simplices.append((float(distances[left, right]), 1, (left, right)))
    for tri in itertools.combinations(range(count), 3):
        filtration = max(float(distances[a, b]) for a, b in itertools.combinations(tri, 2))
        simplices.append((filtration, 2, tri))
    simplices.sort(key=lambda item: (item[0], item[1], item[2]))
    simplex_index = {simplex: index for index, (_, _, simplex) in enumerate(simplices)}
    reduced_columns: list[set[int]] = []
    low_to_column: dict[int, int] = {}
    for column, (_, dimension, simplex) in enumerate(simplices):
        if dimension == 0:
            boundary: set[int] = set()
        else:
            boundary = {
                simplex_index[tuple(face)]
                for face in itertools.combinations(simplex, dimension)
            }
        while boundary and max(boundary) in low_to_column:
            boundary ^= reduced_columns[low_to_column[max(boundary)]]
        if boundary:
            low_to_column[max(boundary)] = column
        reduced_columns.append(boundary)
    births_to_deaths = {birth: death for birth, death in low_to_column.items()}
    bars: dict[int, list[tuple[float, float]]] = {0: [], 1: []}
    for index, column in enumerate(reduced_columns):
        filtration, dimension, _ = simplices[index]
        if dimension <= 1 and not column:
            death_index = births_to_deaths.get(index)
            death = math.inf if death_index is None else simplices[death_index][0]
            bars[dimension].append((float(filtration), float(death)))
    return bars


def persistent_window_features(points: np.ndarray) -> tuple[float, float, float]:
    """Return Betti_0, Betti_1 and total finite H1 persistence at a natural scale."""

    if len(points) < 2:
        return 0.0, 0.0, 0.0
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    positive = distances[np.triu_indices(len(points), k=1)]
    positive = positive[positive > EPS]
    scale = float(np.median(positive)) if positive.size else 0.0
    bars = _vietoris_rips_barcodes(points)
    beta0 = sum(1 for birth, death in bars[0] if birth <= scale < death)
    beta1 = sum(1 for birth, death in bars[1] if birth <= scale < death)
    finite_h1 = sum(max(0.0, death - birth) for birth, death in bars[1] if math.isfinite(death))
    return float(beta0), float(beta1), float(finite_h1)


def build_causal_geometry_phase_features(
    frame: pd.DataFrame,
    state: np.ndarray,
    layer_one: np.ndarray,
    layer_two: np.ndarray,
    include_persistence: bool,
    topology_window: int,
) -> pd.DataFrame:
    """Construct causal geometry, information, and phase-space features.

    The metric is conformal, g(s)=exp(2 phi(s)) I with
    phi(s)=0.15*tanh(mean(s)).  Its Levi-Civita acceleration is exactly
    a_cov=a+2(v.grad phi)v-||v||^2 grad phi, avoiding an unjustified dense metric
    estimate.  A true Lie bracket is not identifiable from one sampled curve, so
    the reported ``commutator_proxy`` is explicitly a discrete tangent-turn proxy.
    """

    rows, dimension = state.shape
    values: dict[str, np.ndarray] = {
        name: np.zeros(rows, dtype=np.float32)
        for name in [
            "delta_t",
            "traj_vel_norm",
            "traj_acc_norm",
            "traj_jerk_norm",
            "traj_curvature",
            "traj_torsion",
            "covariant_acc_norm",
            "geodesic_speed",
            "conformal_phi",
            "commutator_proxy",
            "layer_velocity_alignment",
            "phase_attractor_distance",
            "phase_energy",
            "renyi_state_divergence",
            "entropy_tail_kurtosis_proxy",
            "kinematic_v_available",
            "kinematic_a_available",
            "kinematic_j_available",
            "topology_available",
        ]
    }
    if include_persistence:
        values.update(
            {
                "persistent_beta0": np.zeros(rows, dtype=np.float32),
                "persistent_beta1": np.zeros(rows, dtype=np.float32),
                "persistent_h1_total": np.zeros(rows, dtype=np.float32),
            }
        )

    # Grouping is source-qualified and the frame is already sorted by trajectory/step.
    for _, group in frame.groupby("trajectory_id", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        sequence = state[indices].astype(np.float64, copy=False)
        one = layer_one[indices].astype(np.float64, copy=False)
        two = layer_two[indices].astype(np.float64, copy=False)
        length = len(indices)
        steps = group["step"].to_numpy(dtype=np.float64)
        delta = np.ones(length, dtype=np.float64)
        if length > 1:
            delta[1:] = np.maximum(steps[1:] - steps[:-1], 1.0)
        velocity = np.zeros_like(sequence)
        acceleration = np.zeros_like(sequence)
        jerk = np.zeros_like(sequence)
        if length > 1:
            velocity[1:] = (sequence[1:] - sequence[:-1]) / delta[1:, None]
        if length > 2:
            acceleration[2:] = (velocity[2:] - velocity[1:-1]) / delta[2:, None]
        if length > 3:
            jerk[3:] = (acceleration[3:] - acceleration[2:-1]) / delta[3:, None]
        velocity_one = np.zeros_like(one)
        velocity_two = np.zeros_like(two)
        if length > 1:
            velocity_one[1:] = (one[1:] - one[:-1]) / delta[1:, None]
            velocity_two[1:] = (two[1:] - two[:-1]) / delta[1:, None]

        norm_v = np.linalg.norm(velocity, axis=1)
        norm_a = np.linalg.norm(acceleration, axis=1)
        norm_j = np.linalg.norm(jerk, axis=1)
        phi = 0.15 * np.tanh(sequence.mean(axis=1))
        gradient_phi_scalar = 0.15 * (1.0 - np.tanh(sequence.mean(axis=1)) ** 2) / dimension
        gradient_phi = np.repeat(gradient_phi_scalar[:, None], dimension, axis=1)
        covariant = acceleration + 2.0 * np.sum(velocity * gradient_phi, axis=1, keepdims=True) * velocity
        covariant -= (norm_v**2)[:, None] * gradient_phi

        values["delta_t"][indices] = delta
        values["traj_vel_norm"][indices] = norm_v
        values["traj_acc_norm"][indices] = norm_a
        values["traj_jerk_norm"][indices] = norm_j
        values["covariant_acc_norm"][indices] = np.linalg.norm(covariant, axis=1)
        values["geodesic_speed"][indices] = np.exp(phi) * norm_v
        values["conformal_phi"][indices] = phi
        values["phase_energy"][indices] = 0.5 * norm_v**2 + 0.025 * np.sum((sequence - sequence[0]) ** 2, axis=1)
        values["kinematic_v_available"][indices[1:]] = 1.0
        if length > 2:
            values["kinematic_a_available"][indices[2:]] = 1.0
        if length > 3:
            values["kinematic_j_available"][indices[3:]] = 1.0

        for local in range(1, length):
            first_norm = np.linalg.norm(velocity_one[local])
            second_norm = np.linalg.norm(velocity_two[local])
            if first_norm * second_norm > EPS:
                values["layer_velocity_alignment"][indices[local]] = float(
                    np.dot(velocity_one[local], velocity_two[local]) / (first_norm * second_norm)
                )
            p = _softmax_np(sequence[local])
            q = _softmax_np(sequence[local - 1])
            alpha = 1.5
            values["renyi_state_divergence"][indices[local]] = float(
                np.log(np.sum(np.power(p, alpha) * np.power(q, 1.0 - alpha)) + EPS) / (alpha - 1.0)
            )
        for local in range(2, length):
            v = velocity[local]
            a = acceleration[local]
            v_previous = velocity[local - 1]
            a_previous = acceleration[local - 1]
            v2 = float(np.dot(v, v))
            a2 = float(np.dot(a, a))
            va = float(np.dot(v, a))
            area_sq = max(v2 * a2 - va * va, 0.0)
            values["traj_curvature"][indices[local]] = math.sqrt(area_sq) / (max(v2, EPS) ** 1.5 + EPS)
            gram = np.array(
                [[v2, va, float(np.dot(v, jerk[local]))], [va, a2, float(np.dot(a, jerk[local]))],
                 [float(np.dot(jerk[local], v)), float(np.dot(jerk[local], a)), float(np.dot(jerk[local], jerk[local]))]],
                dtype=np.float64,
            )
            values["traj_torsion"][indices[local]] = math.sqrt(max(float(np.linalg.det(gram)), 0.0)) / (area_sq + EPS)
            values["commutator_proxy"][indices[local]] = float(np.linalg.norm(v * a_previous - v_previous * a))
            current_phase = np.concatenate([sequence[local], velocity[local], acceleration[local]])
            past_phase = np.concatenate(
                [sequence[: local - 1], velocity[: local - 1], acceleration[: local - 1]], axis=1
            )
            if len(past_phase):
                values["phase_attractor_distance"][indices[local]] = float(
                    np.min(np.linalg.norm(past_phase - current_phase, axis=1))
                )
        if include_persistence:
            for local in range(length):
                left = max(0, local - topology_window + 1)
                window = sequence[left : local + 1]
                beta0, beta1, h1_total = persistent_window_features(window)
                values["persistent_beta0"][indices[local]] = beta0
                values["persistent_beta1"][indices[local]] = beta1
                values["persistent_h1_total"][indices[local]] = h1_total
                if len(window) >= 2:
                    values["topology_available"][indices[local]] = 1.0

    entropy_mean, _ = safe_float_series(frame, "entropy_mean")
    entropy_std, _ = safe_float_series(frame, "entropy_std")
    values["entropy_tail_kurtosis_proxy"] = (
        np.square(entropy_std / (np.abs(entropy_mean) + 1.0e-3))
    ).astype(np.float32)
    return pd.DataFrame(values, index=frame.index)


def build_feature_frame(
    frame: pd.DataFrame,
    include_persistence: bool,
    topology_window: int,
) -> pd.DataFrame:
    """Build all unsupervised per-step features before fold-local scaling."""

    feature_data: dict[str, np.ndarray] = {}
    for column in BASE_NUMERIC_COLUMNS:
        numeric, missing = safe_float_series(frame, column)
        feature_data[column] = numeric
        feature_data[f"{column}__missing"] = missing
    mid_one, one_available = parse_projection_column(frame, "mid_hidden_1_proj")
    mid_two, two_available = parse_projection_column(frame, "mid_hidden_2_proj")
    for index in range(PROJECTION_DIM):
        feature_data[f"mid1_{index:03d}"] = mid_one[:, index]
        feature_data[f"mid2_{index:03d}"] = mid_two[:, index]
    feature_data["mid1_available"] = one_available
    feature_data["mid2_available"] = two_available
    feature_data["both_mid_available"] = one_available * two_available

    # s_t is 128-dimensional: the concatenation preserves both mid-layer views.
    geometry = build_causal_geometry_phase_features(
        frame,
        np.concatenate([mid_one, mid_two], axis=1),
        mid_one,
        mid_two,
        include_persistence=include_persistence,
        topology_window=topology_window,
    )
    feature_data.update({column: geometry[column].to_numpy(dtype=np.float32) for column in geometry.columns})
    result = pd.DataFrame(feature_data, index=frame.index)
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return result


def select_trace_paths(input_dir: Path, include_all_cells: bool, max_cells: int | None) -> list[Path]:
    pattern = "**/trace_steps.csv" if include_all_cells else "global_*/trace_steps.csv"
    paths = sorted(input_dir.glob(pattern))
    if max_cells is not None:
        paths = paths[: max(1, max_cells)]
    if not paths:
        scope = "all cells" if include_all_cells else "canonical global_* cells"
        raise FileNotFoundError(f"No {scope} trace_steps.csv files under {input_dir}")
    return paths


def load_trace_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    input_dir = Path(args.input_dir)
    paths = select_trace_paths(input_dir, args.include_all_cells, args.max_cells)
    frames: list[pd.DataFrame] = []
    files: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(input_dir).as_posix()
        source_cell = path.parent.relative_to(input_dir).as_posix()
        try:
            cell = pd.read_csv(path, low_memory=False)
        except Exception as error:
            raise RuntimeError(f"Unable to load {path}: {error}") from error
        cell["source_cell"] = source_cell
        cell["source_path"] = relative
        frames.append(cell)
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    frame = pd.concat(frames, ignore_index=True, sort=False)
    required = {"run_id", "task_id", "step", "correct", "source_cell"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Trace corpus lacks required columns: {missing}")
    frame["run_id"] = frame["run_id"].astype(str)
    frame["task_id"] = frame["task_id"].astype(str)
    frame["trajectory_id"] = frame["source_cell"].astype(str) + "::" + frame["run_id"]
    frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
    frame["correct"] = pd.to_numeric(frame["correct"], errors="coerce")
    if frame["step"].isna().any() or frame["correct"].isna().any():
        raise ValueError("step/correct contains non-numeric values; refusing to fabricate labels or ordering")
    frame["step"] = frame["step"].astype(int)
    frame["correct"] = frame["correct"].astype(int)
    if not frame["correct"].isin([0, 1]).all():
        invalid = sorted(frame.loc[~frame["correct"].isin([0, 1]), "correct"].unique().tolist())
        raise ValueError(f"correct must be binary; observed {invalid[:10]}")
    frame = frame.sort_values(["trajectory_id", "step"], kind="stable").reset_index(drop=True)
    duplicate = frame.duplicated(["trajectory_id", "step"], keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, ["trajectory_id", "step"]].head(10).to_dict("records")
        raise ValueError(f"Duplicate source-qualified trajectory steps: {examples}")
    task_cardinality = frame.groupby("trajectory_id", sort=False)["task_id"].nunique()
    if (task_cardinality != 1).any():
        bad = task_cardinality[task_cardinality != 1].head().to_dict()
        raise ValueError(f"A trajectory maps to multiple task IDs: {bad}")
    frame["row_id"] = np.arange(len(frame), dtype=np.int64)
    raw_collisions = int((frame.groupby("run_id")["source_cell"].nunique() > 1).sum())
    manifest: dict[str, Any] = {
        "input_dir": str(input_dir),
        "include_all_cells": bool(args.include_all_cells),
        "selected_cell_count": len(paths),
        "files": files,
        "rows": int(len(frame)),
        "source_qualified_trajectories": int(frame["trajectory_id"].nunique()),
        "raw_run_ids": int(frame["run_id"].nunique()),
        "raw_run_id_cross_cell_collisions": raw_collisions,
        "task_ids": int(frame["task_id"].nunique()),
        "sequence_length_distribution": {
            str(length): int(count)
            for length, count in frame.groupby("trajectory_id", sort=False).size().value_counts().sort_index().items()
        },
    }
    manifest["dataset_fingerprint"] = stable_json_hash(manifest["files"])
    LOG.info(
        "Loaded %d cells, %d source-qualified trajectories, %d rows, %d task groups (raw-ID collisions=%d).",
        manifest["selected_cell_count"],
        manifest["source_qualified_trajectories"],
        manifest["rows"],
        manifest["task_ids"],
        raw_collisions,
    )
    return frame, manifest


def build_sequence_store(frame: pd.DataFrame, features: pd.DataFrame) -> SequenceStore:
    if len(frame) != len(features):
        raise ValueError("Feature and trace rows are misaligned")
    groups = list(frame.groupby("trajectory_id", sort=False))
    run_count = len(groups)
    max_len = max(len(group) for _, group in groups)
    feature_values = features.to_numpy(dtype=np.float32, copy=True)
    x = np.zeros((run_count, max_len, feature_values.shape[1]), dtype=np.float32)
    y = np.zeros((run_count, max_len), dtype=np.int64)
    next_y = np.zeros((run_count, max_len), dtype=np.int64)
    lengths = np.zeros(run_count, dtype=np.int64)
    row_ids = np.full((run_count, max_len), -1, dtype=np.int64)
    steps = np.zeros((run_count, max_len), dtype=np.int64)
    thought_tokens = np.zeros((run_count, max_len), dtype=np.float32)
    k2_tokens = np.zeros((run_count, max_len), dtype=np.float32)
    k2_agreement = np.zeros((run_count, max_len), dtype=np.int64)
    trajectory_ids: list[str] = []
    task_ids: list[str] = []
    source_cells: list[str] = []
    raw_thought, _ = safe_float_series(frame, "thought_token_count")
    raw_k2, _ = safe_float_series(frame, "k2_raw_generation_tokens")
    raw_agreement, _ = safe_float_series(frame, "k2_agreement")
    for run_index, (trajectory_id, group) in enumerate(groups):
        indices = group.index.to_numpy(dtype=np.int64)
        length = len(indices)
        label = group["correct"].to_numpy(dtype=np.int64)
        x[run_index, :length] = feature_values[indices]
        y[run_index, :length] = label
        if length > 1:
            next_y[run_index, : length - 1] = label[1:]
        next_y[run_index, length - 1] = label[-1]
        lengths[run_index] = length
        row_ids[run_index, :length] = group["row_id"].to_numpy(dtype=np.int64)
        steps[run_index, :length] = group["step"].to_numpy(dtype=np.int64)
        thought_tokens[run_index, :length] = raw_thought[indices]
        k2_tokens[run_index, :length] = raw_k2[indices]
        k2_agreement[run_index, :length] = raw_agreement[indices].astype(np.int64)
        trajectory_ids.append(str(trajectory_id))
        task_ids.append(str(group["task_id"].iloc[0]))
        source_cells.append(str(group["source_cell"].iloc[0]))
    return SequenceStore(
        x=x,
        y=y,
        next_y=next_y,
        lengths=lengths,
        row_ids=row_ids,
        steps=steps,
        thought_tokens=thought_tokens,
        k2_tokens=k2_tokens,
        k2_agreement=k2_agreement,
        trajectory_ids=np.asarray(trajectory_ids, dtype=object),
        task_ids=np.asarray(task_ids, dtype=object),
        source_cells=np.asarray(source_cells, dtype=object),
        feature_names=list(features.columns),
    )


class FoldRobustScaler:
    """Median/IQR scaler fit solely on valid timesteps of outer-training runs."""

    def __init__(self) -> None:
        self.median: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, x: np.ndarray, lengths: np.ndarray, indices: np.ndarray) -> "FoldRobustScaler":
        selected = x[indices]
        mask = np.arange(x.shape[1])[None, :] < lengths[indices, None]
        flat = selected[mask]
        if len(flat) == 0:
            raise ValueError("Cannot scale an empty training fold")
        self.median = np.nanmedian(flat, axis=0).astype(np.float32)
        q25 = np.nanpercentile(flat, 25.0, axis=0).astype(np.float32)
        q75 = np.nanpercentile(flat, 75.0, axis=0).astype(np.float32)
        self.scale = np.maximum(q75 - q25, 1.0e-3).astype(np.float32)
        return self

    def transform(self, x: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        if self.median is None or self.scale is None:
            raise RuntimeError("FoldRobustScaler.transform called before fit")
        result = (x - self.median[None, None, :]) / self.scale[None, None, :]
        result = np.clip(np.nan_to_num(result, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
        mask = np.arange(x.shape[1])[None, :] < lengths[:, None]
        result[~mask] = 0.0
        return np.ascontiguousarray(result.astype(np.float32))


def torch_valid_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


class TensorSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, x: np.ndarray, y: np.ndarray, next_y: np.ndarray, lengths: np.ndarray) -> None:
        self.x = torch.from_numpy(np.ascontiguousarray(x))
        self.y = torch.from_numpy(np.ascontiguousarray(y))
        self.next_y = torch.from_numpy(np.ascontiguousarray(next_y))
        self.lengths = torch.from_numpy(np.ascontiguousarray(lengths))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index], self.next_y[index], self.lengths[index]


class FixedBatchSampler(Sampler[list[int]]):
    """Fixed-size shuffled batches; pads only the last batch by resampling training runs.

    Fixed shapes allow Tensor Core kernels, torch.compile and optional CUDA graph
    replay without dropping any tail observations.
    """

    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        self.size = int(size)
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        generator = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = generator.permutation(self.size)
        for start in range(0, self.size, self.batch_size):
            batch = order[start : start + self.batch_size]
            if len(batch) < self.batch_size:
                fill = generator.choice(order, size=self.batch_size - len(batch), replace=True)
                batch = np.concatenate([batch, fill])
            yield batch.astype(np.int64).tolist()

    def __len__(self) -> int:
        return int(math.ceil(self.size / self.batch_size))


def make_train_loader(
    x: np.ndarray,
    y: np.ndarray,
    next_y: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    num_workers: int,
    seed: int,
    cuda: bool,
) -> DataLoader:
    """Pinned, persistent worker data path requested for the Blackwell VM."""

    dataset = TensorSequenceDataset(x, y, next_y, lengths)
    sampler = FixedBatchSampler(len(dataset), batch_size=batch_size, seed=seed)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": max(0, int(num_workers)),
        "pin_memory": bool(cuda),
        "persistent_workers": bool(num_workers > 0),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(**kwargs)


# ---------------------------------------------------------------------------
# Prefix-causal experts
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    def __init__(self, width: int, expansion: int = 2) -> None:
        super().__init__()
        self.in_proj = nn.Linear(width, width * expansion * 2)
        self.out_proj = nn.Linear(width * expansion, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(value * F.silu(gate))


class FeatureStem(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TruncatedStickBreakingBetaExpert(nn.Module):
    """Five-component truncated DP-like Beta mixture.

    For a binary observation, the identifiable marginal is the mixture mean.  The
    concentration is therefore regularized toward a finite prior in the loss rather
    than incorrectly treated as a directly observed epistemic quantity.
    """

    def __init__(self, width: int, components: int = 5) -> None:
        super().__init__()
        self.components = components
        self.pre = nn.Sequential(nn.LayerNorm(width), SwiGLU(width))
        self.sticks = nn.Linear(width, components - 1)
        self.mean_head = nn.Linear(width, components)
        self.log_concentration_head = nn.Linear(width, components)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = h + self.pre(h)
        stick_values = torch.sigmoid(self.sticks(hidden).float()).clamp(1.0e-4, 1.0 - 1.0e-4)
        remaining = torch.ones_like(stick_values[..., :1])
        weights: list[torch.Tensor] = []
        for index in range(self.components - 1):
            current = remaining * stick_values[..., index : index + 1]
            weights.append(current)
            remaining = remaining * (1.0 - stick_values[..., index : index + 1])
        weights.append(remaining)
        mixture_weights = torch.cat(weights, dim=-1)
        mixture_weights = mixture_weights / mixture_weights.sum(dim=-1, keepdim=True).clamp_min(EPS)
        means = torch.sigmoid(self.mean_head(hidden).float()).clamp(1.0e-5, 1.0 - 1.0e-5)
        concentration = (2.0 + F.softplus(self.log_concentration_head(hidden).float())).clamp(max=1000.0)
        alpha = means * concentration
        beta = (1.0 - means) * concentration
        component_variance = (alpha * beta) / ((concentration.square()) * (concentration + 1.0)).clamp_min(EPS)
        mean = (mixture_weights * means).sum(dim=-1)
        total_variance = (mixture_weights * component_variance).sum(dim=-1)
        total_variance = total_variance + (mixture_weights * (means - mean.unsqueeze(-1)).square()).sum(dim=-1)
        q_logit = torch.logit(mean.clamp(1.0e-5, 1.0 - 1.0e-5))
        log_concentration = (mixture_weights * torch.log(concentration)).sum(dim=-1)
        zero_invalid = mask.to(q_logit.dtype)
        return (
            hidden * zero_invalid.unsqueeze(-1),
            q_logit * zero_invalid,
            total_variance * zero_invalid,
            log_concentration * zero_invalid,
        )


class CausalGRUExpert(nn.Module):
    """Deployable replacement for the historical future-looking BiGRU."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(width, width, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # A unidirectional recurrence makes the prefix-causality proof immediate by
        # induction over t. Right-padding does not influence any earlier output.
        output, _ = self.gru(h)
        output = self.dropout(self.norm(output)) * mask.unsqueeze(-1)
        q_logit = self.head(output).squeeze(-1)
        zero = torch.zeros_like(q_logit, dtype=torch.float32)
        return output, q_logit, zero, zero


class CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, channels]. Only left padding enforces causality.
        channels_first = x.transpose(1, 2)
        padded = F.pad(channels_first, (self.left_padding, 0))
        return self.conv(padded).transpose(1, 2)


class CausalResidualTCNBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm_one = nn.LayerNorm(width)
        self.conv_one = CausalConv1d(width, kernel_size=3, dilation=dilation)
        self.norm_two = nn.LayerNorm(width)
        self.conv_two = CausalConv1d(width, kernel_size=3, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        update = self.conv_one(self.norm_one(x))
        update = self.dropout(F.gelu(update))
        update = self.conv_two(self.norm_two(update))
        return (x + self.dropout(F.gelu(update))) * mask.unsqueeze(-1)


class CausalTCNExpert(nn.Module):
    def __init__(self, width: int, blocks: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [CausalResidualTCNBlock(width, dilation=2**index, dropout=dropout) for index in range(blocks)]
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = h * mask.unsqueeze(-1)
        for block in self.blocks:
            output = block(output, mask)
        output = self.norm(output) * mask.unsqueeze(-1)
        q_logit = self.head(output).squeeze(-1)
        zero = torch.zeros_like(q_logit, dtype=torch.float32)
        return output, q_logit, zero, zero


class SelectiveDiagonalSSMExpert(nn.Module):
    """Input-selective diagonal SSM with stable FP32 ZOH discretization.

    A=-exp(A_log), Abar=exp(delta*A), Bbar=expm1(delta*A)/A * B.  Delta,
    B and C are input-conditioned as in selective SSMs; recurrence state is only
    updated at valid timesteps.  The recurrence deliberately stays FP32 under BF16
    autocast because exp/expm1 and cancellation are numerically sensitive.
    """

    def __init__(self, width: int, state_size: int, dropout: float) -> None:
        super().__init__()
        self.width = width
        self.state_size = state_size
        self.input_proj = nn.Linear(width, width)
        self.delta_proj = nn.Linear(width, width)
        self.b_proj = nn.Linear(width, width * state_size)
        self.c_proj = nn.Linear(width, width * state_size)
        self.a_log = nn.Parameter(torch.empty(width, state_size))
        nn.init.uniform_(self.a_log, -4.0, -1.0)
        self.skip = nn.Parameter(torch.ones(width))
        self.out_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, width = h.shape
        u = self.input_proj(h)
        delta = F.softplus(self.delta_proj(u).float()).clamp(1.0e-3, 1.0)
        b_values = self.b_proj(u).float().view(batch, length, width, self.state_size)
        c_values = self.c_proj(u).float().view(batch, length, width, self.state_size)
        a = -torch.exp(self.a_log.float().clamp(-8.0, 3.0))
        state = torch.zeros(batch, width, self.state_size, dtype=torch.float32, device=h.device)
        outputs: list[torch.Tensor] = []
        for time_index in range(length):
            dt = delta[:, time_index].unsqueeze(-1)
            a_bar = torch.exp(dt * a.unsqueeze(0))
            # a is strictly negative, so division is stable; clamp is defensive.
            b_bar = torch.expm1(dt * a.unsqueeze(0)) / a.unsqueeze(0).clamp(max=-1.0e-5)
            candidate = a_bar * state + b_bar * b_values[:, time_index] * u[:, time_index].float().unsqueeze(-1)
            valid = mask[:, time_index].view(batch, 1, 1)
            state = torch.where(valid, candidate, state)
            readout = (state * c_values[:, time_index]).sum(dim=-1) + self.skip.float() * u[:, time_index].float()
            readout = torch.where(valid.squeeze(-1), readout, torch.zeros_like(readout))
            outputs.append(readout)
        output = torch.stack(outputs, dim=1).to(dtype=h.dtype)
        output = self.dropout(self.out_norm(output)) * mask.unsqueeze(-1)
        q_logit = self.head(output).squeeze(-1)
        zero = torch.zeros_like(q_logit, dtype=torch.float32)
        return output, q_logit, zero, zero


def apply_rope(q_or_k: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to [batch, heads, time, head_dim] Q or K tensors."""

    _, _, length, head_dim = q_or_k.shape
    if head_dim % 2:
        raise ValueError("RoPE requires an even attention head dimension")
    positions = torch.arange(length, device=q_or_k.device, dtype=torch.float32)
    inv_frequency = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=q_or_k.device, dtype=torch.float32) / head_dim))
    angles = positions[:, None] * inv_frequency[None, :]
    cosine = torch.cos(angles)[None, None, :, :]
    sine = torch.sin(angles)[None, None, :, :]
    pairs = q_or_k.float().reshape(*q_or_k.shape[:-1], head_dim // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack([even * cosine - odd * sine, even * sine + odd * cosine], dim=-1)
    return rotated.flatten(-2).to(dtype=q_or_k.dtype)


class CausalSDPABlock(nn.Module):
    """Actual RoPE Q/K + F.scaled_dot_product_attention causal block."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads or (width // heads) % 2:
            raise ValueError("d_model/head count must yield an even head dimension")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.norm_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.norm_ff = nn.LayerNorm(width)
        self.ff = SwiGLU(width)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        normalized = self.norm_attention(x)
        qkv = self.qkv(normalized).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = apply_rope(q.permute(0, 2, 1, 3))
        k = apply_rope(k.permute(0, 2, 1, 3))
        v = v.permute(0, 2, 1, 3)
        # Right padding plus is_causal means every valid query attends only valid
        # history.  Supplying a general mask would often disable Flash/SDPA kernels.
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.permute(0, 2, 1, 3).reshape(batch, length, self.width)
        x = x + F.dropout(self.out(attended), p=self.dropout, training=self.training)
        x = x + F.dropout(self.ff(self.norm_ff(x)), p=self.dropout, training=self.training)
        return x * mask.unsqueeze(-1)


class CausalRoPETransformerExpert(nn.Module):
    def __init__(self, width: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([CausalSDPABlock(width, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = h * mask.unsqueeze(-1)
        for layer in self.layers:
            output = layer(output, mask)
        output = self.norm(output) * mask.unsqueeze(-1)
        q_logit = self.head(output).squeeze(-1)
        zero = torch.zeros_like(q_logit, dtype=torch.float32)
        return output, q_logit, zero, zero


class CausalFourierOperatorExpert(nn.Module):
    """Causal Fourier neural operator using FFT linear convolution.

    Padding to 2T-1 turns the spectral product into linear convolution.  Taking the
    first T outputs yields sum_{tau<=t} K[t-tau] h[tau], so this is a genuine causal
    spectral operator rather than a future-looking FNO diagnostic.
    """

    def __init__(self, width: int, modes: int, dropout: float) -> None:
        super().__init__()
        self.width = width
        self.modes = modes
        scale = 1.0 / math.sqrt(width)
        self.weight_real = nn.Parameter(scale * torch.randn(width, width, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(width, width, modes))
        self.skip = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.ff = SwiGLU(width)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, length, _ = h.shape
        fft_length = 2 * length - 1
        spectrum = torch.fft.rfft(h.float().transpose(1, 2), n=fft_length, dim=-1)
        active_modes = min(self.modes, spectrum.shape[-1])
        # A raw truncated spectral multiplier is circular and would wrap future
        # positions into an early output.  Convert its Fourier parameterization to
        # a time kernel, explicitly retain only non-negative lags, then transform
        # that causal kernel back before the FFT convolution.
        parameter_spectrum = torch.zeros(
            self.width, self.width, spectrum.shape[-1], dtype=spectrum.dtype, device=h.device
        )
        parameter_spectrum[:, :, :active_modes] = torch.complex(
            self.weight_real[:, :, :active_modes], self.weight_imag[:, :, :active_modes]
        )
        unconstrained_kernel = torch.fft.irfft(parameter_spectrum, n=fft_length, dim=-1)
        causal_kernel = torch.zeros_like(unconstrained_kernel)
        causal_kernel[..., :length] = unconstrained_kernel[..., :length]
        transfer = torch.fft.rfft(causal_kernel, n=fft_length, dim=-1)
        transformed = torch.einsum("bdm,odm->bom", spectrum, transfer)
        convolution = torch.fft.irfft(transformed, n=fft_length, dim=-1)[..., :length].transpose(1, 2)
        output = convolution.to(dtype=h.dtype) + self.skip(h)
        output = output + self.dropout(self.ff(self.norm(output)))
        output = self.norm(output) * mask.unsqueeze(-1)
        q_logit = self.head(output).squeeze(-1)
        zero = torch.zeros_like(q_logit, dtype=torch.float32)
        return output, q_logit, zero, zero


class TransitionHazardHead(nn.Module):
    """Separate repair/corruption heads with units compatible with the policy."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), SwiGLU(width), nn.Linear(width, 2))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(h)
        return logits[..., 0], logits[..., 1]


class MineCritic(nn.Module):
    """Fold-local Donsker--Varadhan MINE critic used only as an opt-in auxiliary."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(2, width)
        self.net = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, representation: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_feature = self.label_embedding(labels.long().clamp(0, 1))
        return self.net(torch.cat([representation.float(), label_feature.float()], dim=-1)).squeeze(-1).clamp(-20.0, 20.0)


def mine_lower_bound(critic: MineCritic, representation: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    representation_flat = representation[mask]
    labels_flat = labels[mask]
    if representation_flat.shape[0] < 8 or labels_flat.unique().numel() < 2:
        return representation.sum() * 0.0
    joint = critic(representation_flat, labels_flat)
    permutation = torch.randperm(labels_flat.shape[0], device=labels_flat.device)
    marginal = critic(representation_flat, labels_flat[permutation])
    return joint.mean() - torch.logsumexp(marginal, dim=0) + math.log(float(marginal.numel()))


class FiveExpertStoppingMoE(nn.Module):
    """Beta, causal GRU, causal TCN, selective SSM and causal RoPE experts."""

    def __init__(self, width: int, config: ModelConfig) -> None:
        super().__init__()
        self.beta = TruncatedStickBreakingBetaExpert(width)
        self.gru = CausalGRUExpert(width, config.dropout)
        self.tcn = CausalTCNExpert(width, config.tcn_blocks, config.dropout)
        self.ssm = SelectiveDiagonalSSMExpert(width, config.ssm_state, config.dropout)
        self.transformer = CausalRoPETransformerExpert(
            width, config.attention_heads, config.transformer_layers, config.dropout
        )
        self.gate = nn.Sequential(nn.LayerNorm(width), SwiGLU(width), nn.Linear(width, 5))
        self.gate_temperature = max(float(config.gate_temperature), 0.05)

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expert_outputs = [
            self.beta(h, mask),
            self.gru(h, mask),
            self.tcn(h, mask),
            self.ssm(h, mask),
            self.transformer(h, mask),
        ]
        embeddings = torch.stack([result[0] for result in expert_outputs], dim=-2)
        logits = torch.stack([result[1].float() for result in expert_outputs], dim=-1)
        variances = torch.stack([result[2].float() for result in expert_outputs], dim=-1)
        log_concentrations = torch.stack([result[3].float() for result in expert_outputs], dim=-1)
        gate_logits = self.gate(h).float() / self.gate_temperature
        log_gates = F.log_softmax(gate_logits, dim=-1)
        gates = log_gates.exp() * mask.unsqueeze(-1).float()
        # Stable mixture probability/logit: avoid a BF16 probability-space sum.
        log_probability = torch.logsumexp(log_gates + F.logsigmoid(logits), dim=-1)
        log_one_minus_probability = torch.logsumexp(log_gates + F.logsigmoid(-logits), dim=-1)
        q_logit = (log_probability - log_one_minus_probability) * mask.float()
        fused = (gates.unsqueeze(-1).to(embeddings.dtype) * embeddings).sum(dim=-2)
        beta_variance = (gates * variances).sum(dim=-1)
        beta_log_concentration = (gates * log_concentrations).sum(dim=-1)
        return fused, q_logit, beta_variance, beta_log_concentration, gates


class CausalStoppingProbe(nn.Module):
    """One model wrapper that exposes common correctness and hazard outputs."""

    def __init__(self, input_dim: int, kind: str, config: ModelConfig) -> None:
        super().__init__()
        self.kind = kind
        self.stem = FeatureStem(input_dim, config.d_model, config.dropout)
        if kind == "beta":
            self.expert: nn.Module = TruncatedStickBreakingBetaExpert(config.d_model)
        elif kind == "gru":
            self.expert = CausalGRUExpert(config.d_model, config.dropout)
        elif kind == "tcn":
            self.expert = CausalTCNExpert(config.d_model, config.tcn_blocks, config.dropout)
        elif kind == "ssm":
            self.expert = SelectiveDiagonalSSMExpert(config.d_model, config.ssm_state, config.dropout)
        elif kind == "transformer":
            self.expert = CausalRoPETransformerExpert(
                config.d_model, config.attention_heads, config.transformer_layers, config.dropout
            )
        elif kind == "fno":
            self.expert = CausalFourierOperatorExpert(config.d_model, config.fno_modes, config.dropout)
        elif kind == "moe":
            self.expert = FiveExpertStoppingMoE(config.d_model, config)
        else:
            raise ValueError(f"Unknown model kind: {kind}")
        self.hazards = TransitionHazardHead(config.d_model)
        self.mine_critic = MineCritic(config.d_model)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = torch_valid_mask(lengths, x.shape[1])
        hidden = self.stem(x) * mask.unsqueeze(-1)
        if self.kind == "moe":
            representation, q_logit, beta_variance, beta_log_concentration, gates = self.expert(hidden, mask)  # type: ignore[misc]
        else:
            representation, q_logit, beta_variance, beta_log_concentration = self.expert(hidden, mask)  # type: ignore[misc]
            gates = torch.ones(*q_logit.shape, 1, device=q_logit.device, dtype=torch.float32)
        repair_logit, corruption_logit = self.hazards(representation)
        return (
            q_logit.float(),
            repair_logit.float(),
            corruption_logit.float(),
            beta_variance.float(),
            beta_log_concentration.float(),
            gates.float(),
            representation,
        )


def make_probe(input_dim: int, kind: str, config: ModelConfig, device: torch.device) -> CausalStoppingProbe:
    model = CausalStoppingProbe(input_dim, kind, config).to(device)
    return model


# ---------------------------------------------------------------------------
# Stable losses, calibration, and training
# ---------------------------------------------------------------------------


def binary_pos_weight(targets: np.ndarray) -> float:
    if targets.size == 0:
        return 1.0
    positives = float(np.sum(targets == 1))
    negatives = float(np.sum(targets == 0))
    if positives < 1.0 or negatives < 1.0:
        return 1.0
    return float(np.clip(negatives / positives, 0.25, 20.0))


@dataclass
class LossWeights:
    correctness: float
    repair: float
    corruption: float


def derive_loss_weights(y: np.ndarray, next_y: np.ndarray, lengths: np.ndarray) -> LossWeights:
    mask = np.arange(y.shape[1])[None, :] < lengths[:, None]
    has_next = np.arange(y.shape[1])[None, :] < (lengths[:, None] - 1)
    repair_mask = mask & has_next & (y == 0)
    corruption_mask = mask & has_next & (y == 1)
    return LossWeights(
        correctness=binary_pos_weight(y[mask]),
        repair=binary_pos_weight(next_y[repair_mask]),
        corruption=binary_pos_weight((1 - next_y[corruption_mask]).astype(np.int64)),
    )


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    if not bool(mask.any()):
        return logits.sum() * 0.0
    selected_logits = logits[mask].float().clamp(-30.0, 30.0)
    selected_targets = targets[mask].float()
    weight = torch.tensor(positive_weight, dtype=torch.float32, device=logits.device)
    return F.binary_cross_entropy_with_logits(selected_logits, selected_targets, pos_weight=weight)


def tournament_loss(
    model: CausalStoppingProbe,
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    y: torch.Tensor,
    next_y: torch.Tensor,
    lengths: torch.Tensor,
    config: ModelConfig,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Numerically stable correctness, calibrated-Beta, hazard and MoE losses."""

    q_logit, repair_logit, corruption_logit, beta_variance, beta_log_concentration, gates, representation = output
    mask = torch_valid_mask(lengths, y.shape[1])
    has_next = torch.arange(y.shape[1], device=y.device)[None, :] < (lengths[:, None] - 1)
    repair_mask = mask & has_next & (y == 0)
    corruption_mask = mask & has_next & (y == 1)
    correctness = masked_bce_with_logits(q_logit, y, mask, weights.correctness)
    probabilities = torch.sigmoid(q_logit.float())
    brier = ((probabilities - y.float()).square() * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
    repair = masked_bce_with_logits(repair_logit, next_y, repair_mask, weights.repair)
    corruption_targets = 1 - next_y
    corruption = masked_bce_with_logits(corruption_logit, corruption_targets, corruption_mask, weights.corruption)

    if gates.shape[-1] > 1:
        importance = (gates * mask.unsqueeze(-1).float()).sum(dim=(0, 1)) / mask.float().sum().clamp_min(1.0)
        # This is zero at uniform expert usage, unlike the legacy sum-of-squares term.
        load_balance = gates.shape[-1] * importance.square().sum() - 1.0
    else:
        load_balance = q_logit.sum() * 0.0
    concentration_prior = ((beta_log_concentration - math.log(8.0)).square() * mask.float()).sum()
    concentration_prior = concentration_prior / mask.float().sum().clamp_min(1.0)
    variance_penalty = (beta_variance * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
    mine_bound = mine_lower_bound(model.mine_critic, representation, y, mask)
    total = (
        correctness
        + config.brier_weight * brier
        + config.hazard_weight * (repair + corruption)
        + config.moe_balance_weight * load_balance
        + config.concentration_weight * concentration_prior
        + config.beta_variance_weight * variance_penalty
        - config.mine_weight * mine_bound
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "correctness": float(correctness.detach().cpu()),
        "repair": float(repair.detach().cpu()),
        "corruption": float(corruption.detach().cpu()),
        "brier": float(brier.detach().cpu()),
        "load_balance": float(load_balance.detach().cpu()),
        "concentration_prior": float(concentration_prior.detach().cpu()),
        "variance": float(variance_penalty.detach().cpu()),
        "mine_bound": float(mine_bound.detach().cpu()),
    }
    return total, metrics


def autocast_context(device: torch.device, precision: str) -> contextlib.AbstractContextManager[Any]:
    if device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast("cuda", dtype=dtype)


def pin_to_device(array: np.ndarray, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device.type == "cuda":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


class StaticCUDAGraphStep:
    """Optional explicit static forward/backward CUDA graph for BF16 fixed batches.

    It is intentionally profile-gated and mutually exclusive with torch.compile:
    compiled max-autotune already enables compatible CUDA-graph paths on supported
    PyTorch versions, whereas a manually captured graph needs static addresses and
    cannot safely use a dynamic FP16 GradScaler.
    """

    def __init__(
        self,
        model: CausalStoppingProbe,
        optimizer: torch.optim.Optimizer,
        first_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        config: ModelConfig,
        weights: LossWeights,
        precision: str,
    ) -> None:
        if precision != "bf16":
            raise ValueError("Explicit CUDA graph capture is supported only for BF16 static steps")
        x, y, next_y, lengths = first_batch
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.weights = weights
        self.static_x = torch.empty_like(x)
        self.static_y = torch.empty_like(y)
        self.static_next_y = torch.empty_like(next_y)
        self.static_lengths = torch.empty_like(lengths)
        self.static_x.copy_(x)
        self.static_y.copy_(y)
        self.static_next_y.copy_(next_y)
        self.static_lengths.copy_(lengths)
        self.graph = torch.cuda.CUDAGraph()
        warmup_stream = torch.cuda.Stream()
        with torch.cuda.stream(warmup_stream):
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    output = model(self.static_x, self.static_lengths)
                    loss, _ = tournament_loss(model, output, self.static_y, self.static_next_y, self.static_lengths, config, weights)
                loss.backward()
                optimizer.step()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(self.static_x, self.static_lengths)
                self.static_loss, _ = tournament_loss(
                    model, output, self.static_y, self.static_next_y, self.static_lengths, config, weights
                )
            self.static_loss.backward()
            optimizer.step()

    def step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        x, y, next_y, lengths = batch
        self.static_x.copy_(x)
        self.static_y.copy_(y)
        self.static_next_y.copy_(next_y)
        self.static_lengths.copy_(lengths)
        self.graph.replay()
        return float(self.static_loss.detach().cpu())


def maybe_compile_model(
    model: CausalStoppingProbe,
    kind: str,
    args: argparse.Namespace,
    is_trial: bool,
) -> nn.Module:
    """Use Inductor only for final refits; compiling every Optuna trial wastes hours."""

    if not args.compile or is_trial or args.cuda_graphs:
        return model
    if kind in {"gru", "fno"}:
        LOG.info("Leaving %s eager: recurrent/FFT graph capture is workload dependent.", kind)
        return model
    try:
        LOG.info("Compiling final %s model with torch.compile(mode=%s).", kind, args.compile_mode)
        return torch.compile(model, mode=args.compile_mode, dynamic=False)
    except Exception as error:  # Lazy compilation may still fall back during the first batch.
        LOG.warning("torch.compile setup failed for %s; using eager model: %s", kind, error)
        return model


def infer_probe(
    model: nn.Module,
    x: np.ndarray,
    lengths: np.ndarray,
    device: torch.device,
    precision: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw correctness, repair, and corruption logits in host arrays."""

    model.eval()
    total, max_len = x.shape[:2]
    q_logits = np.zeros((total, max_len), dtype=np.float32)
    repair_logits = np.zeros_like(q_logits)
    corruption_logits = np.zeros_like(q_logits)
    with torch.no_grad():
        for start in range(0, total, max(1, batch_size)):
            stop = min(total, start + max(1, batch_size))
            batch_x = pin_to_device(x[start:stop], device, torch.float32)
            batch_lengths = pin_to_device(lengths[start:stop], device, torch.long)
            with autocast_context(device, precision):
                output = model(batch_x, batch_lengths)
            q_logits[start:stop] = output[0].float().cpu().numpy()
            repair_logits[start:stop] = output[1].float().cpu().numpy()
            corruption_logits[start:stop] = output[2].float().cpu().numpy()
    return q_logits, repair_logits, corruption_logits


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    finite = np.isfinite(scores)
    if finite.sum() == 0 or np.unique(labels[finite]).size < 2:
        return float("nan")
    return float(roc_auc_score(labels[finite], scores[finite]))


def validation_auc(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    device: torch.device,
    precision: str,
    batch_size: int,
) -> float:
    q_logits, _, _ = infer_probe(model, x, lengths, device, precision, batch_size)
    mask = np.arange(y.shape[1])[None, :] < lengths[:, None]
    return safe_auc(y[mask], q_logits[mask])


def train_probe(
    kind: str,
    input_dim: int,
    config: ModelConfig,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_next_y: np.ndarray,
    train_lengths: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    *,
    epochs: int,
    num_workers: int,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    trial: Any | None = None,
    is_trial: bool = False,
) -> tuple[CausalStoppingProbe, float]:
    """Train one candidate; Optuna pruning observes task-group-held-out AUC."""

    base_model = make_probe(input_dim, kind, config, device)
    execution_model = maybe_compile_model(base_model, kind, args, is_trial)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=config.lr, weight_decay=config.weight_decay, fused=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    use_scaler = device.type == "cuda" and args.precision == "fp16"
    # BF16's exponent range makes scaling unnecessary; FP16 uses the requested scaler.
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    weights = derive_loss_weights(train_y, train_next_y, train_lengths)
    loader = make_train_loader(
        train_x,
        train_y,
        train_next_y,
        train_lengths,
        batch_size=min(args.batch_size, max(1, len(train_x)) if args.cap_batch_to_data else args.batch_size),
        num_workers=num_workers,
        seed=args.seed + (0 if trial is None else int(trial.number)),
        cuda=device.type == "cuda",
    )
    best_auc = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    graph_step: StaticCUDAGraphStep | None = None
    graph_disabled = not (args.cuda_graphs and device.type == "cuda" and not is_trial and args.precision == "bf16")
    try:
        for epoch in range(epochs):
            execution_model.train()
            losses: list[float] = []
            for cpu_x, cpu_y, cpu_next_y, cpu_lengths in loader:
                batch = (
                    cpu_x.to(device, non_blocking=True),
                    cpu_y.to(device, non_blocking=True),
                    cpu_next_y.to(device, non_blocking=True),
                    cpu_lengths.to(device, non_blocking=True),
                )
                if not graph_disabled:
                    try:
                        if graph_step is None:
                            graph_step = StaticCUDAGraphStep(
                                base_model, optimizer, batch, config, weights, args.precision
                            )
                            LOG.info("Captured static CUDA graph for %s training steps.", kind)
                        losses.append(graph_step.step(batch))
                        continue
                    except Exception as error:
                        LOG.warning("CUDA graph capture failed; reverting to eager steps: %s", error)
                        graph_disabled = True
                        graph_step = None
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, args.precision):
                    output = execution_model(batch[0], batch[3])
                    loss, _ = tournament_loss(
                        base_model, output, batch[1], batch[2], batch[3], config, weights
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite {kind} loss at epoch {epoch}")
                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                    optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scheduler.step()
            if validation is None:
                continue
            val_x, val_y, val_lengths = validation
            score = validation_auc(
                base_model, val_x, val_y, val_lengths, device, args.precision, args.eval_batch_size
            )
            if not math.isfinite(score):
                score = -1.0
            if trial is not None:
                trial.report(score, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"pruned {kind}: val_auc={score:.5f}")
            if score > best_auc + 1.0e-5:
                best_auc = score
                stale_epochs = 0
                best_state = {key: value.detach().cpu().clone() for key, value in base_model.state_dict().items()}
            else:
                stale_epochs += 1
            LOG.info(
                "%s epoch %d/%d: train_loss=%.5f val_auc=%.5f",
                kind,
                epoch + 1,
                epochs,
                float(np.mean(losses)) if losses else float("nan"),
                score,
            )
            if validation is not None and stale_epochs >= args.early_stopping_patience:
                break
    finally:
        # Persistent DataLoader workers are cleaned up once the local loader is released.
        del loader
    if best_state is not None:
        base_model.load_state_dict(best_state)
    if not math.isfinite(best_auc):
        best_auc = float("nan")
    return base_model, best_auc


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, logits: np.ndarray, targets: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        finite = np.isfinite(logits) & np.isfinite(targets)
        if finite.sum() < 8 or np.unique(targets[finite]).size < 2:
            self.temperature = 1.0
            return self
        raw_logits = torch.tensor(logits[finite], dtype=torch.float32)
        raw_targets = torch.tensor(targets[finite], dtype=torch.float32)
        log_temperature = torch.zeros((), dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.25, max_iter=50, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature).clamp(0.05, 10.0)
            loss = F.binary_cross_entropy_with_logits(raw_logits / temperature, raw_targets)
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
            self.temperature = float(torch.exp(log_temperature).detach().clamp(0.05, 10.0))
        except RuntimeError:
            self.temperature = 1.0
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=np.float64) / self.temperature, -30.0, 30.0)))


class ConstantLogitModel:
    def __init__(self, probability: float) -> None:
        probability = float(np.clip(probability, 1.0e-5, 1.0 - 1.0e-5))
        self.logit = math.log(probability / (1.0 - probability))

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.logit, dtype=np.float32)


def fit_logit_model(x: np.ndarray, targets: np.ndarray) -> LogisticRegression | ConstantLogitModel:
    targets = targets.astype(np.int64)
    if len(targets) == 0:
        return ConstantLogitModel(0.5)
    if np.unique(targets).size < 2:
        return ConstantLogitModel(float(targets.mean()))
    model = LogisticRegression(max_iter=1500, class_weight="balanced", solver="lbfgs")
    model.fit(x, targets)
    return model


def predict_logit_model(model: LogisticRegression | ConstantLogitModel, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.decision_function(x), dtype=np.float32)


def flatten_runs(
    x: np.ndarray,
    y: np.ndarray,
    next_y: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.arange(x.shape[1])[None, :] < lengths[:, None]
    has_next = np.arange(x.shape[1])[None, :] < (lengths[:, None] - 1)
    return x[mask], y[mask], next_y[mask], mask, has_next


def classical_baseline_predictions(
    scaled_x: np.ndarray,
    store: SequenceStore,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold-local logistic baseline with separately calibrated transition hazards."""

    def mask_for(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = store.valid_mask_np(indices)
        has_next = np.arange(store.max_len)[None, :] < (store.lengths[indices, None] - 1)
        return valid, has_next

    fit_mask, fit_has_next = mask_for(fit_indices)
    fit_x = scaled_x[fit_indices]
    fit_y = store.y[fit_indices]
    fit_next = store.next_y[fit_indices]
    correct_model = fit_logit_model(fit_x[fit_mask], fit_y[fit_mask])
    repair_condition = fit_mask & fit_has_next & (fit_y == 0)
    corruption_condition = fit_mask & fit_has_next & (fit_y == 1)
    repair_model = fit_logit_model(fit_x[repair_condition], fit_next[repair_condition])
    corruption_model = fit_logit_model(fit_x[corruption_condition], 1 - fit_next[corruption_condition])

    def predict(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        local_x = scaled_x[indices]
        shape = local_x.shape[:2]
        flat = local_x.reshape(-1, local_x.shape[-1])
        return (
            predict_logit_model(correct_model, flat).reshape(shape),
            predict_logit_model(repair_model, flat).reshape(shape),
            predict_logit_model(corruption_model, flat).reshape(shape),
        )

    calibration_logits = predict(calibration_indices)
    cal_mask, cal_has_next = mask_for(calibration_indices)
    cal_y = store.y[calibration_indices]
    cal_next = store.next_y[calibration_indices]
    q_temperature = TemperatureScaler().fit(calibration_logits[0][cal_mask], cal_y[cal_mask])
    repair_condition = cal_mask & cal_has_next & (cal_y == 0)
    corruption_condition = cal_mask & cal_has_next & (cal_y == 1)
    repair_temperature = TemperatureScaler().fit(calibration_logits[1][repair_condition], cal_next[repair_condition])
    corruption_temperature = TemperatureScaler().fit(
        calibration_logits[2][corruption_condition], (1 - cal_next[corruption_condition])
    )
    test_logits = predict(test_indices)
    return (
        q_temperature.transform(test_logits[0]).astype(np.float32),
        repair_temperature.transform(test_logits[1]).astype(np.float32),
        corruption_temperature.transform(test_logits[2]).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Task-grouped model selection and stopping evaluation
# ---------------------------------------------------------------------------


def assert_disjoint_task_groups(task_ids: np.ndarray, *partitions: np.ndarray) -> None:
    seen: set[str] = set()
    for partition in partitions:
        current = set(task_ids[partition].tolist())
        overlap = seen & current
        if overlap:
            raise AssertionError(f"Task leakage across partitions; examples: {sorted(overlap)[:5]}")
        seen |= current


def group_holdout(indices: np.ndarray, task_ids: np.ndarray, slot: int, requested_splits: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return train/holdout run indices with no task appearing in both sets."""

    if len(indices) < 2:
        raise ValueError("A task-group split needs at least two run trajectories")
    group_count = len(np.unique(task_ids[indices]))
    splits = min(requested_splits, group_count)
    if splits < 2:
        raise ValueError("A task-group split needs at least two distinct task IDs")
    local = np.arange(len(indices))
    splitter = GroupKFold(n_splits=splits)
    partitions = list(splitter.split(local, groups=task_ids[indices]))
    train_local, holdout_local = partitions[slot % len(partitions)]
    train, holdout = indices[train_local], indices[holdout_local]
    assert_disjoint_task_groups(task_ids, train, holdout)
    return train, holdout


def outer_and_inner_partitions(
    store: SequenceStore,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return tuning-train, tuning-validation, model-fit, calibration partitions."""

    model_fit, calibration = group_holdout(outer_train, store.task_ids, slot=fold, requested_splits=5)
    tuning_train, tuning_validation = group_holdout(model_fit, store.task_ids, slot=fold + 1, requested_splits=5)
    assert_disjoint_task_groups(store.task_ids, tuning_train, tuning_validation, calibration, outer_test)
    return tuning_train, tuning_validation, model_fit, calibration


def base_model_config(args: argparse.Namespace) -> ModelConfig:
    d_model = int(args.d_model)
    heads = int(args.attention_heads)
    if d_model % heads or (d_model // heads) % 2:
        raise ValueError("--d-model must divide evenly into an even --attention-heads dimension")
    return ModelConfig(
        d_model=d_model,
        dropout=float(args.dropout),
        tcn_blocks=int(args.tcn_blocks),
        transformer_layers=int(args.transformer_layers),
        attention_heads=heads,
        ssm_state=int(args.ssm_state),
        fno_modes=int(args.fno_modes),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        hazard_weight=float(args.hazard_weight),
        brier_weight=float(args.brier_weight),
        moe_balance_weight=float(args.moe_balance_weight),
        concentration_weight=float(args.concentration_weight),
        beta_variance_weight=float(args.beta_variance_weight),
        mine_weight=float(args.mine_weight),
        gate_temperature=float(args.gate_temperature),
    )


def suggest_model_config(trial: Any, args: argparse.Namespace) -> ModelConfig:
    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    # All candidates are multiples of eight with an even per-head dimension.
    return ModelConfig(
        d_model=int(d_model),
        dropout=trial.suggest_float("dropout", 0.0, 0.25),
        tcn_blocks=trial.suggest_int("tcn_blocks", 2, 4),
        transformer_layers=trial.suggest_int("transformer_layers", 1, 3),
        attention_heads=8,
        ssm_state=trial.suggest_categorical("ssm_state", [16, 24, 32]),
        fno_modes=trial.suggest_categorical("fno_modes", [4, 8, 12]),
        lr=trial.suggest_float("lr", 3.0e-4, 4.0e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1.0e-6, 3.0e-3, log=True),
        hazard_weight=trial.suggest_float("hazard_weight", 0.20, 0.90),
        brier_weight=trial.suggest_float("brier_weight", 0.02, 0.20),
        moe_balance_weight=trial.suggest_float("moe_balance_weight", 0.001, 0.05, log=True),
        concentration_weight=trial.suggest_float("concentration_weight", 1.0e-4, 0.02, log=True),
        # Retained as an explicit ablation; default remains zero because a binary
        # Beta-Bernoulli observation does not identify concentration.
        beta_variance_weight=0.0,
        mine_weight=0.0,
        gate_temperature=trial.suggest_float("gate_temperature", 0.65, 1.50),
    )


def optuna_storage_url(path: Path) -> str:
    # Posix form also produces a valid sqlite URL on Windows (sqlite:///C:/...).
    return "sqlite:///" + path.resolve().as_posix()


def tune_fold(
    fold: int,
    input_dim: int,
    tuning_train_x: np.ndarray,
    tuning_train_y: np.ndarray,
    tuning_train_next_y: np.ndarray,
    tuning_train_lengths: np.ndarray,
    tuning_validation_x: np.ndarray,
    tuning_validation_y: np.ndarray,
    tuning_validation_lengths: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    fingerprint: str,
    telemetry: TelemetryDB,
) -> ModelConfig:
    fallback = base_model_config(args)
    if args.trials_per_fold <= 0:
        LOG.info("Fold %d uses the explicit base model config; tuning disabled.", fold + 1)
        return fallback
    if optuna is None:
        raise RuntimeError("Optuna is required for --trials-per-fold > 0. Install optuna>=4.")
    storage_path = Path(args.output_dir) / f"ultimate_optuna_fold_{fold + 1}.sqlite3"
    study_name = f"ultimate_moe_{fingerprint[:16]}_fold_{fold + 1}"
    study = optuna.create_study(
        study_name=study_name,
        storage=optuna_storage_url(storage_path),
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=args.seed + fold, multivariate=True, group=True),
        pruner=HyperbandPruner(min_resource=1, max_resource=max(1, args.tune_epochs), reduction_factor=3),
    )
    remaining = max(0, int(args.trials_per_fold) - len(study.trials))
    LOG.info(
        "Fold %d Optuna study has %d persisted trials; scheduling %d/%d target trials.",
        fold + 1,
        len(study.trials),
        remaining,
        args.trials_per_fold,
    )

    def objective(trial: Any) -> float:
        config = suggest_model_config(trial, args)
        trial.set_user_attr("model_config", asdict(config))
        try:
            model, score = train_probe(
                "moe",
                input_dim,
                config,
                tuning_train_x,
                tuning_train_y,
                tuning_train_next_y,
                tuning_train_lengths,
                device,
                args,
                epochs=args.tune_epochs,
                num_workers=args.tune_num_workers,
                validation=(tuning_validation_x, tuning_validation_y, tuning_validation_lengths),
                trial=trial,
                is_trial=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            telemetry.trial(fold + 1, int(trial.number), "COMPLETE", float(score), asdict(config))
            return float(score)
        except optuna.TrialPruned:
            telemetry.trial(fold + 1, int(trial.number), "PRUNED", None, asdict(config))
            raise
        except RuntimeError as error:
            message = str(error).lower()
            if "out of memory" in message or "cuda error" in message:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                telemetry.trial(fold + 1, int(trial.number), "PRUNED_OOM", None, asdict(config))
                raise optuna.TrialPruned(str(error)) from error
            telemetry.trial(fold + 1, int(trial.number), "FAIL", None, asdict(config))
            raise

    if remaining:
        remaining_seconds = None
        deadline = getattr(args, "_deadline", None)
        if deadline is not None:
            remaining_seconds = max(0.0, float(deadline) - time.time())
        if remaining_seconds is None or remaining_seconds > 0.0:
            study.optimize(
                objective,
                n_trials=remaining,
                timeout=remaining_seconds,
                n_jobs=1,
                gc_after_trial=True,
                show_progress_bar=False,
            )
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None]
    if not completed:
        LOG.warning("Fold %d had no completed Optuna trial; using explicit base config.", fold + 1)
        return fallback
    best = max(completed, key=lambda trial: float(trial.value))
    config_data = best.user_attrs.get("model_config")
    if not isinstance(config_data, dict):
        return fallback
    selected = ModelConfig(**config_data)
    LOG.info("Fold %d selected Optuna trial %d (inner task-group AUC %.5f).", fold + 1, best.number, best.value)
    return selected


def calculate_ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    finite = np.isfinite(probabilities)
    probabilities, labels = probabilities[finite], labels[finite]
    if len(probabilities) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (probabilities >= lower) & ((probabilities < upper) if upper < 1.0 else (probabilities <= upper))
        if not in_bin.any():
            continue
        ece += float(in_bin.mean()) * abs(float(probabilities[in_bin].mean()) - float(labels[in_bin].mean()))
    return float(ece)


def evaluate_stopping_policy(
    store: SequenceStore,
    run_indices: np.ndarray,
    q: np.ndarray,
    repair: np.ndarray,
    corruption: np.ndarray,
    *,
    hysteresis: bool = False,
    lower: float = 0.10,
    upper: float = 0.90,
) -> pd.DataFrame:
    """Evaluate the CTMDP one-step HJB proxy without peeking beyond the stop step."""

    records: list[dict[str, Any]] = []
    for local_index, run_index in enumerate(run_indices):
        length = int(store.lengths[run_index])
        final_position = length - 1
        stop_position = final_position
        triggered_k2_positions: list[int] = []
        for position in range(length):
            step = int(store.steps[run_index, position])
            if step < T_MIN:
                continue
            q_t = float(q[local_index, position])
            repair_t = float(repair[local_index, position])
            corruption_t = float(corruption[local_index, position])
            if hysteresis and lower < q_t < upper:
                triggered_k2_positions.append(position)
                if int(store.k2_agreement[run_index, position]) == 0:
                    continue
            gain = (1.0 - q_t) * repair_t - q_t * corruption_t - STEP_COST
            if gain <= 0.0:
                stop_position = position
                break
        stop_step = int(store.steps[run_index, stop_position])
        final_step = int(store.steps[run_index, final_position])
        stopped_correct = int(store.y[run_index, stop_position])
        final_correct = int(store.y[run_index, final_position])
        stopped_tokens = float(store.thought_tokens[run_index, : stop_position + 1].sum())
        all_tokens = float(store.thought_tokens[run_index, :length].sum())
        if hysteresis:
            stopped_tokens += float(sum(store.k2_tokens[run_index, p] for p in triggered_k2_positions if p <= stop_position))
            all_tokens += float(store.k2_tokens[run_index, :length].sum())
        records.append(
            {
                "trajectory_id": str(store.trajectory_ids[run_index]),
                "stop_step": stop_step,
                "never_stop_step": final_step,
                "stop_correct": stopped_correct,
                "never_stop_correct": final_correct,
                "stop_utility": stopped_correct - STEP_COST * (stop_step - 1),
                "never_stop_utility": final_correct - STEP_COST * (final_step - 1),
                "stop_utility_token": stopped_correct - TOKEN_PRICE * stopped_tokens,
                "never_stop_utility_token": final_correct - TOKEN_PRICE * all_tokens,
            }
        )
    return pd.DataFrame(records)


def policy_summary(policy: pd.DataFrame) -> tuple[float, float, str]:
    if policy.empty:
        return float("nan"), float("nan"), "0/0/0"
    delta = policy["stop_utility"] - policy["never_stop_utility"]
    wins = int((delta > 1.0e-12).sum())
    ties = int((np.abs(delta) <= 1.0e-12).sum())
    losses = int((delta < -1.0e-12).sum())
    return float(policy["stop_utility"].mean()), float(policy["stop_utility_token"].mean()), f"{wins}/{ties}/{losses}"


# ---------------------------------------------------------------------------
# Blackwell preflight / VRAM audit and durable result artifacts
# ---------------------------------------------------------------------------


def nvidia_smi_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        row = next(csv.reader([completed.stdout.strip().splitlines()[0]]))
        keys = ["name", "memory_total_mib", "memory_used_mib", "utilization_gpu_percent", "temperature_c", "power_draw_w", "power_limit_w"]
        values: dict[str, Any] = {}
        for key, value in zip(keys, row):
            cleaned = value.strip()
            if key == "name":
                values[key] = cleaned
            else:
                try:
                    values[key] = float(cleaned)
                except ValueError:
                    values[key] = None
        return values
    except (OSError, subprocess.SubprocessError, IndexError, StopIteration) as error:
        return {"nvidia_smi_error": str(error)}


def cuda_preflight(args: argparse.Namespace) -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available():
        if args.require_blackwell:
            raise RuntimeError("CUDA is unavailable but --require-blackwell was requested")
        return torch.device("cpu"), {"cuda_available": False}
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    total_mib = int(properties.total_memory // (1024 * 1024))
    arch_list = torch.cuda.get_arch_list()
    report: dict[str, Any] = {
        "cuda_available": True,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": properties.name,
        "compute_capability": f"sm_{properties.major}{properties.minor}",
        "total_memory_mib": total_mib,
        "torch_arch_list": arch_list,
        "nvidia_smi": nvidia_smi_snapshot(),
    }
    LOG.info("CUDA preflight: %s", json.dumps(report, sort_keys=True))
    if args.require_blackwell:
        errors: list[str] = []
        if properties.major < 12:
            errors.append(f"need Blackwell-class sm_120+, found sm_{properties.major}{properties.minor}")
        if total_mib < int(args.min_vram_gib * 1024):
            errors.append(f"need >= {args.min_vram_gib:.1f} GiB visible VRAM, found {total_mib / 1024:.1f} GiB")
        if not any("sm_120" in arch for arch in arch_list):
            errors.append("current PyTorch build does not advertise sm_120 support")
        if errors:
            raise RuntimeError("Blackwell preflight failed: " + "; ".join(errors))
    return device, report


def _audit_batch(store: SequenceStore, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = np.arange(batch_size, dtype=np.int64) % store.n_runs
    return store.x[source], store.y[source], store.next_y[source], store.lengths[source]


def run_vram_audit(
    store: SequenceStore,
    config: ModelConfig,
    args: argparse.Namespace,
    device: torch.device,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    """Measure real model steps and BF16 Tensor Core matmul throughput.

    This deliberately never allocates meaningless filler tensors just to make a
    memory percentage look impressive.  With five-step trajectories, a scientifically
    valid 4k--8k batch may use far less than 98GB; that observation is recorded.
    """

    if device.type != "cuda":
        report = {"status": "skipped", "reason": "CUDA unavailable", "preflight": preflight}
        atomic_json_dump(Path(args.output_dir) / "ultimate_vram_audit.json", report)
        return report
    candidate_batches = sorted({min(args.batch_size, args.max_batch_size), args.max_batch_size, 1024, 2048, 4096, 8192})
    candidate_batches = [batch for batch in candidate_batches if batch > 0 and batch <= args.max_batch_size]
    measurements: list[dict[str, Any]] = []
    for batch_size in candidate_batches:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = make_probe(store.x.shape[-1], "moe", config, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, fused=True)
            x_np, y_np, next_y_np, lengths_np = _audit_batch(store, batch_size)
            x = pin_to_device(x_np, device, torch.float32)
            y = pin_to_device(y_np, device, torch.long)
            next_y = pin_to_device(next_y_np, device, torch.long)
            lengths = pin_to_device(lengths_np, device, torch.long)
            weights = derive_loss_weights(y_np, next_y_np, lengths_np)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    output = model(x, lengths)
                    loss, _ = tournament_loss(model, output, y, next_y, lengths, config, weights)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            start.record()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(x, lengths)
                loss, _ = tournament_loss(model, output, y, next_y, lengths, config, weights)
            loss.backward()
            optimizer.step()
            end.record()
            torch.cuda.synchronize()
            milliseconds = float(start.elapsed_time(end))
            peak_mib = int(torch.cuda.max_memory_allocated() // (1024 * 1024))
            measurements.append(
                {
                    "batch_size": batch_size,
                    "status": "ok",
                    "step_milliseconds": milliseconds,
                    "peak_allocated_mib": peak_mib,
                    "peak_fraction_of_visible": peak_mib / max(1, int(preflight.get("total_memory_mib", 1))),
                    "loss": float(loss.detach().cpu()),
                }
            )
            del model, optimizer, x, y, next_y, lengths
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            measurements.append({"batch_size": batch_size, "status": "oom"})
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                torch.cuda.empty_cache()
                measurements.append({"batch_size": batch_size, "status": "oom"})
            else:
                measurements.append({"batch_size": batch_size, "status": "error", "error": str(error)})
        gc.collect()

    throughput: dict[str, Any]
    try:
        dimension = int(args.audit_matrix_size)
        a = torch.randn(dimension, dimension, device=device, dtype=torch.bfloat16)
        b = torch.randn(dimension, dimension, device=device, dtype=torch.bfloat16)
        for _ in range(3):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iterations = 5
        start.record()
        for _ in range(iterations):
            torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        seconds = max(start.elapsed_time(end) / 1000.0, 1.0e-6)
        throughput = {
            "bf16_matmul_dimension": dimension,
            "iterations": iterations,
            "estimated_tensorcore_tflops": (2.0 * dimension**3 * iterations) / seconds / 1.0e12,
        }
        del a, b
    except RuntimeError as error:
        throughput = {"error": str(error)}
    torch.cuda.empty_cache()
    report = {
        "status": "ok",
        "preflight": preflight,
        "real_probe_batch_measurements": measurements,
        "bf16_tensor_core_benchmark": throughput,
        "nvidia_smi_after": nvidia_smi_snapshot(),
        "note": (
            "VRAM target is an observation, not a pass/fail gate. The canonical corpus has short sequences; "
            "dummy allocations would not improve a stopping model."
        ),
    }
    atomic_json_dump(Path(args.output_dir) / "ultimate_vram_audit.json", report)
    LOG.info("Wrote VRAM audit to %s", Path(args.output_dir) / "ultimate_vram_audit.json")
    return report


def write_research_graph(
    output_dir: Path,
    manifest: dict[str, Any],
    model_configs: dict[str, Any],
    result_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Portable graph artifact when a Memory Graph MCP endpoint is not available."""

    entities: list[dict[str, Any]] = [
        {
            "id": f"dataset:{manifest['dataset_fingerprint'][:16]}",
            "type": "TrajectoryCorpus",
            "observations": {
                "cells": manifest["selected_cell_count"],
                "trajectories": manifest["source_qualified_trajectories"],
                "task_ids": manifest["task_ids"],
                "source_qualified": True,
            },
        }
    ]
    links: list[dict[str, str]] = []
    for name, config in model_configs.items():
        identifier = f"model:{name}"
        entities.append({"id": identifier, "type": "StoppingProbe", "observations": config})
        links.append({"from": identifier, "relation": "trained_on", "to": f"dataset:{manifest['dataset_fingerprint'][:16]}"})
    for row in result_rows or []:
        identifier = f"result:{row['configuration']}"
        entities.append({"id": identifier, "type": "OOFResult", "observations": row})
        links.append({"from": identifier, "relation": "evaluates", "to": f"dataset:{manifest['dataset_fingerprint'][:16]}"})
    atomic_json_dump(output_dir / "ultimate_research_graph.json", {"entities": entities, "relations": links})


def load_checkpoint(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        LOG.warning("Ignoring unreadable checkpoint %s: %s", path, error)
        return None
    if checkpoint.get("fingerprint") != fingerprint:
        LOG.warning("Checkpoint fingerprint differs from this corpus/configuration; not resuming it.")
        return None
    return checkpoint


def save_fold_checkpoint(
    path: Path,
    fingerprint: str,
    completed_folds: set[int],
    oof_q: dict[str, np.ndarray],
    oof_repair: dict[str, np.ndarray],
    oof_corruption: dict[str, np.ndarray],
    selected_configs: dict[str, Any],
) -> None:
    atomic_torch_save(
        path,
        {
            "fingerprint": fingerprint,
            "completed_folds": sorted(completed_folds),
            "oof_q": oof_q,
            "oof_repair": oof_repair,
            "oof_corruption": oof_corruption,
            "selected_configs": selected_configs,
            "saved_at": time.time(),
        },
    )


def write_result_artifacts(
    output_dir: Path,
    store: SequenceStore,
    oof_q: dict[str, np.ndarray],
    oof_repair: dict[str, np.ndarray],
    oof_corruption: dict[str, np.ndarray],
    manifest: dict[str, Any],
    selected_configs: dict[str, Any],
) -> list[dict[str, Any]]:
    mask = store.valid_mask_np()
    rows: list[dict[str, Any]] = []
    all_indices = np.arange(store.n_runs)
    display = {
        "linear": "Baseline Linear (task-grouped)",
        "beta": "Truncated Beta Mixture",
        "gru": "Causal GRU (prefix-safe)",
        "tcn": "Causal Residual TCN",
        "ssm": "Selective SSM (ZOH)",
        "transformer": "Causal RoPE SDPA Transformer",
        "fno": "Causal Fourier Neural Operator",
        "moe": "Five-Expert Causal MoE",
        "moe_hysteresis": "Five-Expert Causal MoE + Hysteresis",
    }
    for key, probability in oof_q.items():
        if not np.isfinite(probability[mask]).all():
            LOG.warning("Skipping %s result table row: OOF predictions are incomplete.", key)
            continue
        policy = evaluate_stopping_policy(
            store, all_indices, probability, oof_repair[key], oof_corruption[key], hysteresis=False
        )
        step_utility, token_utility, wtl = policy_summary(policy)
        rows.append(
            {
                "configuration": display.get(key, key),
                "OOF ROC-AUC": safe_auc(store.y[mask], probability[mask]),
                "ECE": calculate_ece(probability[mask], store.y[mask]),
                "Step Utility": step_utility,
                "Token Utility": token_utility,
                "Win/Tie/Loss": wtl,
            }
        )
    if "moe" in oof_q and np.isfinite(oof_q["moe"][mask]).all():
        policy = evaluate_stopping_policy(
            store,
            all_indices,
            oof_q["moe"],
            oof_repair["moe"],
            oof_corruption["moe"],
            hysteresis=True,
        )
        step_utility, token_utility, wtl = policy_summary(policy)
        rows.append(
            {
                "configuration": display["moe_hysteresis"],
                "OOF ROC-AUC": safe_auc(store.y[mask], oof_q["moe"][mask]),
                "ECE": calculate_ece(oof_q["moe"][mask], store.y[mask]),
                "Step Utility": step_utility,
                "Token Utility": token_utility,
                "Win/Tie/Loss": wtl,
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("OOF ROC-AUC", ascending=False, kind="stable")
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "ultimate_tournament_results.csv", index=False)
    header = [
        "ULTIMATE CAUSAL MULTI-DAY TOURNAMENT VERDICT",
        "=" * 112,
        (
            f"Corpus: {manifest['selected_cell_count']} cells, {manifest['source_qualified_trajectories']} "
            f"source-qualified trajectories, {manifest['task_ids']} task groups."
        ),
        (
            "Integrity note: historical all-cell raw-run-id scores (including 0.8656) are not directly comparable; "
            "this table uses source-qualified trajectories and task-grouped outer folds."
        ),
        "",
    ]
    if table.empty:
        body = "No complete OOF result is available yet; resume from the fold checkpoint.\n"
    else:
        body = table.to_string(index=False, formatters={
            "OOF ROC-AUC": "{:.4f}".format,
            "ECE": "{:.4f}".format,
            "Step Utility": "{:+.4f}".format,
            "Token Utility": "{:+.4f}".format,
        })
    footer = [
        "",
        "Stopping policy: mu=(1-q)*P(repair|currently wrong)-q*P(corruption|currently correct)-0.05; stop when mu<=0.",
        "Token utility: correctness - 0.0002 * generated reasoning tokens.",
        "=" * 112,
    ]
    (output_dir / "ultimate_tournament_results.log").write_text("\n".join(header + [body] + footer) + "\n", encoding="utf-8")
    np.savez_compressed(
        output_dir / "ultimate_oof_predictions.npz",
        **{f"q__{key}": value for key, value in oof_q.items()},
        **{f"repair__{key}": value for key, value in oof_repair.items()},
        **{f"corruption__{key}": value for key, value in oof_corruption.items()},
    )
    write_research_graph(output_dir, manifest, selected_configs, rows)
    return rows


# ---------------------------------------------------------------------------
# Command line protocol and outer tournament
# ---------------------------------------------------------------------------


def parse_model_kinds(value: str) -> list[str]:
    allowed = {"beta", "gru", "tcn", "ssm", "transformer", "fno", "moe"}
    result = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("--models must name at least one neural candidate")
    unknown = sorted(set(result) - allowed)
    if unknown:
        raise ValueError(f"Unknown model kinds {unknown}; choose from {sorted(allowed)}")
    return list(dict.fromkeys(result))


def apply_smoke_test_overrides(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    args.n_splits = 2
    args.trials_per_fold = 0
    args.tune_epochs = 1
    args.epochs = 1
    args.batch_size = min(args.batch_size, 32)
    args.max_batch_size = min(args.max_batch_size, 64)
    args.eval_batch_size = min(args.eval_batch_size, 64)
    args.num_workers = 0
    args.tune_num_workers = 0
    args.models = "moe"
    args.max_cells = 2 if args.max_cells is None else min(args.max_cells, 2)
    args.require_blackwell = False
    args.compile = False
    args.cuda_graphs = False
    args.max_hours = 0.0


def causality_self_test() -> None:
    """Regression test: changing x[t+1:] must not affect outputs through t."""

    torch.manual_seed(7)
    config = ModelConfig(
        d_model=32,
        dropout=0.0,
        tcn_blocks=2,
        transformer_layers=1,
        attention_heads=4,
        ssm_state=8,
        fno_modes=4,
    )
    x = torch.randn(3, 5, 12)
    lengths = torch.tensor([5, 5, 5], dtype=torch.long)
    perturbed = x.clone()
    perturbed[:, 3:, :] += 10.0 * torch.randn_like(perturbed[:, 3:, :])
    for kind in ["beta", "gru", "tcn", "ssm", "transformer", "fno", "moe"]:
        model = CausalStoppingProbe(12, kind, config).eval()
        with torch.no_grad():
            original = model(x, lengths)
            changed = model(perturbed, lengths)
        for output_index, label in [(0, "q"), (1, "repair"), (2, "corruption")]:
            difference = (original[output_index][:, :3] - changed[output_index][:, :3]).abs().max().item()
            if difference > 2.0e-5:
                raise AssertionError(f"Causality failure in {kind}/{label}: prefix difference={difference}")
    print("Causality self-test passed for beta, GRU, TCN, SSM, RoPE-SDPA, causal FNO, and MoE.")


def assign_oof(
    destination: np.ndarray,
    test_indices: np.ndarray,
    values: np.ndarray,
    store: SequenceStore,
    name: str,
) -> None:
    mask = store.valid_mask_np(test_indices)
    existing = destination[test_indices]
    if np.isfinite(existing[mask]).any():
        raise AssertionError(f"Duplicate OOF assignment for {name}; outer folds are not disjoint")
    destination[test_indices] = values


def neural_predictions_with_calibration(
    model: CausalStoppingProbe,
    scaled_x: np.ndarray,
    store: SequenceStore,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    calibration_logits = infer_probe(
        model,
        scaled_x[calibration_indices],
        store.lengths[calibration_indices],
        device,
        args.precision,
        args.eval_batch_size,
    )
    cal_mask = store.valid_mask_np(calibration_indices)
    cal_has_next = np.arange(store.max_len)[None, :] < (store.lengths[calibration_indices, None] - 1)
    cal_y = store.y[calibration_indices]
    cal_next = store.next_y[calibration_indices]
    q_temperature = TemperatureScaler().fit(calibration_logits[0][cal_mask], cal_y[cal_mask])
    repair_condition = cal_mask & cal_has_next & (cal_y == 0)
    corruption_condition = cal_mask & cal_has_next & (cal_y == 1)
    repair_temperature = TemperatureScaler().fit(calibration_logits[1][repair_condition], cal_next[repair_condition])
    corruption_temperature = TemperatureScaler().fit(
        calibration_logits[2][corruption_condition], 1 - cal_next[corruption_condition]
    )
    test_logits = infer_probe(
        model,
        scaled_x[test_indices],
        store.lengths[test_indices],
        device,
        args.precision,
        args.eval_batch_size,
    )
    return (
        q_temperature.transform(test_logits[0]).astype(np.float32),
        repair_temperature.transform(test_logits[1]).astype(np.float32),
        corruption_temperature.transform(test_logits[2]).astype(np.float32),
    )


def run_tournament(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    configure_logging(output_dir)
    device, preflight = cuda_preflight(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    frame, manifest = load_trace_frame(args)
    features = build_feature_frame(frame, args.persistent_homology, args.topology_window)
    store = build_sequence_store(frame, features)
    manifest.update(
        {
            "feature_count": len(store.feature_names),
            "feature_fingerprint": stable_json_hash(store.feature_names),
            "preflight": preflight,
            "protocol": {
                "outer_grouping": "task_id",
                "trajectory_key": "source_cell::run_id",
                "outer_folds": args.n_splits,
                "t_min": T_MIN,
                "step_cost": STEP_COST,
                "precision": args.precision,
                "persistent_homology": args.persistent_homology,
            },
        }
    )
    protocol_args = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_") and key not in {"output_dir", "audit_only", "vram_audit", "dry_run", "resume"}
    }
    fingerprint = stable_json_hash(
        {
            "dataset": manifest["dataset_fingerprint"],
            "features": manifest["feature_fingerprint"],
            "protocol": protocol_args,
        }
    )
    manifest["run_fingerprint"] = fingerprint
    atomic_json_dump(output_dir / "ultimate_tournament_manifest.json", manifest)
    atomic_json_dump(
        output_dir / "ultimate_tournament_status.json",
        {"status": "running", "complete": False, "fingerprint": fingerprint, "started_at": time.time()},
    )
    telemetry = TelemetryDB(output_dir / "ultimate_tournament_telemetry.sqlite3")
    telemetry.put_meta("manifest", manifest)
    telemetry.put_meta("run_fingerprint", fingerprint)
    write_research_graph(output_dir, manifest, {})
    try:
        initial_config = base_model_config(args)
        if args.vram_audit:
            audit = run_vram_audit(store, initial_config, args, device, preflight)
            telemetry.event("vram_audit", audit)
        if args.audit_only:
            LOG.info("Audit-only mode completed.")
            return 0
        if args.dry_run:
            LOG.info("Dry run completed: source-qualified ingestion and feature construction passed.")
            return 0

        model_kinds = parse_model_kinds(args.models)
        all_keys = ["linear", *model_kinds]
        oof_q = {key: np.full((store.n_runs, store.max_len), np.nan, dtype=np.float32) for key in all_keys}
        oof_repair = {key: np.full_like(value, np.nan) for key, value in oof_q.items()}
        oof_corruption = {key: np.full_like(value, np.nan) for key, value in oof_q.items()}
        selected_configs: dict[str, Any] = {}
        checkpoint_path = output_dir / "ultimate_tournament_checkpoint.pth"
        completed_folds: set[int] = set()
        if args.resume:
            checkpoint = load_checkpoint(checkpoint_path, fingerprint)
            if checkpoint is not None:
                restored_keys = set(checkpoint.get("oof_q", {}).keys())
                if restored_keys == set(all_keys):
                    oof_q = checkpoint["oof_q"]
                    oof_repair = checkpoint["oof_repair"]
                    oof_corruption = checkpoint["oof_corruption"]
                    completed_folds = set(int(item) for item in checkpoint.get("completed_folds", []))
                    selected_configs = dict(checkpoint.get("selected_configs", {}))
                    LOG.info("Resumed %d completed folds from %s.", len(completed_folds), checkpoint_path)
                else:
                    LOG.warning("Checkpoint candidate set differs from requested models; beginning a clean run.")

        distinct_tasks = len(np.unique(store.task_ids))
        if distinct_tasks < args.n_splits:
            raise ValueError(f"Need {args.n_splits} distinct task IDs for GroupKFold; corpus has {distinct_tasks}")
        outer_splitter = GroupKFold(n_splits=args.n_splits)
        outer_splits = list(outer_splitter.split(np.arange(store.n_runs), groups=store.task_ids))
        args._deadline = None if args.max_hours <= 0 else time.time() + args.max_hours * 3600.0
        LOG.info(
            "Starting %d-fold task-grouped tournament: models=%s trials/fold=%d epochs=%d batch=%d.",
            args.n_splits,
            ",".join(all_keys),
            args.trials_per_fold,
            args.epochs,
            args.batch_size,
        )

        for fold, (outer_train, outer_test) in enumerate(outer_splits):
            if fold in completed_folds:
                LOG.info("Fold %d/%d already checkpointed; skipping.", fold + 1, args.n_splits)
                continue
            if args._deadline is not None and time.time() >= args._deadline:
                LOG.warning("Configured time budget reached before fold %d; checkpoint is resumable.", fold + 1)
                save_fold_checkpoint(
                    checkpoint_path, fingerprint, completed_folds, oof_q, oof_repair, oof_corruption, selected_configs
                )
                atomic_json_dump(
                    output_dir / "ultimate_tournament_status.json",
                    {
                        "status": "time_budget_reached",
                        "complete": False,
                        "fingerprint": fingerprint,
                        "completed_folds": sorted(completed_folds),
                        "updated_at": time.time(),
                    },
                )
                return 0
            assert_disjoint_task_groups(store.task_ids, outer_train, outer_test)
            tuning_train, tuning_validation, model_fit, calibration = outer_and_inner_partitions(
                store, outer_train, outer_test, fold
            )
            LOG.info(
                "Fold %d/%d: outer train=%d test=%d; tune train=%d val=%d; fit=%d calibration=%d runs.",
                fold + 1,
                args.n_splits,
                len(outer_train),
                len(outer_test),
                len(tuning_train),
                len(tuning_validation),
                len(model_fit),
                len(calibration),
            )
            telemetry.fold(fold + 1, "running", {"outer_train": len(outer_train), "outer_test": len(outer_test)})

            tuning_scaler = FoldRobustScaler().fit(store.x, store.lengths, tuning_train)
            tuning_x = tuning_scaler.transform(store.x, store.lengths)
            config = tune_fold(
                fold,
                store.x.shape[-1],
                tuning_x[tuning_train],
                store.y[tuning_train],
                store.next_y[tuning_train],
                store.lengths[tuning_train],
                tuning_x[tuning_validation],
                store.y[tuning_validation],
                store.lengths[tuning_validation],
                device,
                args,
                fingerprint,
                telemetry,
            )
            selected_configs[f"fold_{fold + 1}"] = asdict(config)
            # Refit scaler from all non-calibration outer-training task groups.
            scaler = FoldRobustScaler().fit(store.x, store.lengths, model_fit)
            scaled_x = scaler.transform(store.x, store.lengths)

            linear_q, linear_repair, linear_corruption = classical_baseline_predictions(
                scaled_x, store, model_fit, calibration, outer_test
            )
            assign_oof(oof_q["linear"], outer_test, linear_q, store, "linear/q")
            assign_oof(oof_repair["linear"], outer_test, linear_repair, store, "linear/repair")
            assign_oof(oof_corruption["linear"], outer_test, linear_corruption, store, "linear/corruption")

            for kind in model_kinds:
                if args._deadline is not None and time.time() >= args._deadline:
                    # Do not serialize a partially assigned fold: resuming it from
                    # scratch is safer than treating a mixture of old/new OOF rows
                    # as complete. Optuna's per-fold SQLite study is already durable.
                    LOG.warning("Time budget reached during fold %d; completed folds remain resumable.", fold + 1)
                    telemetry.event("time_budget_reached", {"fold": fold + 1, "completed_folds": sorted(completed_folds)})
                    atomic_json_dump(
                        output_dir / "ultimate_tournament_status.json",
                        {
                            "status": "time_budget_reached",
                            "complete": False,
                            "fingerprint": fingerprint,
                            "completed_folds": sorted(completed_folds),
                            "updated_at": time.time(),
                        },
                    )
                    return 0
                LOG.info("Fold %d: fitting final %s candidate.", fold + 1, kind)
                model, _ = train_probe(
                    kind,
                    store.x.shape[-1],
                    config,
                    scaled_x[model_fit],
                    store.y[model_fit],
                    store.next_y[model_fit],
                    store.lengths[model_fit],
                    device,
                    args,
                    epochs=args.epochs,
                    num_workers=args.num_workers,
                    validation=None,
                    is_trial=False,
                )
                q, repair, corruption = neural_predictions_with_calibration(
                    model, scaled_x, store, calibration, outer_test, device, args
                )
                assign_oof(oof_q[kind], outer_test, q, store, f"{kind}/q")
                assign_oof(oof_repair[kind], outer_test, repair, store, f"{kind}/repair")
                assign_oof(oof_corruption[kind], outer_test, corruption, store, f"{kind}/corruption")
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            completed_folds.add(fold)
            fold_summary = {
                "fold": fold + 1,
                "status": "complete",
                "outer_train_runs": int(len(outer_train)),
                "outer_test_runs": int(len(outer_test)),
                "outer_train_task_count": int(len(np.unique(store.task_ids[outer_train]))),
                "outer_test_task_count": int(len(np.unique(store.task_ids[outer_test]))),
                "selected_config": asdict(config),
                "completed_at": time.time(),
            }
            atomic_json_dump(output_dir / f"ultimate_fold_{fold + 1:02d}_summary.json", fold_summary)
            save_fold_checkpoint(
                checkpoint_path, fingerprint, completed_folds, oof_q, oof_repair, oof_corruption, selected_configs
            )
            telemetry.fold(fold + 1, "complete", fold_summary)
            telemetry.event("fold_complete", fold_summary)
            write_research_graph(output_dir, manifest, selected_configs)

        result_rows = write_result_artifacts(
            output_dir, store, oof_q, oof_repair, oof_corruption, manifest, selected_configs
        )
        telemetry.event("tournament_complete", {"rows": result_rows, "completed_folds": sorted(completed_folds)})
        atomic_json_dump(
            output_dir / "ultimate_tournament_status.json",
            {
                "status": "complete",
                "complete": True,
                "fingerprint": fingerprint,
                "completed_folds": sorted(completed_folds),
                "completed_at": time.time(),
            },
        )
        LOG.info("Tournament complete. Results: %s", output_dir / "ultimate_tournament_results.log")
        return 0
    finally:
        telemetry.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", default="research/outputs/experiments_v2", help="Directory containing trace cell folders")
    parser.add_argument("--output-dir", default=None, help="Artifact directory (defaults to --input-dir)")
    parser.add_argument("--include-all-cells", action=argparse.BooleanOptionalAction, default=False, help="Include noncanonical sweep/pilot cells")
    parser.add_argument("--max-cells", type=int, default=None, help="Limit selected cells for a diagnostic run")
    parser.add_argument("--persistent-homology", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--topology-window", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5, help="Outer GroupKFold count; protocol default is five")
    parser.add_argument("--models", default="beta,gru,tcn,ssm,transformer,fno,moe")
    parser.add_argument("--trials-per-fold", type=int, default=500)
    parser.add_argument("--tune-epochs", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--cap-batch-to-data", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-workers", type=int, default=0, help="Default to 0 in containers to prevent /dev/shm IPC shared memory exhaustion")
    parser.add_argument("--tune-num-workers", type=int, default=0, help="Avoid worker churn across hundreds of trials")
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-mode", choices=["max-autotune", "reduce-overhead", "max-autotune-no-cudagraphs"], default="max-autotune")
    parser.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--tcn-blocks", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--ssm-state", type=int, default=24)
    parser.add_argument("--fno-modes", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hazard-weight", type=float, default=0.50)
    parser.add_argument("--brier-weight", type=float, default=0.10)
    parser.add_argument("--moe-balance-weight", type=float, default=0.010)
    parser.add_argument("--concentration-weight", type=float, default=0.002)
    parser.add_argument("--beta-variance-weight", type=float, default=0.0)
    parser.add_argument("--mine-weight", type=float, default=0.0, help="Opt-in MINE auxiliary; zero keeps it diagnostic")
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--max-hours", type=float, default=71.5, help="Graceful checkpoint/resume budget; <=0 disables")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-blackwell", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-vram-gib", type=float, default=90.0)
    parser.add_argument("--vram-audit", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--audit-matrix-size", type=int, default=8192)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    try:
        import torch.multiprocessing as mp
        mp.set_sharing_strategy('file_system')
    except Exception:
        pass
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        causality_self_test()
        return 0
    if args.output_dir is None:
        args.output_dir = args.input_dir
    apply_smoke_test_overrides(args)
    if args.audit_only:
        args.vram_audit = True
    if args.n_splits < 2:
        parser.error("--n-splits must be at least two")
    if args.batch_size < 1 or args.max_batch_size < args.batch_size:
        parser.error("Require 1 <= --batch-size <= --max-batch-size")
    if args.topology_window < 2:
        parser.error("--topology-window must be >= 2")
    try:
        return run_tournament(args)
    except KeyboardInterrupt:
        LOG.warning("Interrupted; completed fold checkpoints remain resumable.")
        return 130
    except Exception as error:
        if LOG.handlers:
            LOG.exception("Ultimate tournament failed: %s", error)
        else:
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
