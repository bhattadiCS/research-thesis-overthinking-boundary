from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SEED = 7
OBS_LOW = -0.15
OBS_HIGH = 0.15
ENTROPY_THRESHOLD = 0.58
CUSUM_THRESHOLD = 0.05
CUSUM_MARGIN = 0.0015
DEFAULT_PLOT_SAMPLE_SIZE = 2048


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    q0: float
    horizon: int
    lambda_cost: float
    alpha_base: float
    alpha_decay: float
    beta_floor: float
    beta_rise: float
    beta_slope: float
    beta_midpoint: float
    prm_bias: float
    prm_bias_slope: float
    prm_bias_midpoint: float
    regime_midpoint: float
    regime_slope: float
    state_noise: float
    prm_noise: float
    probe_noise: float
    observable_noise: float


@dataclass
class ScenarioAggregate:
    cfg: ScenarioConfig
    count: int
    sum_optimal_stop: float
    sum_true_boundary: float
    sum_safe_stop: float
    sum_eb_stop: float
    sum_naive_stop: float
    sum_cusum_stop: float
    sum_entropy_stop: float
    sum_prm_peak_stop: float
    sum_safe_gap: float
    sum_eb_gap: float
    sum_naive_gap: float
    sum_cusum_gap: float
    sum_entropy_gap: float
    sum_prm_peak_gap: float
    count_safe_false_early: float
    count_eb_false_early: float
    count_naive_false_early: float
    count_cusum_false_early: float
    count_entropy_false_early: float
    count_prm_postboundary: float
    sum_true_mu: np.ndarray
    sum_proxy_mu: np.ndarray
    sum_entropy: np.ndarray
    sum_hidden_shift: np.ndarray
    representative_score: float
    representative_run: dict[str, np.ndarray | int | float]
    gap_samples: np.ndarray


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-values))


def alpha_curve(times: np.ndarray, cfg: ScenarioConfig) -> np.ndarray:
    return cfg.alpha_base * np.exp(-cfg.alpha_decay * times)


def beta_curve(times: np.ndarray, cfg: ScenarioConfig) -> np.ndarray:
    return cfg.beta_floor + cfg.beta_rise * sigmoid(cfg.beta_slope * (times - cfg.beta_midpoint))


def bias_curve(times: np.ndarray, cfg: ScenarioConfig) -> np.ndarray:
    return cfg.prm_bias * sigmoid(cfg.prm_bias_slope * (times - cfg.prm_bias_midpoint))


def regime_curve(times: np.ndarray, cfg: ScenarioConfig) -> np.ndarray:
    return sigmoid(cfg.regime_slope * (times - cfg.regime_midpoint))


def bounded_confidence(range_width: float, sample_count: int, delta: float) -> float:
    return range_width * math.sqrt(math.log(1.0 / delta) / (2.0 * sample_count))


def empirical_bernstein_radius(sample_variance: float, range_width: float, sample_count: int, delta: float) -> float:
    log_term = math.log(3.0 / delta)
    return math.sqrt(2.0 * max(sample_variance, 0.0) * log_term / sample_count) + 3.0 * range_width * log_term / sample_count


def time_uniform_delta(delta: float, step: int) -> float:
    value = 6.0 * delta / (math.pi**2 * (step + 1) ** 2)
    return min(max(value, 1e-12), 0.999999)


def simulate_single_run(
    rng: np.random.Generator,
    cfg: ScenarioConfig,
    probe_count: int,
    delta: float,
) -> dict[str, np.ndarray | int | float]:
    times = np.arange(cfg.horizon, dtype=float)
    alpha = alpha_curve(times, cfg)
    beta = beta_curve(times, cfg)
    bias = bias_curve(times, cfg)
    regime = regime_curve(times, cfg)

    q = np.zeros(cfg.horizon)
    value = np.zeros(cfg.horizon)
    prm = np.zeros(cfg.horizon)
    true_mu = np.zeros(cfg.horizon)
    proxy_mu = np.zeros(cfg.horizon)
    estimated_mu = np.zeros(cfg.horizon)
    safe_upper_bound = np.zeros(cfg.horizon)
    eb_upper_bound = np.zeros(cfg.horizon)
    naive_upper_bound = np.zeros(cfg.horizon)
    entropy = np.zeros(cfg.horizon)
    hidden_shift = np.zeros(cfg.horizon)

    q[0] = cfg.q0
    prm[0] = cfg.q0
    value[0] = q[0]

    range_width = OBS_HIGH - OBS_LOW
    safe_stop = cfg.horizon - 1
    eb_stop = cfg.horizon - 1
    naive_stop = cfg.horizon - 1
    cusum_stop = cfg.horizon - 1
    entropy_stop = cfg.horizon - 1
    safe_found = False
    eb_found = False
    naive_found = False
    cusum_found = False
    entropy_found = False
    cusum_score = 0.0
    entropy_ema = 0.0

    for t in range(cfg.horizon):
        value[t] = q[t] - cfg.lambda_cost * t
        true_mu[t] = (1.0 - q[t]) * alpha[t] - q[t] * beta[t] - cfg.lambda_cost
        proxy_mu[t] = true_mu[t] + bias[t]

        probe_observations = np.clip(
            true_mu[t] + cfg.probe_noise * rng.standard_normal(probe_count),
            OBS_LOW,
            OBS_HIGH,
        )
        estimated_mu[t] = float(np.mean(probe_observations))
        probe_variance = float(np.var(probe_observations, ddof=1)) if probe_count > 1 else 0.0

        safe_conf = bounded_confidence(range_width=range_width, sample_count=probe_count, delta=time_uniform_delta(delta, t))
        eb_conf = empirical_bernstein_radius(
            sample_variance=probe_variance,
            range_width=range_width,
            sample_count=probe_count,
            delta=time_uniform_delta(delta, t),
        )
        naive_conf = bounded_confidence(range_width=range_width, sample_count=probe_count, delta=delta)

        safe_upper_bound[t] = estimated_mu[t] + safe_conf
        eb_upper_bound[t] = estimated_mu[t] + eb_conf
        naive_upper_bound[t] = estimated_mu[t] + naive_conf

        entropy[t] = float(
            np.clip(
                0.18
                + 0.52 * regime[t]
                + 0.18 * sigmoid(12.0 * (beta[t] - alpha[t]))
                + cfg.observable_noise * rng.standard_normal(),
                0.05,
                1.25,
            )
        )
        hidden_shift[t] = float(
            np.clip(
                0.04
                + 0.46 * regime[t]
                + 0.12 * abs(true_mu[t])
                + cfg.observable_noise * rng.standard_normal(),
                0.0,
                1.2,
            )
        )

        if not safe_found and safe_upper_bound[t] <= 0.0:
            safe_stop = t
            safe_found = True
        if not eb_found and eb_upper_bound[t] <= 0.0:
            eb_stop = t
            eb_found = True
        if not naive_found and naive_upper_bound[t] <= 0.0:
            naive_stop = t
            naive_found = True

        entropy_ema = entropy[t] if t == 0 else 0.7 * entropy_ema + 0.3 * entropy[t]
        if not entropy_found and t >= 3 and entropy_ema >= ENTROPY_THRESHOLD:
            entropy_stop = t
            entropy_found = True

        cusum_increment = max(0.0, -(estimated_mu[t] + CUSUM_MARGIN))
        cusum_score = max(0.0, 0.6 * cusum_score + cusum_increment)
        if not cusum_found and t >= 3 and cusum_score >= CUSUM_THRESHOLD:
            cusum_stop = t
            cusum_found = True

        if t == cfg.horizon - 1:
            continue

        state_noise = cfg.state_noise * math.sqrt(max(q[t] * (1.0 - q[t]), 1e-6)) * rng.standard_normal()
        delta_q = (1.0 - q[t]) * alpha[t] - q[t] * beta[t] + state_noise
        q[t + 1] = float(np.clip(q[t] + delta_q, 1e-4, 1.0 - 1e-4))

        prm_drift = proxy_mu[t] + cfg.prm_noise * rng.standard_normal()
        prm[t + 1] = prm[t] + prm_drift

    true_boundary_candidates = np.where(true_mu <= 0.0)[0]
    true_boundary = int(true_boundary_candidates[0]) if len(true_boundary_candidates) else cfg.horizon - 1
    optimal_stop = int(np.argmax(value))
    prm_peak_stop = int(np.argmax(prm))

    return {
        "times": times,
        "q": q,
        "value": value,
        "prm": prm,
        "true_mu": true_mu,
        "proxy_mu": proxy_mu,
        "estimated_mu": estimated_mu,
        "safe_upper_bound": safe_upper_bound,
        "eb_upper_bound": eb_upper_bound,
        "naive_upper_bound": naive_upper_bound,
        "entropy": entropy,
        "hidden_shift": hidden_shift,
        "optimal_stop": optimal_stop,
        "true_boundary": true_boundary,
        "safe_stop": safe_stop,
        "eb_stop": eb_stop,
        "naive_stop": naive_stop,
        "cusum_stop": cusum_stop,
        "entropy_stop": entropy_stop,
        "prm_peak_stop": prm_peak_stop,
        "oracle_value": float(value[optimal_stop]),
        "safe_value": float(value[safe_stop]),
        "eb_value": float(value[eb_stop]),
        "naive_value": float(value[naive_stop]),
        "cusum_value": float(value[cusum_stop]),
        "entropy_value": float(value[entropy_stop]),
        "prm_peak_value": float(value[prm_peak_stop]),
    }


def aggregate_scenario_runs(
    cfg: ScenarioConfig,
    run_count: int,
    probe_count: int,
    delta: float,
    seed: int,
    plot_sample_size: int,
) -> ScenarioAggregate:
    rng = np.random.default_rng(seed)
    sum_true_mu = np.zeros(cfg.horizon, dtype=float)
    sum_proxy_mu = np.zeros(cfg.horizon, dtype=float)
    sum_entropy = np.zeros(cfg.horizon, dtype=float)
    sum_hidden_shift = np.zeros(cfg.horizon, dtype=float)
    gap_samples: list[list[float]] = []
    representative_run: dict[str, np.ndarray | int | float] | None = None
    representative_score = math.inf

    sum_optimal_stop = 0.0
    sum_true_boundary = 0.0
    sum_safe_stop = 0.0
    sum_eb_stop = 0.0
    sum_naive_stop = 0.0
    sum_cusum_stop = 0.0
    sum_entropy_stop = 0.0
    sum_prm_peak_stop = 0.0
    sum_safe_gap = 0.0
    sum_eb_gap = 0.0
    sum_naive_gap = 0.0
    sum_cusum_gap = 0.0
    sum_entropy_gap = 0.0
    sum_prm_peak_gap = 0.0
    count_safe_false_early = 0.0
    count_eb_false_early = 0.0
    count_naive_false_early = 0.0
    count_cusum_false_early = 0.0
    count_entropy_false_early = 0.0
    count_prm_postboundary = 0.0

    for run_index in range(run_count):
        run = simulate_single_run(rng=rng, cfg=cfg, probe_count=probe_count, delta=delta)

        optimal_stop = float(run["optimal_stop"])
        true_boundary = float(run["true_boundary"])
        safe_stop = float(run["safe_stop"])
        eb_stop = float(run["eb_stop"])
        naive_stop = float(run["naive_stop"])
        cusum_stop = float(run["cusum_stop"])
        entropy_stop = float(run["entropy_stop"])
        prm_peak_stop = float(run["prm_peak_stop"])
        oracle_value = float(run["oracle_value"])
        safe_gap = oracle_value - float(run["safe_value"])
        eb_gap = oracle_value - float(run["eb_value"])
        naive_gap = oracle_value - float(run["naive_value"])
        cusum_gap = oracle_value - float(run["cusum_value"])
        entropy_gap = oracle_value - float(run["entropy_value"])
        prm_peak_gap = oracle_value - float(run["prm_peak_value"])

        sum_optimal_stop += optimal_stop
        sum_true_boundary += true_boundary
        sum_safe_stop += safe_stop
        sum_eb_stop += eb_stop
        sum_naive_stop += naive_stop
        sum_cusum_stop += cusum_stop
        sum_entropy_stop += entropy_stop
        sum_prm_peak_stop += prm_peak_stop
        sum_safe_gap += safe_gap
        sum_eb_gap += eb_gap
        sum_naive_gap += naive_gap
        sum_cusum_gap += cusum_gap
        sum_entropy_gap += entropy_gap
        sum_prm_peak_gap += prm_peak_gap
        count_safe_false_early += float(safe_stop < true_boundary)
        count_eb_false_early += float(eb_stop < true_boundary)
        count_naive_false_early += float(naive_stop < true_boundary)
        count_cusum_false_early += float(cusum_stop < true_boundary)
        count_entropy_false_early += float(entropy_stop < true_boundary)
        count_prm_postboundary += float(prm_peak_stop > true_boundary)
        sum_true_mu += np.asarray(run["true_mu"], dtype=float)
        sum_proxy_mu += np.asarray(run["proxy_mu"], dtype=float)
        sum_entropy += np.asarray(run["entropy"], dtype=float)
        sum_hidden_shift += np.asarray(run["hidden_shift"], dtype=float)

        current_score = abs(optimal_stop - eb_stop)
        if representative_run is None or current_score < representative_score:
            representative_score = current_score
            representative_run = run

        gap_row = [safe_gap, eb_gap, cusum_gap, entropy_gap, prm_peak_gap]
        if len(gap_samples) < plot_sample_size:
            gap_samples.append(gap_row)
        else:
            replace_index = int(rng.integers(0, run_index + 1))
            if replace_index < plot_sample_size:
                gap_samples[replace_index] = gap_row

    if representative_run is None:
        raise ValueError(f"No runs were simulated for scenario {cfg.name}")

    return ScenarioAggregate(
        cfg=cfg,
        count=run_count,
        sum_optimal_stop=sum_optimal_stop,
        sum_true_boundary=sum_true_boundary,
        sum_safe_stop=sum_safe_stop,
        sum_eb_stop=sum_eb_stop,
        sum_naive_stop=sum_naive_stop,
        sum_cusum_stop=sum_cusum_stop,
        sum_entropy_stop=sum_entropy_stop,
        sum_prm_peak_stop=sum_prm_peak_stop,
        sum_safe_gap=sum_safe_gap,
        sum_eb_gap=sum_eb_gap,
        sum_naive_gap=sum_naive_gap,
        sum_cusum_gap=sum_cusum_gap,
        sum_entropy_gap=sum_entropy_gap,
        sum_prm_peak_gap=sum_prm_peak_gap,
        count_safe_false_early=count_safe_false_early,
        count_eb_false_early=count_eb_false_early,
        count_naive_false_early=count_naive_false_early,
        count_cusum_false_early=count_cusum_false_early,
        count_entropy_false_early=count_entropy_false_early,
        count_prm_postboundary=count_prm_postboundary,
        sum_true_mu=sum_true_mu,
        sum_proxy_mu=sum_proxy_mu,
        sum_entropy=sum_entropy,
        sum_hidden_shift=sum_hidden_shift,
        representative_score=representative_score,
        representative_run=representative_run,
        gap_samples=np.asarray(gap_samples, dtype=float),
    )


def summarize_aggregate(aggregate: ScenarioAggregate) -> dict[str, float | str]:
    count = float(aggregate.count)
    return {
        "scenario": aggregate.cfg.name,
        "mean_optimal_stop": aggregate.sum_optimal_stop / count,
        "mean_true_boundary": aggregate.sum_true_boundary / count,
        "mean_safe_stop": aggregate.sum_safe_stop / count,
        "mean_eb_stop": aggregate.sum_eb_stop / count,
        "mean_naive_stop": aggregate.sum_naive_stop / count,
        "mean_cusum_stop": aggregate.sum_cusum_stop / count,
        "mean_entropy_stop": aggregate.sum_entropy_stop / count,
        "mean_prm_peak_stop": aggregate.sum_prm_peak_stop / count,
        "mean_safe_optimality_gap": aggregate.sum_safe_gap / count,
        "mean_eb_optimality_gap": aggregate.sum_eb_gap / count,
        "mean_naive_optimality_gap": aggregate.sum_naive_gap / count,
        "mean_cusum_optimality_gap": aggregate.sum_cusum_gap / count,
        "mean_entropy_optimality_gap": aggregate.sum_entropy_gap / count,
        "mean_prm_peak_optimality_gap": aggregate.sum_prm_peak_gap / count,
        "safe_false_early_rate": aggregate.count_safe_false_early / count,
        "eb_false_early_rate": aggregate.count_eb_false_early / count,
        "naive_false_early_rate": aggregate.count_naive_false_early / count,
        "cusum_false_early_rate": aggregate.count_cusum_false_early / count,
        "entropy_false_early_rate": aggregate.count_entropy_false_early / count,
        "prm_postboundary_rate": aggregate.count_prm_postboundary / count,
    }


def write_summary_csv(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.csv"
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_representatives(
    representatives: list[tuple[ScenarioConfig, dict[str, np.ndarray | int | float]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(representatives), 1, figsize=(11.5, 13.5), sharex=True)
    if len(representatives) == 1:
        axes = [axes]

    for ax, (cfg, run) in zip(axes, representatives):
        times = run["times"]
        value = run["value"]
        prm = run["prm"]
        true_mu = run["true_mu"]
        proxy_mu = run["proxy_mu"]

        ax.plot(times, value, label="True stop-value V_t", color="#0f766e", linewidth=2.0)
        ax.plot(times, prm, label="Raw PRM proxy P_t", color="#b91c1c", linewidth=1.8, alpha=0.9)
        ax.axvline(run["optimal_stop"], color="#15803d", linestyle="-.", linewidth=1.8, label="Oracle stop")
        ax.axvline(run["true_boundary"], color="#7c3aed", linestyle=":", linewidth=2.0, label="True boundary")
        ax.axvline(run["safe_stop"], color="#ea580c", linestyle="--", linewidth=1.5, label="Anytime Hoeffding")
        ax.axvline(run["eb_stop"], color="#0284c7", linestyle="--", linewidth=1.8, label="Anytime EmpBern")
        ax.axvline(run["cusum_stop"], color="#0f766e", linestyle=":", linewidth=1.5, alpha=0.8, label="CUSUM")
        ax.axvline(run["entropy_stop"], color="#a16207", linestyle="-", linewidth=1.3, label="Entropy stop")
        ax.axvline(run["prm_peak_stop"], color="#6b7280", linestyle="-", linewidth=1.3, label="PRM argmax")

        ax2 = ax.twinx()
        ax2.plot(times, true_mu, label="True drift mu_t", color="#1d4ed8", linestyle="--", linewidth=1.5)
        ax2.plot(times, proxy_mu, label="Proxy drift mu_t + kappa_t", color="#d97706", linestyle=":", linewidth=1.5)

        ax.set_title(cfg.name.replace("_", " ").title())
        ax.set_ylabel("Value / PRM")
        ax2.set_ylabel("Drift")
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=7.5)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Reasoning step")
    fig.tight_layout()
    fig.savefig(output_dir / "representative_trajectories.png", dpi=200)
    plt.close(fig)


def plot_gap_distributions(aggregates: list[ScenarioAggregate], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(aggregates), figsize=(15, 4.5), sharey=True)
    if len(aggregates) == 1:
        axes = [axes]

    for ax, aggregate in zip(axes, aggregates):
        cfg = aggregate.cfg
        gap_samples = aggregate.gap_samples
        ax.boxplot(
            [
                gap_samples[:, 0],
                gap_samples[:, 1],
                gap_samples[:, 2],
                gap_samples[:, 3],
                gap_samples[:, 4],
            ],
            tick_labels=["Hoeffding", "EmpBern", "CUSUM", "Entropy", "PRM"],
            showfliers=False,
        )
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_title(cfg.name.replace("_", " ").title())
        ax.set_ylabel("Oracle value - baseline value")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "monte_carlo_gaps.png", dpi=200)
    plt.close(fig)


def plot_average_drifts(aggregates: list[ScenarioAggregate], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(aggregates), figsize=(15, 4.5), sharey=True)
    if len(aggregates) == 1:
        axes = [axes]

    for ax, aggregate in zip(axes, aggregates):
        cfg = aggregate.cfg
        mean_true_mu = aggregate.sum_true_mu / aggregate.count
        mean_proxy_mu = aggregate.sum_proxy_mu / aggregate.count
        mean_boundary = aggregate.sum_true_boundary / aggregate.count
        ax.plot(mean_true_mu, linewidth=2.0, color="#1d4ed8", label="True drift mu_t")
        ax.plot(mean_proxy_mu, linewidth=2.0, color="#d97706", linestyle="--", label="Proxy drift mu_t + kappa_t")
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
        ax.axvline(mean_boundary, color="#7c3aed", linestyle=":", linewidth=1.5, label="Mean true boundary")
        ax.set_title(cfg.name.replace("_", " ").title())
        ax.set_xlabel("Reasoning step")
        ax.set_ylabel("Drift")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "average_drifts.png", dpi=200)
    plt.close(fig)


def plot_observable_signals(aggregates: list[ScenarioAggregate], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(aggregates), figsize=(15, 4.5), sharey=True)
    if len(aggregates) == 1:
        axes = [axes]

    for ax, aggregate in zip(axes, aggregates):
        cfg = aggregate.cfg
        mean_entropy = aggregate.sum_entropy / aggregate.count
        mean_hidden_shift = aggregate.sum_hidden_shift / aggregate.count
        ax.plot(mean_entropy, linewidth=2.0, color="#a16207", label="Entropy proxy")
        ax.plot(mean_hidden_shift, linewidth=2.0, color="#0f766e", linestyle="--", label="Hidden-state shift proxy")
        ax.axhline(ENTROPY_THRESHOLD, color="#7c2d12", linestyle=":", linewidth=1.2, label="Entropy threshold")
        ax.set_title(cfg.name.replace("_", " ").title())
        ax.set_xlabel("Reasoning step")
        ax.set_ylabel("Observable magnitude")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "observable_signals.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate the overthinking boundary under synthetic regimes.")
    parser.add_argument("--n-trials", type=int, default=400, help="Monte Carlo trials per scenario.")
    parser.add_argument("--probe-count", type=int, default=2048, help="Probe samples per reasoning step.")
    parser.add_argument("--delta", type=float, default=0.05, help="Confidence level for time-uniform bounds.")
    parser.add_argument("--seed", type=int, default=SEED, help="Base RNG seed.")
    parser.add_argument("--parallel", action="store_true", help="Run scenarios in parallel processes.")
    parser.add_argument(
        "--plot-sample-size",
        type=int,
        default=DEFAULT_PLOT_SAMPLE_SIZE,
        help="Maximum sampled gaps retained per scenario for boxplots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary tables and plots.",
    )
    return parser.parse_args()


def build_scenarios() -> list[ScenarioConfig]:
    return [
        ScenarioConfig(
            name="helpful_reasoning",
            q0=0.28,
            horizon=60,
            lambda_cost=0.003,
            alpha_base=0.18,
            alpha_decay=0.045,
            beta_floor=0.010,
            beta_rise=0.045,
            beta_slope=0.20,
            beta_midpoint=42,
            prm_bias=0.000,
            prm_bias_slope=0.25,
            prm_bias_midpoint=40,
            regime_midpoint=38,
            regime_slope=0.22,
            state_noise=0.018,
            prm_noise=0.010,
            probe_noise=0.060,
            observable_noise=0.035,
        ),
        ScenarioConfig(
            name="overthinking",
            q0=0.30,
            horizon=60,
            lambda_cost=0.003,
            alpha_base=0.17,
            alpha_decay=0.060,
            beta_floor=0.012,
            beta_rise=0.115,
            beta_slope=0.28,
            beta_midpoint=24,
            prm_bias=0.000,
            prm_bias_slope=0.25,
            prm_bias_midpoint=28,
            regime_midpoint=24,
            regime_slope=0.28,
            state_noise=0.020,
            prm_noise=0.012,
            probe_noise=0.070,
            observable_noise=0.040,
        ),
        ScenarioConfig(
            name="reward_hacking",
            q0=0.30,
            horizon=60,
            lambda_cost=0.003,
            alpha_base=0.16,
            alpha_decay=0.065,
            beta_floor=0.015,
            beta_rise=0.120,
            beta_slope=0.30,
            beta_midpoint=22,
            prm_bias=0.020,
            prm_bias_slope=0.35,
            prm_bias_midpoint=20,
            regime_midpoint=20,
            regime_slope=0.32,
            state_noise=0.020,
            prm_noise=0.012,
            probe_noise=0.075,
            observable_noise=0.045,
        ),
    ]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [
        *build_scenarios(),
    ]

    aggregates: list[ScenarioAggregate] = []
    if args.parallel:
        with ProcessPoolExecutor(max_workers=min(len(scenarios), len(scenarios))) as executor:
            futures = [
                executor.submit(
                    aggregate_scenario_runs,
                    cfg,
                    args.n_trials,
                    args.probe_count,
                    args.delta,
                    args.seed + scenario_index,
                    args.plot_sample_size,
                )
                for scenario_index, cfg in enumerate(scenarios)
            ]
            for future in futures:
                aggregates.append(future.result())
    else:
        for scenario_index, cfg in enumerate(scenarios):
            aggregates.append(
                aggregate_scenario_runs(
                    cfg=cfg,
                    run_count=args.n_trials,
                    probe_count=args.probe_count,
                    delta=args.delta,
                    seed=args.seed + scenario_index,
                    plot_sample_size=args.plot_sample_size,
                )
            )

    summary_rows = [summarize_aggregate(aggregate) for aggregate in aggregates]
    representatives = [(aggregate.cfg, aggregate.representative_run) for aggregate in aggregates]

    write_summary_csv(summary_rows, output_dir=output_dir)
    plot_representatives(representatives, output_dir=output_dir)
    plot_gap_distributions(aggregates, output_dir=output_dir)
    plot_average_drifts(aggregates, output_dir=output_dir)
    plot_observable_signals(aggregates, output_dir=output_dir)

    for row in summary_rows:
        print(
            f"{row['scenario']}: "
            f"oracle={row['mean_optimal_stop']:.2f}, "
            f"boundary={row['mean_true_boundary']:.2f}, "
            f"hoeffding={row['mean_safe_stop']:.2f} (gap={row['mean_safe_optimality_gap']:.4f}), "
            f"empbern={row['mean_eb_stop']:.2f} (gap={row['mean_eb_optimality_gap']:.4f}), "
            f"cusum={row['mean_cusum_stop']:.2f} (gap={row['mean_cusum_optimality_gap']:.4f}), "
            f"entropy={row['mean_entropy_stop']:.2f} (gap={row['mean_entropy_optimality_gap']:.4f}), "
            f"prm={row['mean_prm_peak_stop']:.2f} (gap={row['mean_prm_peak_optimality_gap']:.4f})"
        )

    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()