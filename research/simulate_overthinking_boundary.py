from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SEED = 7
OBS_LOW = -0.15
OBS_HIGH = 0.15
ENTROPY_THRESHOLD = 0.58
CUSUM_THRESHOLD = 0.05
CUSUM_MARGIN = 0.0015


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


def summarize_runs(name: str, runs: list[dict[str, np.ndarray | int | float]]) -> dict[str, float | str]:
    optimal_stops = np.array([run["optimal_stop"] for run in runs], dtype=float)
    true_boundaries = np.array([run["true_boundary"] for run in runs], dtype=float)
    safe_stops = np.array([run["safe_stop"] for run in runs], dtype=float)
    eb_stops = np.array([run["eb_stop"] for run in runs], dtype=float)
    naive_stops = np.array([run["naive_stop"] for run in runs], dtype=float)
    cusum_stops = np.array([run["cusum_stop"] for run in runs], dtype=float)
    entropy_stops = np.array([run["entropy_stop"] for run in runs], dtype=float)
    prm_peak_stops = np.array([run["prm_peak_stop"] for run in runs], dtype=float)

    oracle_values = np.array([run["oracle_value"] for run in runs], dtype=float)
    safe_values = np.array([run["safe_value"] for run in runs], dtype=float)
    eb_values = np.array([run["eb_value"] for run in runs], dtype=float)
    naive_values = np.array([run["naive_value"] for run in runs], dtype=float)
    cusum_values = np.array([run["cusum_value"] for run in runs], dtype=float)
    entropy_values = np.array([run["entropy_value"] for run in runs], dtype=float)
    prm_peak_values = np.array([run["prm_peak_value"] for run in runs], dtype=float)

    safe_gaps = oracle_values - safe_values
    eb_gaps = oracle_values - eb_values
    naive_gaps = oracle_values - naive_values
    cusum_gaps = oracle_values - cusum_values
    entropy_gaps = oracle_values - entropy_values
    prm_peak_gaps = oracle_values - prm_peak_values

    return {
        "scenario": name,
        "mean_optimal_stop": float(np.mean(optimal_stops)),
        "mean_true_boundary": float(np.mean(true_boundaries)),
        "mean_safe_stop": float(np.mean(safe_stops)),
        "mean_eb_stop": float(np.mean(eb_stops)),
        "mean_naive_stop": float(np.mean(naive_stops)),
        "mean_cusum_stop": float(np.mean(cusum_stops)),
        "mean_entropy_stop": float(np.mean(entropy_stops)),
        "mean_prm_peak_stop": float(np.mean(prm_peak_stops)),
        "mean_safe_optimality_gap": float(np.mean(safe_gaps)),
        "mean_eb_optimality_gap": float(np.mean(eb_gaps)),
        "mean_naive_optimality_gap": float(np.mean(naive_gaps)),
        "mean_cusum_optimality_gap": float(np.mean(cusum_gaps)),
        "mean_entropy_optimality_gap": float(np.mean(entropy_gaps)),
        "mean_prm_peak_optimality_gap": float(np.mean(prm_peak_gaps)),
        "safe_false_early_rate": float(np.mean(safe_stops < true_boundaries)),
        "eb_false_early_rate": float(np.mean(eb_stops < true_boundaries)),
        "naive_false_early_rate": float(np.mean(naive_stops < true_boundaries)),
        "cusum_false_early_rate": float(np.mean(cusum_stops < true_boundaries)),
        "entropy_false_early_rate": float(np.mean(entropy_stops < true_boundaries)),
        "prm_postboundary_rate": float(np.mean(prm_peak_stops > true_boundaries)),
    }


def representative_run(runs: list[dict[str, np.ndarray | int | float]]) -> dict[str, np.ndarray | int | float]:
    gaps = np.array([abs(run["optimal_stop"] - run["eb_stop"]) for run in runs], dtype=float)
    return runs[int(np.argmin(gaps))]


def write_summary_csv(rows: list[dict[str, float | str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "summary.csv"
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_representatives(representatives: list[tuple[ScenarioConfig, dict[str, np.ndarray | int | float]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
    fig.savefig(OUTPUT_DIR / "representative_trajectories.png", dpi=200)
    plt.close(fig)


def plot_gap_distributions(runs_by_scenario: list[tuple[ScenarioConfig, list[dict[str, np.ndarray | int | float]]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(runs_by_scenario), figsize=(15, 4.5), sharey=True)
    if len(runs_by_scenario) == 1:
        axes = [axes]

    for ax, (cfg, runs) in zip(axes, runs_by_scenario):
        oracle_values = np.array([run["oracle_value"] for run in runs], dtype=float)
        safe_values = np.array([run["safe_value"] for run in runs], dtype=float)
        eb_values = np.array([run["eb_value"] for run in runs], dtype=float)
        cusum_values = np.array([run["cusum_value"] for run in runs], dtype=float)
        entropy_values = np.array([run["entropy_value"] for run in runs], dtype=float)
        prm_peak_values = np.array([run["prm_peak_value"] for run in runs], dtype=float)

        ax.boxplot(
            [
                oracle_values - safe_values,
                oracle_values - eb_values,
                oracle_values - cusum_values,
                oracle_values - entropy_values,
                oracle_values - prm_peak_values,
            ],
            tick_labels=["Hoeffding", "EmpBern", "CUSUM", "Entropy", "PRM"],
            showfliers=False,
        )
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_title(cfg.name.replace("_", " ").title())
        ax.set_ylabel("Oracle value - baseline value")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "monte_carlo_gaps.png", dpi=200)
    plt.close(fig)


def plot_average_drifts(runs_by_scenario: list[tuple[ScenarioConfig, list[dict[str, np.ndarray | int | float]]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(runs_by_scenario), figsize=(15, 4.5), sharey=True)
    if len(runs_by_scenario) == 1:
        axes = [axes]

    for ax, (cfg, runs) in zip(axes, runs_by_scenario):
        mean_true_mu = np.mean(np.stack([run["true_mu"] for run in runs]), axis=0)
        mean_proxy_mu = np.mean(np.stack([run["proxy_mu"] for run in runs]), axis=0)
        mean_boundary = np.mean([run["true_boundary"] for run in runs])
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
    fig.savefig(OUTPUT_DIR / "average_drifts.png", dpi=200)
    plt.close(fig)


def plot_observable_signals(runs_by_scenario: list[tuple[ScenarioConfig, list[dict[str, np.ndarray | int | float]]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(runs_by_scenario), figsize=(15, 4.5), sharey=True)
    if len(runs_by_scenario) == 1:
        axes = [axes]

    for ax, (cfg, runs) in zip(axes, runs_by_scenario):
        mean_entropy = np.mean(np.stack([run["entropy"] for run in runs]), axis=0)
        mean_hidden_shift = np.mean(np.stack([run["hidden_shift"] for run in runs]), axis=0)
        ax.plot(mean_entropy, linewidth=2.0, color="#a16207", label="Entropy proxy")
        ax.plot(mean_hidden_shift, linewidth=2.0, color="#0f766e", linestyle="--", label="Hidden-state shift proxy")
        ax.axhline(ENTROPY_THRESHOLD, color="#7c2d12", linestyle=":", linewidth=1.2, label="Entropy threshold")
        ax.set_title(cfg.name.replace("_", " ").title())
        ax.set_xlabel("Reasoning step")
        ax.set_ylabel("Observable magnitude")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "observable_signals.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    scenarios = [
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

    probe_count = 2048
    delta = 0.05
    run_count = 400

    runs_by_scenario: list[tuple[ScenarioConfig, list[dict[str, np.ndarray | int | float]]]] = []
    summary_rows: list[dict[str, float | str]] = []
    representatives: list[tuple[ScenarioConfig, dict[str, np.ndarray | int | float]]] = []

    for cfg in scenarios:
        runs = [simulate_single_run(rng=rng, cfg=cfg, probe_count=probe_count, delta=delta) for _ in range(run_count)]
        runs_by_scenario.append((cfg, runs))
        summary_rows.append(summarize_runs(cfg.name, runs))
        representatives.append((cfg, representative_run(runs)))

    write_summary_csv(summary_rows)
    plot_representatives(representatives)
    plot_gap_distributions(runs_by_scenario)
    plot_average_drifts(runs_by_scenario)
    plot_observable_signals(runs_by_scenario)

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

    print(f"Wrote outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()