"""Independently audit paired live outcomes and quantify their uncertainty.

The accuracy interval uses simultaneous Clopper-Pearson bounds on the two
discordance probabilities and a Bonferroni union bound. It is conservative
for iid task pairs, and does not presume independence between the categories.
The token-saving interval is a descriptive task-cluster percentile bootstrap.
Neither interval addresses model selection or new-model generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


def clopper_pearson(k: int, n: int, error: float) -> tuple[float, float]:
    if (isinstance(k, bool) or not isinstance(k, Integral)
            or isinstance(n, bool) or not isinstance(n, Integral) or n < 1 or not 0 <= k <= n):
        raise ValueError("Binomial counts must be integers with n >= 1 and 0 <= k <= n")
    if (isinstance(error, bool) or not isinstance(error, Real)
            or not math.isfinite(error) or not 0 < error < 1):
        raise ValueError("Binomial interval error must be finite and strictly between zero and one")
    return (0.0 if k == 0 else float(beta.ppf(error / 2, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(1 - error / 2, k + 1, n - k)))


def analyze(folder: Path, draws: int, seed: int) -> dict:
    if isinstance(draws, bool) or not isinstance(draws, Integral) or draws < 1:
        raise ValueError("Bootstrap draws must be a positive integer")
    source = folder / "live_paired_results.csv"
    token_columns = ["baseline_generated_tokens", "active_generated_tokens"]
    # Read identifiers literally (e.g. "NA") and token counts as exact decimal
    # text; float conversion can silently round an integer above 2**53.
    frame = pd.read_csv(source, dtype={key: "string" for key in ["task_id", *token_columns]},
                        keep_default_na=False)
    required = {"task_id", "baseline_correct", "active_correct", *token_columns}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing paired-result columns: {sorted(required - set(frame.columns))}")
    if (len(frame) == 0 or frame.task_id.str.strip().eq("").any()
            or frame.task_id.duplicated().any()):
        raise ValueError("One nonempty paired observation per unique task is required")
    for key in ("baseline_correct", "active_correct"):
        if not frame[key].isin([True, False, 0, 1]).all():
            raise ValueError(f"Not binary: {key}")
    n = len(frame)
    baseline = frame.baseline_correct.astype(int)
    active = frame.active_correct.astype(int)
    improved = int(((active == 1) & (baseline == 0)).sum())
    worsened = int(((active == 0) & (baseline == 1)).sum())
    lower_plus, upper_plus = clopper_pearson(improved, n, .025)
    lower_minus, upper_minus = clopper_pearson(worsened, n, .025)
    def exact_count(raw: str) -> int:
        try:
            if len(raw) > 1040:
                raise ValueError("Token-count literal exceeds supported size")
            value = Decimal(raw)
            # Bound before int: Decimal('1e1000000000') has a tiny textual
            # representation but would materialize an enormous integer.
            if (value.is_finite() and value.adjusted() < 1024 and value >= 0
                    and value == value.to_integral_value()):
                return int(value)
        except (InvalidOperation, ValueError):
            pass
        raise ValueError("Completion-token counts must be measured nonnegative integers")

    tokens = np.array([[exact_count(raw) for raw in row]
                       for row in frame[token_columns].itertuples(index=False, name=None)], dtype=object)
    baseline_tokens, active_tokens = (int(value) for value in tokens.sum(axis=0))
    if baseline_tokens == 0:
        raise ValueError("Baseline completion-token denominator is zero")
    rng = np.random.default_rng(seed)
    savings = []
    for start in range(0, draws, 1000):
        indices = rng.integers(0, n, size=(min(1000, draws - start), n))
        totals = tokens[indices].sum(axis=1)
        if (totals[:, 0] == 0).any():
            raise ValueError("A bootstrap sample has no baseline tokens")
        # Subtract integer totals before division so a small nonzero saving is
        # not cancelled by rounding two nearly equal floating-point ratios.
        savings.extend((totals[:, 0] - totals[:, 1]) / totals[:, 0])
    recorded = json.loads((folder / "live_metrics.json").read_text(encoding="utf-8"))
    manifest_path = folder / "live_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    purposive = manifest.get("source", {}).get("kind") == "custom_public_task_bank"
    expected = {"problems_or_trajectories": n, "paired_improved": improved, "paired_worsened": worsened,
                "baseline_correct": int(baseline.sum()), "active_correct": int(active.sum()),
                "baseline_generated_tokens": baseline_tokens, "active_generated_tokens": active_tokens}
    for key, value in expected.items():
        if recorded[key] != value:
            raise ValueError(f"Independent audit disagrees with reported {key}: {value} versus {recorded[key]}")
    return {"source": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "independent_aggregate_audit": "passed", "counts": expected,
            "accuracy_delta": float((active - baseline).mean()),
            "accuracy_delta_conservative_exact_95ci": [lower_plus - upper_minus, upper_plus - lower_minus],
            "accuracy_interval_method": "Bonferroni simultaneous 97.5% Clopper-Pearson intervals on paired improvement and worsening probabilities",
            "completion_token_savings": (baseline_tokens - active_tokens) / baseline_tokens,
            "completion_token_savings_cluster_bootstrap_95ci": [float(x) for x in np.quantile(savings, [.025, .975])],
            "bootstrap_draws": draws, "bootstrap_seed": seed,
            "scope": ("Handpicked question bank: iid-reference accuracy interval and descriptive resampling only; no randomized adversarial-population coverage"
                      if purposive else "iid task-pair reference model conditional on this fixed development policy; bootstrap interval is approximate and does not establish untouched external evaluation"),
            "noninferiority": "No noninferiority margin was prespecified; no noninferiority claim is made"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=804)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("draws must be positive")
    for folder in args.folders:
        result = analyze(folder, args.draws, args.seed)
        (folder / "live_uncertainty.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({"folder": str(folder), **result}))


if __name__ == "__main__":
    main()
