"""Phase H: symbolic regression and master equation distillation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "symbolic_regression"
WEIGHTS_PATH = Path(__file__).resolve().parent / "outputs" / "universal_feature_analysis" / "universal_hazard_weights.csv"
TRACES_PATH = Path(__file__).resolve().parent / "outputs" / "universal_feature_analysis" / "family_transition_summary.csv" # Actually need raw steps


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def master_law_heuristic(params: np.ndarray, x: pd.DataFrame) -> np.ndarray:
    """
    Proposed Universal Law of Overthinking (ULV1):
    Hazard = Sigmoid(k0 + k1*entropy + k2*latent_shift + k3*answer_changed*entropy)
    """
    k0, k1, k2, k3 = params
    # We use z-scored features from the analysis
    logits = k0 + k1 * x["entropy_mean"] + k2 * x["hidden_l2_shift"] + k3 * (x["answer_changed"] * x["entropy_mean"])
    return sigmoid(logits)


def load_all_trace_steps() -> pd.DataFrame:
    base = Path(__file__).resolve().parent / "outputs"
    runs = {
        "Qwen 0.5B": "real_traces_l4_qwen_0p5b",
        "DeepSeek 1.5B": "real_traces_l4_deepseek_1p5b",
        "Mistral 7B": "real_traces_l4_mistral_7b",
        "Qwen 7B": "real_traces_l4_qwen_7b_4bit",
    }
    frames = []
    for family, run_dir in runs.items():
        p = base / run_dir / "trace_steps.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["family"] = family
            # Basic z-scoring as done in Phase 3
            for col in ["entropy_mean", "hidden_l2_shift", "thought_token_count"]:
                if col in df.columns:
                    v = df[col].astype(float)
                    df[col] = (v - v.mean()) / (v.std() + 1e-9)
            
            # Prepare targets
            df["next_correct"] = df.groupby("run_id")["correct"].shift(-1)
            df["valid_corruption"] = (df["correct"] == 1) & df["next_correct"].notna()
            df["event_corruption"] = ((df["correct"] == 1) & (df["next_correct"] == 0)).astype(int)
            frames.append(df)
    return pd.concat(frames)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading empirical traces for master law distillation...")
    full_df = load_all_trace_steps()
    # Filter for corruption-eligible steps (Beta hazard)
    beta_df = full_df[full_df["valid_corruption"]].copy()
    
    if beta_df.empty:
        print("Error: No corruption events found in trace steps.")
        return

    y_true = beta_df["event_corruption"].values
    
    def objective(params: np.ndarray) -> float:
        y_pred = master_law_heuristic(params, beta_df)
        # Log-loss / Binary Cross Entropy
        eps = 1e-15
        loss = -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
        return float(loss)

    print("Optimizing parameters for Universal Law V1 (ULV1)...")
    initial_guess = np.array([-1.0, 0.5, 0.5, 1.0])
    res = minimize(objective, initial_guess, method="BFGS")
    
    final_params = res.x
    y_final_scores = master_law_heuristic(final_params, beta_df)
    auc = roc_auc_score(y_true, y_final_scores)
    
    print(f"Optimization Complete.")
    print(f"Final Weights: k0={final_params[0]:.4f}, k1={final_params[1]:.4f}, k2={final_params[2]:.4f}, k3={final_params[3]:.4f}")
    print(f"Symbolic Law AUC: {auc:.4f}")
    
    # Export Law
    law = {
        "version": "1.0-alpha",
        "equation": "Hazard = Sigmoid(k0 + k1*Entropy + k2*LatentShift + k3*AnswerChange*Entropy)",
        "parameters": {
            "k0_bias": float(final_params[0]),
            "k1_entropy": float(final_params[1]),
            "k2_latentshift": float(final_params[2]),
            "k3_interaction": float(final_params[3]),
        },
        "performance": {
            "beta_auc": float(auc),
            "optimization_residual": float(res.fun)
        }
    }
    
    with open(OUTPUT_DIR / "universal_law_v1.json", "w") as f:
        json.dump(law, f, indent=4)
    
    # Generate LaTeX
    latex = (
        r"\Lambda_{AlgorithmX}(\tau) = \sigma\left("
        f"{final_params[0]:.3f} + {final_params[1]:.3f} \cdot \bar{{H}} + "
        f"{final_params[2]:.3f} \cdot \Delta L_2 + "
        f"{final_params[3]:.3f} \cdot (A \cdot \bar{{H}})"
        r"\right)"
    )
    with open(OUTPUT_DIR / "law_latex.txt", "w") as f:
        f.write(latex)

    print(f"Artifacts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
