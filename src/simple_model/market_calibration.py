import numpy as np
import pandas as pd

from src.simple_model.data_loading import load_raw_data
from src.simple_model.preprocessing import preprocess_data


def compute_calibration_table(
    df: pd.DataFrame,
    prob_col: str = "team_prob",
    outcome_col: str = "win",
    n_bins: int = 10,
) -> pd.DataFrame:
    data = df[[prob_col, outcome_col]].dropna().copy()

    data[prob_col] = data[prob_col].clip(0.0, 1.0)

    # Define bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(n_bins)]

    data["prob_bin"] = pd.cut(
        data[prob_col],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    # Group by bin
    grouped = (
        data.groupby("prob_bin")
        .agg(
            mean_prob=(prob_col, "mean"),
            win_rate=(outcome_col, "mean"),
            count=(outcome_col, "size"),
        )
        .reset_index()
    )

    bin_lowers = [bins[i] for i in range(n_bins)]
    bin_uppers = [bins[i + 1] for i in range(n_bins)]
    grouped["bin_lower"] = bin_lowers
    grouped["bin_upper"] = bin_uppers

    grouped.insert(0, "bin", grouped["prob_bin"])
    grouped = grouped.drop(columns=["prob_bin"])
    grouped = grouped[
        ["bin", "bin_lower", "bin_upper", "mean_prob", "win_rate", "count"]
    ]

    return grouped


def brier_score(df: pd.DataFrame, prob_col: str = "team_prob", outcome_col: str = "win") -> float:
    """
    Compute the Brier score: mean squared error between probability and outcome.
    Lower is better; perfectly calibrated + sharp model has low Brier score.
    """
    data = df[[prob_col, outcome_col]].dropna().copy()
    p = data[prob_col].clip(0.0, 1.0)
    y = data[outcome_col]
    return float(((p - y) ** 2).mean())


def run_calibration(n_bins: int = 10) -> None:
    """
    Load data, preprocess it, and print calibration statistics.
    This focuses on bookmaker implied probabilities (team_prob).
    """
    print(" Loading raw data...")
    df_raw = load_raw_data()

    print(" Preprocessing data...")
    df = preprocess_data(df_raw)

    print("\nData shape after preprocessing:", df.shape)
    print("Columns:", list(df.columns))

    # Basic sanity checks
    if "team_prob" not in df.columns or "win" not in df.columns:
        raise ValueError(
            "Expected columns 'team_prob' and 'win' to exist after preprocessing."
        )

    print("\n Computing calibration table for bookmaker implied probabilities...")
    calib_table = compute_calibration_table(df, prob_col="team_prob", outcome_col="win", n_bins=n_bins)

    print("\nCalibration table (per probability bucket):")
    print(calib_table.to_string(index=False))

    # Ideal calibration line would have win_rate ≈ mean_prob in every bin.
    score = brier_score(df, prob_col="team_prob", outcome_col="win")
    print(f"\nBrier score (bookmaker implied probabilities vs outcome): {score:.4f}")

    # Optional: overall average probability vs win rate
    avg_prob = df["team_prob"].mean()
    avg_win_rate = df["win"].mean()
    print(f"\nAverage implied win probability: {avg_prob:.3f}")
    print(f"Actual overall win rate:        {avg_win_rate:.3f}")

    print("\nInterpretation hints:")
    print("- If win_rate is consistently below mean_prob, bookmakers are overestimating chances (odds too short).")
    print("- If win_rate is consistently above mean_prob, bookmakers are underestimating chances (odds too long).")
    print("- If lines are close, the market is well calibrated.")


if __name__ == "__main__":
    # Default: 10 bins (0–0.1, 0.1–0.2, ..., 0.9–1.0)
    run_calibration(n_bins=10)
