import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

from src.simple_model.data_loading import load_raw_data
from src.simple_model.preprocessing import preprocess_data



# Preliminaries: load & preprocess

def load_clean_data() -> pd.DataFrame:
    """
    Load the raw odds CSV and run the preprocessing pipeline.
    """
    print ("Load Raw Data")
    df_raw = load_raw_data()
    print ("Clean The Raw Data")
    df_clean = preprocess_data(df_raw)

    print(f"Data shape after preprocessing: {df_clean.shape}")
    print("Columns:", list(df_clean.columns))
    print()
    return df_clean



# Calibration table for bookmaker probabilities
def make_bookmaker_calibration_table(
    df: pd.DataFrame,
    prob_col: str = "team_prob",
    outcome_col: str = "win",
    n_bins: int = 10,
) -> Tuple[pd.DataFrame, float]:
    """
    Build a calibration table for bookmaker implied probabilities.
    """
    probs = df[prob_col].values
    outcomes = df[outcome_col].values

    # Brier score
    brier = brier_score_loss(outcomes, probs)

    # Define bin edges, e.g. [0.0, 0.1, 0.2, ..., 1.0]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins, right=True)

    records = []
    for b in range(1, n_bins + 1):
        in_bin = bin_ids == b
        if in_bin.sum() == 0:
            continue

        bin_probs = probs[in_bin]
        bin_outcomes = outcomes[in_bin]

        bin_lower = bins[b - 1]
        bin_upper = bins[b]
        mean_prob = bin_probs.mean()
        win_rate = bin_outcomes.mean()
        count = int(in_bin.sum())

        records.append(
            {
                "bin": f"{bin_lower:.2f}-{bin_upper:.2f}",
                "bin_lower": bin_lower,
                "bin_upper": bin_upper,
                "mean_prob": mean_prob,
                "win_rate": win_rate,
                "count": count,
            }
        )

    calib_df = pd.DataFrame(records)

    return calib_df, float(brier)



# Plotting: calibration curve

def plot_bookmaker_calibration_curve(
    calib_df: pd.DataFrame,
    output_path: pathlib.Path,
    title: str = "Bookmaker calibration curve",
) -> None:
    """
    Plot mean predicted probability vs actual win rate and save as PNG.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    # Main calibration curve
    plt.plot(
        calib_df["mean_prob"],
        calib_df["win_rate"],
        marker="o",
        linestyle="-",
        label="Bookmaker",
    )

    # Perfect calibration diagonal
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    plt.xlabel("Implied win probability (bookmaker)")
    plt.ylabel("Actual win rate")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f" Saved calibration plot to: {output_path}")



# High-level runner

def run_calibration_analysis() -> None:
    """
    Main entry point:
      * load & preprocess data
      * build bookmaker calibration table
      * compute Brier score
      * print results
      * save calibration curve PNG in results/
    """
    df_clean = load_clean_data()

    print("Computing calibration table for bookmaker implied probabilities...\n")
    calib_df, brier = make_bookmaker_calibration_table(df_clean)

    # Print the table
    print("Calibration table (per probability bucket):")
    print(" bin      bin_lower  bin_upper  mean_prob  win_rate   count")
    for _, row in calib_df.iterrows():
        print(
            f"{row['bin']:>8}  "
            f"{row['bin_lower']:.2f}      "
            f"{row['bin_upper']:.2f}      "
            f"{row['mean_prob']:.4f}    "
            f"{row['win_rate']:.4f}    "
            f"{row['count']:5d}"
        )
    print()

    # Brier + overall stats
    avg_implied = df_clean["team_prob"].mean()
    overall_win = df_clean["win"].mean()

    print(f"Brier score (bookmaker implied probabilities vs outcome): {brier:.4f}")
    print()
    print(f"Average implied win probability: {avg_implied:.3f}")
    print(f"Actual overall win rate        : {overall_win:.3f}")
    print()

    print("Interpretation hints:")
    print("   If win_rate is consistently BELOW mean_prob,")
    print("    bookmakers are overestimating chances (odds too short).")
    print("   If win_rate is consistently ABOVE mean_prob,")
    print("    bookmakers are underestimating chances (odds too long).")
    print("   If the curve is close to the diagonal, the market is well calibrated.")
    print()

    # Save plot
    output_path = pathlib.Path("results") / "bookmaker_calibration_curve.png"
    plot_bookmaker_calibration_curve(calib_df, output_path)


if __name__ == "__main__":
    run_calibration_analysis()
