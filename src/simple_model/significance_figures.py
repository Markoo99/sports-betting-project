import os
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.simple_model.significance_tests import (
    load_clean_data,
    train_logistic_model,
    hosmer_lemeshow,
    paired_brier_test,
    paired_logloss_test,
    roi_ttest,
)


def compute_core_metrics() -> Dict:
    """
    Recompute all key metrics and return them in a dict
    that plotting functions can use.
    """
    df = load_clean_data()
    book_prob, model_prob, y_test = train_logistic_model(df)

    # Calibration (HL)
    hl_book, p_book = hosmer_lemeshow(book_prob, y_test, n_bins=10)
    hl_model, p_model = hosmer_lemeshow(model_prob, y_test, n_bins=10)

    # Paired metrics
    b_book, b_model, b_diff, b_t, b_p = paired_brier_test(
        book_prob, model_prob, y_test
    )
    ll_book, ll_model, ll_diff, ll_t, ll_p = paired_logloss_test(
        book_prob, model_prob, y_test
    )

    # ROI results
    ev_files: List[str] = [
        "results/ev_bets_edge_0.005.csv",
        "results/ev_bets_edge_0.010.csv",
        "results/ev_bets_edge_0.020.csv",
    ]
    roi_results = []
    for path in ev_files:
        res = roi_ttest(path)
        if res is None:
            continue
        name, n_bets, mean_profit, t_stat, p_value = res
        threshold = path.split("_edge_")[-1].replace(".csv", "")
        roi_results.append(
            {
                "threshold": threshold,
                "file": name,
                "n_bets": n_bets,
                "mean_profit": mean_profit,
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )

    return {
        "book_prob": book_prob,
        "model_prob": model_prob,
        "y_test": y_test,
        "hl": {
            "book": hl_book,
            "model": hl_model,
            "p_book": p_book,
            "p_model": p_model,
        },
        "brier": {
            "book": b_book,
            "model": b_model,
            "diff": b_diff,
            "t": b_t,
            "p": b_p,
        },
        "logloss": {
            "book": ll_book,
            "model": ll_model,
            "diff": ll_diff,
            "t": ll_t,
            "p": ll_p,
        },
        "roi": roi_results,
    }


#  Simple bar charts (HL, Brier, Log-loss, ROI mean) 

def plot_hl_bar(results: Dict, outdir: str):
    hl_book = results["hl"]["book"]
    hl_model = results["hl"]["model"]

    plt.figure()
    labels = ["Bookmaker", "Model"]
    values = [hl_book, hl_model]
    plt.bar(labels, values)
    plt.ylabel("HL statistic")
    plt.title("Calibration (Hosmer–Lemeshow statistic)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hl_calibration.png"))
    plt.close()


def plot_brier_bar(results: Dict, outdir: str):
    b_book = results["brier"]["book"]
    b_model = results["brier"]["model"]

    plt.figure()
    labels = ["Bookmaker", "Model"]
    values = [b_book, b_model]
    plt.bar(labels, values)
    plt.ylabel("Brier score")
    plt.title("Brier Score Comparison (lower is better)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "brier_comparison.png"))
    plt.close()


def plot_logloss_bar(results: Dict, outdir: str):
    ll_book = results["logloss"]["book"]
    ll_model = results["logloss"]["model"]

    plt.figure()
    labels = ["Bookmaker", "Model"]
    values = [ll_book, ll_model]
    plt.bar(labels, values)
    plt.ylabel("Log-loss")
    plt.title("Log-loss Comparison (lower is better)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "logloss_comparison.png"))
    plt.close()


def plot_roi_bar(results: Dict, outdir: str):
    roi_results = results["roi"]
    if not roi_results:
        return

    thresholds = [r["threshold"] for r in roi_results]
    mean_profits = [r["mean_profit"] for r in roi_results]

    plt.figure()
    plt.bar(thresholds, mean_profits)
    plt.axhline(0.0, linestyle="--")
    plt.ylabel("Mean profit per bet")
    plt.xlabel("Edge threshold")
    plt.title("ROI by Edge Threshold")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roi_by_edge.png"))
    plt.close()


# More informative figures

def plot_calibration_curves(results: Dict, outdir: str, n_bins: int = 10):
    """
    Calibration curves: predicted vs observed probabilities
    for bookmaker and model, plus y=x reference line.
    """
    p_book = results["book_prob"]
    p_model = results["model_prob"]
    y = results["y_test"]

    def bin_curve(probs, outcomes):
        df = pd.DataFrame({"p": probs, "y": outcomes})
        df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
        grouped = df.groupby("bin")
        mean_p = grouped["p"].mean().values
        obs_rate = grouped["y"].mean().values
        return mean_p, obs_rate

    mean_p_book, obs_book = bin_curve(p_book, y)
    mean_p_model, obs_model = bin_curve(p_model, y)

    plt.figure()
    # y=x reference
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, grid, linestyle="--", label="Perfect calibration")

    plt.plot(mean_p_book, obs_book, marker="o", label="Bookmaker")
    plt.plot(mean_p_model, obs_model, marker="s", label="Model")

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed win rate")
    plt.title("Calibration Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "calibration_curves.png"))
    plt.close()


def plot_prob_distributions(results: Dict, outdir: str):
    """
    Histogram of predicted probabilities for bookmaker vs model.
    """
    p_book = results["book_prob"]
    p_model = results["model_prob"]

    plt.figure()
    bins = np.linspace(0, 1, 21)
    plt.hist(p_book, bins=bins, alpha=0.5, label="Bookmaker", density=True)
    plt.hist(p_model, bins=bins, alpha=0.5, label="Model", density=True)
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title("Distribution of Predicted Probabilities")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "probability_distributions.png"))
    plt.close()


def plot_roi_with_ci(results: Dict, outdir: str):
    """
    ROI with 95% confidence intervals: mean ± 1.96 * SE.
    """
    roi_results = results["roi"]
    if not roi_results:
        return

    thresholds = []
    means = []
    errors = []  # half-width of CI

    for r in roi_results:
        thresholds.append(r["threshold"])
        mean_profit = r["mean_profit"]
        # Reconstruct SE from t-stat if needed: t = mean / SE  => SE = mean / t
        t_stat = r["t_stat"]
        if t_stat != 0:
            se = mean_profit / t_stat
        else:
            se = 0.0
        ci_half = 1.96 * abs(se)
        means.append(mean_profit)
        errors.append(ci_half)

    plt.figure()
    x = np.arange(len(thresholds))
    plt.bar(x, means, yerr=errors, capsize=5)
    plt.axhline(0.0, linestyle="--")
    plt.xticks(x, thresholds)
    plt.xlabel("Edge threshold")
    plt.ylabel("Mean profit per bet")
    plt.title("ROI with 95% Confidence Intervals")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roi_with_ci.png"))
    plt.close()


def plot_prob_diff_by_decile(results: Dict, outdir: str, n_bins: int = 10):
    """
    Plot mean(model_prob - book_prob) by bookmaker probability decile.
    Shows where the model systematically deviates from the market.
    """
    p_book = results["book_prob"]
    p_model = results["model_prob"]

    df = pd.DataFrame({"book": p_book, "model": p_model})
    df["book_bin"] = pd.qcut(df["book"], q=n_bins, duplicates="drop")
    grouped = df.groupby("book_bin")
    mean_book = grouped["book"].mean().values
    diff = (grouped["model"].mean() - grouped["book"].mean()).values

    plt.figure()
    plt.plot(mean_book, diff, marker="o")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Bookmaker predicted probability (bin mean)")
    plt.ylabel("Model - bookmaker probability")
    plt.title("Model vs Bookmaker Probability Difference by Decile")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "probability_diff_by_decile.png"))
    plt.close()


def make_all_figures(output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    results = compute_core_metrics()

    # Simple bars
    plot_hl_bar(results, output_dir)
    plot_brier_bar(results, output_dir)
    plot_logloss_bar(results, output_dir)
    plot_roi_bar(results, output_dir)

    # Advanced plots
    plot_calibration_curves(results, output_dir)
    plot_prob_distributions(results, output_dir)
    plot_roi_with_ci(results, output_dir)
    plot_prob_diff_by_decile(results, output_dir)


if __name__ == "__main__":
    make_all_figures()
