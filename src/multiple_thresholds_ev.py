import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import os

def load_data() -> pd.DataFrame:
    """Load the cleaned preprocessed dataset."""
    df_raw = load_raw_data() # this will read the data 
    df_clean = preprocess_data(df_raw) # this is meant to add all the necessary columns such as team_prob, win,  opp_prob, etc.
    return df_clean
def train_model(df):
    """Train logistic regression using same features as earlier."""
    X = df[["team_prob"]]  # model uses implied probs to calibrate
    y = df["win"]

    model = LogisticRegression()
    model.fit(X, y)

    df["model_prob"] = model.predict_proba(X)[:, 1]
    return df, model

def compute_ev_for_threshold(df, threshold):
    """
    Compute expected value at a given mispricing threshold.
    threshold = minimum |model_prob - market_prob|
    """
    df["edge"] = df["model_prob"] - df["team_prob"]

    # Bets taken
    bets = df[df["edge"].abs() >= threshold].copy()

    if len(bets) == 0:
        return {
            "threshold": threshold,
            "num_bets": 0,
            "hit_rate": None,
            "avg_ev_model": None,
            "avg_ev_market": None,
            "realized_return": None
        }

    # Market odds EV = p - (1-p)*odds
    bets["market_odds"] = bets["moneyLine"].abs() / 100
    bets["market_ev"] = bets["team_prob"] - (1 - bets["team_prob"]) * bets["market_odds"]

    # Model EV
    bets["model_ev"] = bets["model_prob"] - (1 - bets["model_prob"]) * bets["market_odds"]

    # Realized profit (1 unit per bet)
    bets["profit"] = np.where(bets["win"] == 1, 1, -bets["market_odds"])

    # Save bet-level file
    out_csv = f"results/ev_bets_edge_{threshold:.3f}.csv"
    bets.to_csv(out_csv, index=False)

    return {
        "threshold": threshold,
        "num_bets": len(bets),
        "hit_rate": bets["win"].mean(),
        "avg_ev_model": bets["model_ev"].mean(),
        "avg_ev_market": bets["market_ev"].mean(),
        "realized_return": bets["profit"].mean()
    }

def run_multi_threshold_ev():
    os.makedirs("results", exist_ok=True)

    df = load_data()
    df, model = train_model(df)

    thresholds = [0.005, 0.01, 0.02]
    results = []

    print("\nRunning multi-threshold EV analysis...\n")

    for t in thresholds:
        print(f"→ Evaluating threshold {t}...")
        res = compute_ev_for_threshold(df, t)
        results.append(res)

    # Save summary
    summary_path = "results/ev_threshold_summary.txt"
    with open(summary_path, "w") as f:
        f.write("EV SUMMARY ACROSS THRESHOLDS\n")
        f.write("=================================\n\n")
        for r in results:
            f.write(f"Threshold: {r['threshold']}\n")
            f.write(f"Number of bets: {r['num_bets']}\n")
            f.write(f"Hit rate: {r['hit_rate']}\n")
            f.write(f"Avg EV (model): {r['avg_ev_model']}\n")
            f.write(f"Avg EV (market): {r['avg_ev_market']}\n")
            f.write(f"Avg realized return: {r['realized_return']}\n")
            f.write("\n")

    print("\nSaved summary to:", summary_path)
    print("Saved bet-level CSVs for each threshold.\n")

if __name__ == "__main__":
    run_multi_threshold_ev()
