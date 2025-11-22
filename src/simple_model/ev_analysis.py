from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.simple_model.data_loading import load_raw_data
from src.simple_model.preprocessing import preprocess_data
from src.simple_model.backtesting import train_model_for_backtest, american_to_decimal


def compute_ev_for_bets(edge_threshold: float = 0.02) -> pd.DataFrame:
    """
    Compute expected values (EV) for bets selected by the model.

    Parameters:
    
    edge_threshold : float
        Minimum difference between the probability predicted by our model and the bookmaker's probability required to place a bet.

    Returns
    
    pd.DataFrame
        Dataframe of selected bets with EVs and realized returns.
    """
    df_raw = load_raw_data()
    df_clean = preprocess_data(df_raw)

    # Train the model and get test set with model_prob already attached
    model, df_train, df_test, y_train, y_test = train_model_for_backtest(df_clean)

    df_test = df_test.copy()

    # Convert odds to decimal odds
    df_test["decimal_odds"] = df_test["moneyLine"].apply(american_to_decimal)

    # Edge = model probability - market implied probability
    df_test["edge"] = df_test["model_prob"] - df_test["team_prob"]

    # Filter bets: only where edge is above threshold
    bets = df_test[df_test["edge"] > edge_threshold].copy()

    if bets.empty:
        print("No bets selected for this edge threshold.")
        return bets

    # Payoff if we bet 1 unit: win -> decimal_odds - 1, loss -> -1
    bets["payoff_if_win"] = bets["decimal_odds"] - 1.0

    # Market-implied EV per bet (using bookmaker probability)
    bets["ev_market"] = (
        bets["team_prob"] * bets["payoff_if_win"]
        - (1.0 - bets["team_prob"])
    )

    # Model-implied EV per bet (using model probability)
    bets["ev_model"] = (
        bets["model_prob"] * bets["payoff_if_win"]
        - (1.0 - bets["model_prob"])
    )

    # Realized return based on actual outcome
    bets["realized_return"] = np.where(
        bets["win"] == 1,
        bets["payoff_if_win"],
        -1.0,
    )

    return bets


def summarize_ev(bets: pd.DataFrame) -> Dict[str, float]:
    """
    Summarize expected values and realized returns for a set of bets.
    """
    n_bets = int(len(bets))
    avg_ev_model = float(bets["ev_model"].mean())
    avg_ev_market = float(bets["ev_market"].mean())
    avg_realized = float(bets["realized_return"].mean())
    hit_rate = float((bets["win"] == 1).mean())

    print("\n EV analysis for selected bets")
    print(f"  Number of bets:       {n_bets}")
    print(f"  Hit rate:             {hit_rate:.3f}")
    print(f"  Avg EV (model):       {avg_ev_model:.4f} units")
    print(f"  Avg EV (market):      {avg_ev_market:.4f} units")
    print(f"  Avg realized return:  {avg_realized:.4f} units")

    return {
        "n_bets": float(n_bets),
        "hit_rate": hit_rate,
        "avg_ev_model": avg_ev_model,
        "avg_ev_market": avg_ev_market,
        "avg_realized": avg_realized,
    }


def run_ev_analysis(edge_threshold: float = 0.02) -> None:
    """
    Run the full EV-based efficiency analysis and save results to CSV.
    """
    bets = compute_ev_for_bets(edge_threshold=edge_threshold)

    if bets.empty:
        return

    metrics = summarize_ev(bets)

    # Save detailed bets to results folder
    output_path = "results/ev_bets_edge_{:.3f}.csv".format(edge_threshold)
    bets.to_csv(output_path, index=False)
    print(f"\n Detailed bet-level results saved to: {output_path}")


if __name__ == "__main__":
    run_ev_analysis(edge_threshold=0.02)
