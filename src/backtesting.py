from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data_loading import load_raw_data
from src.preprocessing import preprocess_data


FEATURE_COLUMNS = ["team_prob", "opp_prob", "spread", "total"]


def american_to_decimal(odds: float) -> float:
    if odds > 0:
        return 1.0 + odds / 100.0
    else:
        return 1.0 + 100.0 / (-odds)
    """ Converts American moneyline odds to decimal odds """

def make_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLUMNS].values
    y = df["win"].values
    return X, y


def train_model_for_backtest(
    df: pd.DataFrame,
) -> Tuple[LogisticRegression, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Train a logistic regression model and return the model plus
    train/test splits and targets.
    """
    X, y = make_feature_matrix(df)

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X,
        y,
        df,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, y_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    print("Model performance on test set (for reference):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Attach model probabilities to the test dataframe
    df_test = df_test.copy()
    df_test["model_prob"] = y_proba

    return model, df_train, df_test, y_train, y_test


def run_backtest(edge_threshold: float = 0.02) -> Dict[str, float]:
    """
    Run a simple backtest:
    - Train model on train set
    - On the test set, bet 1 unit whenever model_prob - team_prob > edge_threshold
    - Compute ROI and related stats.

    Parameters
    ----------
    edge_threshold : float
        Minimum difference (model_prob - implied_prob) required to place a bet.

    Returns
    -------
    Dict[str, float]
        Dictionary with number of bets, hit rate, total profit and ROI.
    """
    df_raw = load_raw_data()
    df_clean = preprocess_data(df_raw)

    model, df_train, df_test, _, _ = train_model_for_backtest(df_clean)

    # Decimal odds from moneyline
    df_test["decimal_odds"] = df_test["moneyLine"].apply(american_to_decimal)

    # Edge = model probability minus bookmaker implied probability
    df_test["edge"] = df_test["model_prob"] - df_test["team_prob"]

    # Strategy: bet on the team whenever edge exceeds threshold
    bets = df_test[df_test["edge"] > edge_threshold].copy()

    if bets.empty:
        print("\n No bets selected with this edge threshold.")
        return {
            "n_bets": 0.0,
            "hit_rate": 0.0,
            "total_profit": 0.0,
            "roi": 0.0,
        }

    # 1 unit stake per bet
    # If team wins -> profit = decimal_odds - 1 (net profit)
    # If team loses -> profit = -1
    bets["bet_return"] = np.where(
        bets["win"] == 1,
        bets["decimal_odds"] - 1.0,
        -1.0,
    )

    total_profit = float(bets["bet_return"].sum())
    n_bets = int(len(bets))
    hit_rate = float((bets["win"] == 1).mean())
    roi = total_profit / n_bets

    print("\n📈 Backtest results")
    print(f"  Edge threshold: {edge_threshold:.3f}")
    print(f"  Number of bets: {n_bets}")
    print(f"  Hit rate:       {hit_rate:.3f}")
    print(f"  Total profit:   {total_profit:.2f} units")
    print(f"  ROI per bet:    {roi:.3f} units")

    return {
        "n_bets": float(n_bets),
        "hit_rate": hit_rate,
        "total_profit": total_profit,
        "roi": roi,
    }


if __name__ == "__main__":
    # Default backtest with 2 percentage points edge
    run_backtest(edge_threshold=0.02)
