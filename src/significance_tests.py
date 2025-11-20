import os
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss
from scipy import stats
# Data + model setup:
def load_clean_data() -> pd.DataFrame:
  """ 
  This loads the preprocessed dataset from the preprocessing file. We need the following columns:
        - team_prob: bookmaker implied probability for the team we are tracking 
        - opp_prob: opponent implied probability
        - win: 1 if team won, 0 if opponent won 
  """
    df = pd.read_csv("data/cleaned_data.csv")
    required_cols = {"team_prob", "opp_prob", "win"}
    missing = required_cols - set(df.columns)
    if missing:
      raise ValueError(f"cleaned_data.csv is missing columns: {missing}")
    return df
def train_logistic_model(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """ Train a simple logistic regression model and return test-set probabilities """
    feature_cols = ["team_prob", "opp_prob"]
    
    X = df[feature_cols]
    y = df["win"].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    model_prob_test = model.predict_proba(X_test)[:, 1] #probability that our team wins according to the model 
    
    book_prob_test = X_test["team_prob"].values #bookmakers probabilities for the same game 

    return book_prob_test, model_prob_test, y_test.to_numpy()
""" Calibration: Hosmer-Lemeshow tests:
def hosmer_lemeshow(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> Tuple [float, float]:
    """ Hosmer-Lemeshow goodness-of-fit test """
    df = pd.DataFrame({"p": probs, "y": outcomes})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    grouped = df.groupby("bin")
    n = grouped.size().values
    p_hat = grouped["p"].mean().values #average predicted probability per bin 
    obs = grouped["y"].sum().values #obsereved wins per bin 
    
    exp = n * p_hat #expected wins per bin
    # HL Statistic
    hl_stat = np.sum((obs - exp) ** 2 / (n * p_hat * (1 - p_hat)))
    dof = len(n) - 2
    p_value = 1.0 - stats.chi2.cdf(hl_stat, dof)
    
    return float(hl_stat), float(p_value)
""" Paired tests for Brier + Log-Los: """
def paired_logloss_test(book_prob: np.ndarray, model_prob: np.ndarray, outcomes: np.ndarray) -> Tuple[float, float, float, float, float]:
    """ this will be used to compare bookmaker and model log-loss using a paired t-test """ 
    eps = 1e-15
    b = np.clip(book_prob, eps, 1 - eps)
    m = np.clip(model_prob, eps, 1 - eps)
    y = outcomes.astype(int)
    
    loss_book = -(y * np.log(b) + (1 - y) * np.log(1 - b))
    loss_model = -(y * np.log(m) + (1 - y) * np.log(1 - m))
    
    ll_book = float(loss_book.mean())
    ll_model = float(loss_model.mean())
    
    diff = loss_book - loss_model
    mean_diff = float(diff.mean())
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    t_stat = float(mean_diff / se)
    df = len(diff) - 1
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df)))
    
    return ll_book, ll_model, mean_diff, t_stat, p_value
def paired_brier_test(book_prob: np.ndarray, model_prob: np.ndarray, outcomes: np.ndarray) -> Tuple[float, float, float, float, float]
     """ this will be used to compare bookmaker and model Brier score using a paired t-test """ 
     y = outcomes.astype(int)

    se_book = (book_prob - y) ** 2
    se_model = (model_prob - y) ** 2
    
    brier_book = float(se_book.mean())
    brier_model = float(se_model.mean())
    
    diff = se_book - se_model
    mean_diff = float(diff.mean())
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    t_stat = float(mean_diff / se)
    df = len(diff) - 1
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df)))
    
    return brier_book, brier_model, mean_diff, t_stat, p_value

""" ROI significance for EV strategies) """
def roi_ttest(csv_path: str) -> Optional[Tuple[str, int, float, float, float]]:
    """ We want to see whether the mean ROI, meaning profit per bet, is statistically different from 0 """
    if not os.path.exists(csv_path):
        return None
    bets = pd.read_csv(csv_path)
    if "profit" not in bets.columns:
      raise ValueError(f"{csv_path} does not contain a 'profit' column.")

    profits = bets["profit"].to_numpy()
    n = len(profits)
    mean_profit = float(profits.mean())
    se = profits.std(ddof=1) / np.sqrt(n)
    t_stat = float(mean_profit / se)
    df = n - 1
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df)))
    
    name = os.path.basename(csv_path)
    return name, n, mean_profit, t_stat, p_value

""" Main Part """
def run_significance_tests() -> None:
    os.makedirs("results", exits_ok=True)
    print("Loading cleaned data and training logistic regression model...")
    df = load_clean_data()
    book_prob, model_prob, y_test = train_logistic_model(df)
    
    # Calibration 
    print("\n================ CALIBRATION TESTS (Hosmer–Lemeshow) ================\n")
    
    hl_book, p_book = hosmer_lemeshow(book_prob, y_test, n_bins=10)
    hl_model, p_model = hosmer_lemeshow(model_prob, y_test, n_bins=10)
    
    print("Bookmaker probabilities:")
    print(f"  HL statistic : {hl_book:.3f}")
    print(f"  p-value      : {p_book:.4f}")
    
    print("\nModel probabilities:")
    print(f"  HL statistic : {hl_model:.3f}")
    print(f"  p-value      : {p_model:.4f}")
    
    # Paired tests 
    print("\n================ PAIRED METRIC TESTS (Book vs Model) ================\n")
    
    b_book, b_model, b_diff, b_t, b_p = paired_brier_test(book_prob, model_prob, y_test)
    print("Brier score (lower is better):")
    print(f"  Bookmaker : {b_book:.6f}")
    print(f"  Model     : {b_model:.6f}")
    print(f"  Mean diff (Book - Model): {b_diff:.6f}")
    print(f"  t-statistic: {b_t:.3f}")
    print(f"  p-value    : {b_p:.4f}\n")
    
    ll_book, ll_model, ll_diff, ll_t, ll_p = paired_logloss_test(
        book_prob, model_prob, y_test
    )
    print("Log-loss (lower is better):")
    print(f"  Bookmaker : {ll_book:.6f}")
    print(f"  Model     : {ll_model:.6f}")
    print(f"  Mean diff (Book - Model): {ll_diff:.6f}")
    print(f"  t-statistic: {ll_t:.3f}")
    print(f"  p-value    : {ll_p:.4f}\n")
    
    # ROI EV strategy tests 
    print("================ ROI SIGNIFICANCE FOR EV STRATEGIES ================\n")
    
    ev_files: List[str] = [
        "results/ev_bets_edge_0.005.csv",
        "results/ev_bets_edge_0.010.csv",
        "results/ev_bets_edge_0.020.csv",
    ]
    
    seen = set()
    for path in ev_files:
        if path in seen:
            continue
        seen.add(path)
        res = roi_ttest(path)
        if res is None:
            continue
        name, n_bets, mean_profit, t_stat, p_value = res
        print(f"{name}:")
        print(f"  Number of bets : {n_bets}")
        print(f"  Mean profit/bet: {mean_profit:.6f}")
        print(f"  t-statistic    : {t_stat:.3f}")
        print(f"  p-value        : {p_value:.4f}\n")
    
    print("Use these p-values and effect sizes to discuss market efficiency.\n")
  if __name__=="__main__":
      run_signifcance_tests()




