import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from src.simple_model.data_loading import load_raw_data
from src.simple_model.preprocessing import preprocess_data


def run_efficiency_analysis() -> None:
    df_raw = load_raw_data()
    df = preprocess_data(df_raw)

    # We’ll use a simple feature set
    feature_cols = ["team_prob", "opp_prob", "spread", "total", "secondHalfTotal"]
    target_col = "win"

    # Safety check: keep only rows without NaNs in these columns.
    cols_needed = feature_cols + [target_col]
    df_model = df[cols_needed].dropna()

    X = df_model[feature_cols]
    y = df_model[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    # Bookmaker benchmark  
    # Use bookmaker implied probability for the team as prediction.
    p_book = X_test["team_prob"].values

    # Make sure to avoid log(0) 
    eps = 1e-15
    p_book_clipped = np.clip(p_book, eps, 1 - eps)

    brier_book = brier_score_loss(y_test, p_book_clipped)
    logloss_book = log_loss(y_test, p_book_clipped)
    acc_book = accuracy_score(y_test, (p_book >= 0.5).astype(int))
    sharpness_book = np.std(p_book)

    # ---- Logistic regression model ----
    print("\n Training logistic regression model...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    p_model = clf.predict_proba(X_test)[:, 1]
    p_model_clipped = np.clip(p_model, eps, 1 - eps)

    brier_model = brier_score_loss(y_test, p_model_clipped)
    logloss_model = log_loss(y_test, p_model_clipped)
    acc_model = accuracy_score(y_test, (p_model >= 0.5).astype(int))
    sharpness_model = np.std(p_model)

    # ---- Print comparison ----
    print("\n================ MARKET EFFICIENCY ANALYSIS ================")
    print("All metrics on the SAME test set of games.\n")

    print("Bookmaker implied probabilities vs outcomes")
    print(f"  Accuracy      : {acc_book: .4f}")
    print(f"  Brier score   : {brier_book: .4f}")
    print(f"  Log-loss      : {logloss_book: .4f}")
    print(f"  Sharpness (std of p): {sharpness_book: .4f}\n")

    print("Logistic regression model vs outcomes")
    print(f"  Accuracy      : {acc_model: .4f}")
    print(f"  Brier score   : {brier_model: .4f}")
    print(f"  Log-loss      : {logloss_model: .4f}")
    print(f"  Sharpness (std of p): {sharpness_model: .4f}\n")

    print("Interpretation hints:")
    print("  • LOWER Brier and log-loss = better probabilistic predictions.")
    print("  • HIGHER sharpness = more confident probabilities.")
    print("  • If model beats bookmaker on Brier/log-loss, there may be exploitable signal.")
   


if __name__ == "__main__":
    run_efficiency_analysis()
