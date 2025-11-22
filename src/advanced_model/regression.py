import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)

from build_advanced_features import build_advanced_features


def train_test_split_indices(y, test_size=0.2, random_state=42):
    """
    Create train/test splits in terms of row indices
    """
    idx = y.index
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # keep win/loss ratio similar in train/test
    )
    return train_idx, test_idx


def fit_logistic_regression(X_train, y_train, C=1.0):
    """
    Fit a regularized logistic regression model.
    C is the inverse of regularization strength (higher C = less regularization).
    """
    model = LogisticRegression(
        penalty="l2",
        C=C,
        max_iter=1000,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Print standard classification metrics for train and test sets.
    """
    # Probabilities for the positive class (win = 1)
    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    # Class predictions via 0.5 threshold (only for accuracy, not EV)
    train_pred = (train_proba >= 0.5).astype(int)
    test_pred = (test_proba >= 0.5).astype(int)

    # Train metrics
    train_acc = accuracy_score(y_train, train_pred)
    train_auc = roc_auc_score(y_train, train_proba)
    train_brier = brier_score_loss(y_train, train_proba)
    train_logloss = log_loss(y_train, train_proba)

    # Test metrics 
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)
    test_brier = brier_score_loss(y_test, test_proba)
    test_logloss = log_loss(y_test, test_proba)

    print(" TRAIN METRICS ")
    print(f"Accuracy:    {train_acc:.4f}")
    print(f"ROC AUC:     {train_auc:.4f}")
    print(f"Brier score: {train_brier:.4f}")
    print(f"Log loss:    {train_logloss:.4f}")
    print()
    print(" TEST METRICS ")
    print(f"Accuracy:    {test_acc:.4f}")
    print(f"ROC AUC:     {test_auc:.4f}")
    print(f"Brier score: {test_brier:.4f}")
    print(f"Log loss:    {test_logloss:.4f}")

    return {
        "train_proba": train_proba,
        "test_proba": test_proba,
    }


def print_top_coefficients(model, feature_names, top_n=20):
    """
    Show the largest positive and negative coefficients to see
    which features push the win probability up or down the most.
    """
    coefs = model.coef_[0]
    coef_df = pd.DataFrame(
        {"feature": feature_names, "coef": coefs, "abs_coef": np.abs(coefs)}
    ).sort_values("abs_coef", ascending=False)

    print("\ TOP POSITIVE / NEGATIVE COEFFICIENTS ")
    print(coef_df.head(top_n)[["feature", "coef"]])


def main():
    # 1. Build the advanced feature dataset
    df = build_advanced_features(path="data/cleaned_data.csv")

    # Target and features
    y = df["win"]
    X = df.drop(columns=["win"])

    feature_names = X.columns.tolist()

    # 2. Train/test split (by indices so we can map back)
    train_idx, test_idx = train_test_split_indices(y, test_size=0.2, random_state=42)

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    # 3. Fit logistic regression
    model = fit_logistic_regression(X_train, y_train, C=1.0)

    # 4. Evaluate model
    proba_dict = evaluate_model(model, X_train, y_train, X_test, y_test)

    # 5. Attach predictions back to full df and save
    df["logit_prob"] = np.nan
    df.loc[train_idx, "logit_prob"] = proba_dict["train_proba"]
    df.loc[test_idx, "logit_prob"] = proba_dict["test_proba"]

    df["set"] = "train"
    df.loc[test_idx, "set"] = "test"

    output_path = "data/advanced_logit_predictions.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved per-game predictions to: {output_path}")

    # 6. Inspect which features matter most
    print_top_coefficients(model, feature_names, top_n=25)


if __name__ == "__main__":
    main()
