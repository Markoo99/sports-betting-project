import pandas as pd
import numpy as np
from datetime import datetime


def build_advanced_features(path="data/cleaned_data.csv"):
    """
    Load the cleaned data needed 
    """

    df = pd.read_csv(path)

    
    # 1. Convert bookmaker moneylines to implied probabilities
    
    def moneyline_to_prob(ml):
        # American odds → implied probability
        # Positive ml: ml = +150 → prob = 100 / (150 + 100)
        # Negative ml: ml = -150 → prob = 150 / (150 + 100)
        if ml > 0:
            return 100 / (ml + 100) # it will return the implied probability for the favorite 
        else:
            return -ml / (-ml + 100) # returns the implied probability for the underdog

    if "team_prob" not in df.columns:
        df["team_prob"] = df["moneyLine"].apply(moneyline_to_prob)

    if "opp_prob" not in df.columns:
        df["opp_prob"] = df["opponentMoneyLine"].apply(moneyline_to_prob)

    
    # 2. Probability-derived features
    
    df["prob_diff"] = df["team_prob"] - df["opp_prob"] # shows how much the book favors our team vs the opponent
    df["prob_ratio"] = df["team_prob"] / (df["opp_prob"] + 1e-9) # it gives relative strength of the odds 
    df["prob_logit"] = np.log(df["team_prob"] / (1 - df["team_prob"] + 1e-9)) # log-odds for linear modeles
    df["book_margin"] = df["team_prob"] + df["opp_prob"] # the bookmaker's edge essentially 
    df["is_favorite"] = (df["team_prob"] > df["opp_prob"]).astype(int) # favorite if their probability > opponent probability

    # Nonlinearities for simple models
    df["team_prob_sq"] = df["team_prob"] ** 2
    df["opp_prob_sq"] = df["opp_prob"] ** 2
    df["team_opp_interaction"] = df["team_prob"] * df["opp_prob"]

    
    # 3. Spread-based features
    
    df["abs_spread"] = df["spread"].abs() 
    # Smooth probability-like transform of the spread
    df["spread_prob_like"] = 1 / (1 + np.exp(-df["spread"] / 6)) # with this, a large positive spread makes this ration converge towards 1 while a large negative spread makes it converge toward 0. /6 controls how quickly the curve moves from 0 to 1 as the spread changes. A smaller number leads to a steeper curve and the spread has a stronger marginal effect while a larger number would flatten out the curve, diminishing its effects

    
    # 4. Total (pace / scoring environment)
   
    df["scaled_total"] = df["total"] / 200.0 # this normalizes the game total to around 1. we divide by 200 because it is a very common total score. 

    
    # 5. Home/Away indicator
    # Used to indicate the home and away teams. Useful because playing at home gives an advanatage, influencing the probability
    df["is_home"] = (df["home/visitor"].str.lower() == "home").astype(int) # it attributes a value of 1 if the team is a home team and 0 if its away. Catches the fact that home teams win more often 

    
    # 6. Date & season timing features
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    df["is_early_season"] = (df["day_of_year"] < 60).astype(int) # =1 if day_of_year < 60. 
    df["is_late_season"] = (df["day_of_year"] > 140).astype(int) # = 0 if day_of_year > 140

    
    # 7. One-hot encode team + opponent
    # This is helpful because some teams systematically out/underperform given the same odds, spreads, etc.
    team_dummies = pd.get_dummies(df["team"], prefix="team") #dummies for each team
    opp_dummies = pd.get_dummies(df["opponent"], prefix="opp") #dummies for each opponent 

    df = pd.concat([df, team_dummies, opp_dummies], axis=1)

   
    # 8. Drop unused columns that shouldn't go into ML model
    # This step is necessary because it removes all the variables that shouldn't be used. It is either variables that do not help us with the probability because they are known only after the game is finished, or variables that have already bean transformed and prepared for use. Keeping those would result in target leakage, redundant information or poorly scaled inputs.
    drop_cols = [
        "score",
        "opponentScore",
        "home/visitor",
        "moneyLine",
        "opponentMoneyLine",
        "total",
        "secondHalfTotal",
        "spread",
        "date",
        "team",
    ]

    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Opponent raw name removed earlier
    if "opponent" in df.columns:
        df = df.drop(columns=["opponent"])

    # Make sure win is present and integer
    df["win"] = df["win"].astype(int)

    return df
