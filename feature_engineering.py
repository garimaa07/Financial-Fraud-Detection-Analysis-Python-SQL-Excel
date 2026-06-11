"""
feature_engineering.py
-----------------------
Creates domain-informed features from cleaned transaction data,
then applies SMOTE to balance the highly imbalanced fraud class.
Saves the final training-ready dataset.
"""

import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

CLEANED_PATH  = os.path.join("data", "cleaned_data.csv")
FEATURED_PATH = os.path.join("data", "featured_data.csv")
SPLITS_DIR    = os.path.join("data", "splits")


# ── Feature creation ──────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    The 'Time' column = seconds elapsed since first transaction.
    Convert to hour-of-day (0–23) assuming dataset starts at midnight.
    """
    df["hour_of_day"] = (df["Time"] // 3600 % 24).astype(int)
    df["high_risk_hour"] = df["hour_of_day"].apply(
        lambda h: 1 if (h >= 23 or h <= 4) else 0
    )
    logger.info("Added time features: hour_of_day, high_risk_hour")
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Amount_Scaled is already standardized; derive statistical anomaly flags.
    """
    # Z-score is the same as Amount_Scaled (mean=0, std=1) but kept explicit
    df["amount_zscore"] = df["Amount_Scaled"]

    # Log of original amount requires reconstruction — use Amount_Scaled proxy
    # (In real usage you'd pass the original Amount before dropping it)
    # We add a flag for unusually large transactions (top 1%)
    threshold = df["Amount_Scaled"].quantile(0.99)
    df["high_value_flag"] = (df["Amount_Scaled"] > threshold).astype(int)
    logger.info("Added amount features: amount_zscore, high_value_flag")
    return df


def add_pca_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    V1–V28 are PCA components from original (anonymized) features.
    Add two interaction terms known to correlate with fraud from literature.
    """
    if "V1" in df.columns and "V3" in df.columns:
        df["V1_V3_interact"] = df["V1"] * df["V3"]
    if "V4" in df.columns and "V11" in df.columns:
        df["V4_V11_interact"] = df["V4"] * df["V11"]
    logger.info("Added PCA interaction features: V1_V3_interact, V4_V11_interact")
    return df


def feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_pca_interaction_features(df)
    # Drop raw 'Time' — no longer needed after hour extraction
    df = df.drop(columns=["Time"], errors="ignore")
    return df


# ── SMOTE resampling ───────────────────────────────────────────────────────────

def apply_smote(X: pd.DataFrame, y: pd.Series,
                random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to balance fraud vs. non-fraud in the training set.
    NEVER apply SMOTE to the test set — that would leak synthetic data
    into evaluation and inflate metrics.
    """
    logger.info(f"Before SMOTE — Fraud: {y.sum():,}  |  Non-fraud: {(y==0).sum():,}")
    sm = SMOTE(random_state=random_state, sampling_strategy=1.0)
    X_res, y_res = sm.fit_resample(X, y)
    logger.info(f"After  SMOTE — Fraud: {y_res.sum():,}  |  Non-fraud: {(y_res==0).sum():,}")
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name="Class")


# ── Train/Test split ──────────────────────────────────────────────────────────

def split_and_save(df: pd.DataFrame,
                   test_size: float = 0.2,
                   random_state: int = 42) -> dict:
    """
    Splits into train/test BEFORE SMOTE, applies SMOTE only to train.
    Returns a dict of X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    # Apply SMOTE only to training data
    X_train_res, y_train_res = apply_smote(X_train, y_train, random_state)

    os.makedirs(SPLITS_DIR, exist_ok=True)
    joblib.dump((X_train_res, X_test, y_train_res, y_test),
                os.path.join(SPLITS_DIR, "train_test_splits.pkl"))
    logger.info(f"Splits saved to {SPLITS_DIR}/train_test_splits.pkl")

    return {
        "X_train": X_train_res,
        "X_test":  X_test,
        "y_train": y_train_res,
        "y_test":  y_test,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_feature_pipeline(cleaned_path: str = CLEANED_PATH,
                          out_path: str = FEATURED_PATH) -> dict:
    df = pd.read_csv(cleaned_path)
    logger.info(f"Loaded cleaned data: {df.shape}")

    df = feature_pipeline(df)
    logger.info(f"Feature-engineered shape: {df.shape}")

    df.to_csv(out_path, index=False)
    logger.info(f"Featured dataset saved to {out_path}")

    splits = split_and_save(df)
    return splits


if __name__ == "__main__":
    run_feature_pipeline()
