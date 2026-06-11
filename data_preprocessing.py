"""
data_preprocessing.py
---------------------
Loads raw credit card transaction data, performs cleaning,
normalization, and saves the cleaned dataset for downstream use.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = os.path.join("data", "raw_data.csv")
CLEANED_DATA_PATH = os.path.join("data", "cleaned_data.csv")


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV data."""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Log basic data quality stats."""
    logger.info("=== Data Overview ===")
    logger.info(f"Shape         : {df.shape}")
    logger.info(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    logger.info(f"Duplicate rows: {df.duplicated().sum()}")
    fraud_pct = df["Class"].mean() * 100
    logger.info(f"Fraud rate    : {fraud_pct:.4f}%  ({df['Class'].sum():,} fraudulent transactions)")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.info(f"Removed {removed:,} duplicate rows")
    return df


def normalize_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    StandardScale the 'Amount' column.
    The 'Time' column is kept as-is for feature engineering later.
    """
    scaler = StandardScaler()
    df["Amount_Scaled"] = scaler.fit_transform(df[["Amount"]])
    logger.info("Normalized 'Amount' → 'Amount_Scaled'")
    return df


def drop_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop original unscaled Amount (replaced by Amount_Scaled)."""
    df = df.drop(columns=["Amount"])
    logger.info("Dropped raw 'Amount' column (replaced by Amount_Scaled)")
    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Move 'Class' label to the last column for convention."""
    cols = [c for c in df.columns if c != "Class"] + ["Class"]
    return df[cols]


def save_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Cleaned data saved to {path}  ({len(df):,} rows)")


def preprocess_pipeline(raw_path: str = RAW_DATA_PATH,
                         out_path: str = CLEANED_DATA_PATH) -> pd.DataFrame:
    df = load_data(raw_path)
    inspect_data(df)
    df = remove_duplicates(df)
    df = normalize_amount(df)
    df = drop_raw_columns(df)
    df = reorder_columns(df)
    save_data(df, out_path)
    return df


if __name__ == "__main__":
    preprocess_pipeline()
