"""
model.py
--------
Trains multiple classifiers on the SMOTE-balanced training data,
performs hyperparameter tuning on the best candidate (Random Forest),
and serializes the final model to outputs/model.pkl.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score,
    recall_score, f1_score, precision_score
)
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

SPLITS_PATH = os.path.join("data", "splits", "train_test_splits.pkl")
MODEL_OUTPUT = os.path.join("outputs", "model.pkl")


# ── Helper ────────────────────────────────────────────────────────────────────

def load_splits(path: str = SPLITS_PATH):
    X_train, X_test, y_train, y_test = joblib.load(path)
    logger.info(f"Splits loaded — Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc   = roc_auc_score(y_test, y_prob)
    rec   = recall_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred)

    logger.info(f"\n{'='*50}")
    logger.info(f"Model        : {name}")
    logger.info(f"ROC-AUC      : {roc:.4f}")
    logger.info(f"Recall       : {rec:.4f}")
    logger.info(f"Precision    : {prec:.4f}")
    logger.info(f"F1 Score     : {f1:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Legit','Fraud'])}")

    return {"name": name, "roc_auc": roc, "recall": rec,
            "precision": prec, "f1": f1, "model": model}


# ── Baseline models ───────────────────────────────────────────────────────────

def train_baseline_models(X_train, y_train, X_test, y_test) -> list[dict]:
    """Train Logistic Regression, Decision Tree, Random Forest, XGBoost baselines."""
    baselines = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),
        ("Decision Tree",       DecisionTreeClassifier(max_depth=10, random_state=42)),
        ("Random Forest",       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("XGBoost",             XGBClassifier(n_estimators=100, use_label_encoder=False,
                                               eval_metric="logloss", random_state=42,
                                               n_jobs=-1, verbosity=0)),
    ]

    results = []
    for name, clf in baselines:
        logger.info(f"Training {name}…")
        clf.fit(X_train, y_train)
        results.append(evaluate_model(name, clf, X_test, y_test))

    return results


# ── Hyperparameter tuning ─────────────────────────────────────────────────────

RF_PARAM_GRID = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
    "max_features":      ["sqrt", "log2"],
}


def tune_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    GridSearchCV with StratifiedKFold, optimising for recall on the fraud class.
    Note: recall is used as scoring because in fraud detection, missing a fraud
    (false negative) is more costly than a false alarm (false positive).
    """
    logger.info("Starting Random Forest hyperparameter tuning (this may take a few minutes)…")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=RF_PARAM_GRID,
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    logger.info(f"Best params : {grid.best_params_}")
    logger.info(f"Best CV recall: {grid.best_score_:.4f}")
    return grid.best_estimator_


# ── Feature importance ────────────────────────────────────────────────────────

def log_feature_importance(model: RandomForestClassifier,
                            feature_names: list[str],
                            top_n: int = 15) -> pd.DataFrame:
    importances = model.feature_importances_
    feat_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(f"\nTop {top_n} features:\n{feat_df.head(top_n).to_string(index=False)}")
    return feat_df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_model_pipeline(splits_path: str = SPLITS_PATH,
                        model_out: str = MODEL_OUTPUT) -> dict:
    X_train, X_test, y_train, y_test = load_splits(splits_path)

    # Step 1: Baseline comparison
    logger.info("\n>>> STEP 1: Baseline model comparison")
    baseline_results = train_baseline_models(X_train, y_train, X_test, y_test)

    # Step 2: Tune best model (Random Forest)
    logger.info("\n>>> STEP 2: Hyperparameter tuning — Random Forest")
    tuned_rf = tune_random_forest(X_train, y_train)
    tuned_result = evaluate_model("Random Forest (Tuned)", tuned_rf, X_test, y_test)

    # Step 3: Feature importance
    feat_importance = log_feature_importance(tuned_rf, list(X_train.columns))

    # Step 4: Compare before/after tuning
    base_rf = next(r for r in baseline_results if r["name"] == "Random Forest")
    recall_gain = (tuned_result["recall"] - base_rf["recall"]) / base_rf["recall"] * 100
    logger.info(f"\n>>> Recall improvement after tuning: +{recall_gain:.1f}%")

    # Step 5: Save model
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(tuned_rf, model_out)
    logger.info(f"Model saved to {model_out}")

    return {
        "baselines":         baseline_results,
        "tuned_model":       tuned_result,
        "feature_importance": feat_importance,
        "recall_gain_pct":   recall_gain,
    }


if __name__ == "__main__":
    run_model_pipeline()
