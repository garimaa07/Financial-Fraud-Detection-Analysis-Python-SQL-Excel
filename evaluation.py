"""
evaluation.py
-------------
Loads the trained model and test split, generates all evaluation plots
(ROC curve, Precision-Recall curve, Confusion Matrix, Feature Importance),
and saves them to outputs/graphs/.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # Non-interactive backend for scripts
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH  = os.path.join("outputs", "model.pkl")
SPLITS_PATH = os.path.join("data", "splits", "train_test_splits.pkl")
GRAPHS_DIR  = os.path.join("outputs", "graphs")

# Consistent colour palette
PALETTE = {"fraud": "#e74c3c", "legit": "#2ecc71", "primary": "#2c3e50", "accent": "#3498db"}


def ensure_dirs():
    os.makedirs(GRAPHS_DIR, exist_ok=True)


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, save: bool = True):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Fraud"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Random Forest (Tuned)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(GRAPHS_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.close()


def plot_roc_curve(y_test, y_prob, save: bool = True):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color=PALETTE["accent"], lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["accent"])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("ROC Curve — Fraud Detection", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save:
        path = os.path.join(GRAPHS_DIR, "roc_curve.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.close()
    return roc_auc


def plot_precision_recall_curve(y_test, y_prob, save: bool = True):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color=PALETTE["fraud"], lw=2,
            label=f"PR Curve (AP = {ap:.4f})")
    ax.axhline(y=y_test.mean(), color="grey", linestyle="--",
               label=f"No-skill baseline ({y_test.mean():.4f})")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision–Recall Curve — Fraud Detection", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save:
        path = os.path.join(GRAPHS_DIR, "precision_recall_curve.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.close()


def plot_feature_importance(model, feature_names: list, top_n: int = 15, save: bool = True):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top_features[::-1], top_values[::-1],
                   color=PALETTE["accent"], edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xlabel("Importance Score (Gini)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest", fontsize=13, fontweight="bold")
    ax.set_xlim(0, top_values.max() * 1.15)
    plt.tight_layout()
    if save:
        path = os.path.join(GRAPHS_DIR, "feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.close()


def plot_model_comparison(results: list[dict], save: bool = True):
    """Bar chart comparing ROC-AUC, Recall, Precision, F1 across all models."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "model"} for r in results])
    df = df.set_index("name")
    metrics = ["roc_auc", "recall", "precision", "f1"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.2
    colors = [PALETTE["accent"], PALETTE["fraud"], PALETTE["legit"], PALETTE["primary"]]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, df[metric], width, label=metric.upper().replace("_", "-"),
                      color=color, edgecolor="white")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df.index, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0.7, 1.01)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — All Metrics", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(GRAPHS_DIR, "model_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.close()


# ── Main evaluation pipeline ──────────────────────────────────────────────────

def run_evaluation(model_path: str = MODEL_PATH,
                   splits_path: str = SPLITS_PATH) -> None:
    ensure_dirs()

    # Load model & test data
    model = joblib.load(model_path)
    _, X_test, _, y_test = joblib.load(splits_path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Print final classification report
    logger.info("\n=== Final Model Evaluation Report ===")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'])}")

    # Generate all plots
    plot_confusion_matrix(y_test, y_pred)
    roc_auc = plot_roc_curve(y_test, y_prob)
    plot_precision_recall_curve(y_test, y_prob)
    plot_feature_importance(model, list(X_test.columns))

    logger.info(f"\nAll evaluation plots saved to {GRAPHS_DIR}/")
    logger.info(f"Final ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    run_evaluation()
