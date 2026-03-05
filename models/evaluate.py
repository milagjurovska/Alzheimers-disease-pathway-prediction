"""
evaluate.py
===========
Shared evaluation utilities used by all three model scripts.

This module provides a consistent evaluation harness so that performance
metrics are reported identically for the Random Forest, XGBoost, and Neural
Network models, making cross-model comparisons both fair and reproducible.

Functions
---------
stratified_split
    Reproducible 80/20 train/test split, stratified by class label.
print_report
    Pretty-print accuracy, per-class precision/recall/F1, and macro/weighted
    averages to stdout.
save_confusion_matrix
    Save a labelled heatmap of the confusion matrix as a PNG file.
save_results_row
    Append a single-row summary of a model's metrics to the shared
    ``data/results/results_summary.csv`` file.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file saving
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
VIZ_DIR = os.path.join(BASE_DIR, "data", "visualizations")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")

os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a reproducible stratified 80/20 train/test split.

    Stratification ensures that the class distribution in the test set
    mirrors the overall distribution, which is important here because the
    pathway classes are significantly imbalanced (the "No Pathway" and
    "Other" groups are much larger than specific Reactome pathway groups).

    Parameters
    ----------
    X:
        Feature matrix, shape (n_samples, n_features).
    y:
        Integer label array, shape (n_samples,).
    test_size:
        Fraction of samples to reserve for the test set (default 0.20).
    random_state:
        NumPy / sklearn random seed for reproducibility (default 42).

    Returns
    -------
    X_train, X_test, y_train, y_test : four numpy arrays.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Metric reporting
# ---------------------------------------------------------------------------

def print_report(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    le: LabelEncoder,
) -> dict:
    """
    Print a full classification report to stdout and return a metrics dict.

    The report includes:
    - Overall accuracy (fraction of proteins correctly classified).
    - Per-class precision, recall, and F1-score.
    - Macro-averaged F1 (mean of per-class F1, unweighted — sensitive to
      rare classes).
    - Weighted-averaged F1 (weighted by support — reflects real-world
      performance on the imbalanced distribution).

    Parameters
    ----------
    model_name:
        Human-readable model identifier printed as a section header.
    y_true:
        Ground-truth integer labels.
    y_pred:
        Model-predicted integer labels.
    le:
        The fitted ``LabelEncoder`` used to map integers → pathway names.

    Returns
    -------
    dict with keys ``model``, ``accuracy``, ``macro_f1``, ``weighted_f1``.
    """
    class_names = list(le.classes_)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'=' * 65}")
    print(f"  {model_name}  -  Evaluation Results")
    print(f"{'=' * 65}")
    print(f"  Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1          : {macro_f1:.4f}")
    print(f"  Weighted F1       : {weighted_f1:.4f}")
    print(f"{'-' * 65}")
    print(
        classification_report(
            y_true, y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )
    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    le: LabelEncoder,
    filename: str,
    title: str = "Confusion Matrix",
) -> None:
    """
    Save a normalised confusion-matrix heatmap to ``data/visualizations/``.

    The matrix is row-normalised (each row sums to 1.0) so that class size
    differences do not distort the visual interpretation.

    Parameters
    ----------
    y_true:
        Ground-truth integer labels.
    y_pred:
        Predicted integer labels.
    le:
        Fitted ``LabelEncoder`` for mapping indices to pathway names.
    filename:
        Output PNG filename (no directory prefix needed).
    title:
        Plot title.
    """
    class_names = list(le.classes_)
    cm = confusion_matrix(y_true, y_pred)

    # Row-normalise (recall per class)
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    np.divide(cm_norm, row_sums, out=cm_norm, where=row_sums != 0)

    n = len(class_names)
    fig_size = max(8, n * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

    # Colour-bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (row-normalised)", fontsize=10)

    # Axis ticks
    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=8)

    # Text annotations inside cells
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            raw = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(
                j, i,
                f"{val:.2f}\n({raw})",
                ha="center", va="center",
                color=color, fontsize=6,
            )

    ax.set_xlabel("Predicted label", fontsize=11, labelpad=10)
    ax.set_ylabel("True label", fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=13, pad=15)
    plt.tight_layout()

    out_path = os.path.join(VIZ_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Shared results CSV
# ---------------------------------------------------------------------------

def save_results_row(metrics: dict) -> None:
    """
    Append (or create) a row in ``data/results/results_summary.csv``.

    This CSV accumulates one row per trained model, allowing easy
    cross-model comparison in your report.  If the file already exists and
    contains a row for the same model name, the old row is replaced.

    Parameters
    ----------
    metrics:
        Dictionary with at minimum the keys returned by ``print_report``:
        ``model``, ``accuracy``, ``macro_f1``, ``weighted_f1``.
        Additional keys (e.g. training time, hyperparameters) are
        written as extra columns.
    """
    path = os.path.join(RESULTS_DIR, "results_summary.csv")

    new_row = pd.DataFrame([metrics])
    if os.path.exists(path):
        existing = pd.read_csv(path)
        # Replace row if same model name exists
        existing = existing[existing["model"] != metrics["model"]]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_csv(path, index=False)
    print(f"  Results saved -> {path}")
