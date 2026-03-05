"""
random_forest.py
================
Random Forest classifier for Alzheimer's disease pathway-membership prediction.

Background
----------
A Random Forest (Breiman, 2001) is an ensemble of decision trees trained on
bootstrap samples of the data ("bagging") with a random subset of features
considered at each split.  This "random subspace" trick decorrelates the
trees, reducing variance without increasing bias, making the ensemble much
more accurate than any single tree.

Why Random Forest for this task?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **Interpretability**: Feature importances (Gini impurity decrease) are
   easy to compute and explain in a biological context.
2. **Robustness to irrelevant features**: The random feature sub-sampling
   naturally ignores noisy or redundant annotation columns.
3. **No scaling required**: Decision trees are invariant to monotone
   transformations of individual features, so our mix of binary GO flags
   and continuous biophysical features can be used as-is.
4. **Handles class imbalance**: Setting ``class_weight='balanced'``
   automatically up-weights rare pathway classes during tree construction.

Hyper-parameter search
-----------------------
We perform a lightweight ``RandomizedSearchCV`` with 5-fold stratified
cross-validation over the most impactful Random Forest hyperparameters:

- ``n_estimators``  : number of trees (more → lower variance, slower).
- ``max_depth``     : maximum tree depth (None = grow fully; lower → more
  regularisation).
- ``min_samples_leaf``: minimum training samples needed in a leaf node
  (higher → smoother, more regularised trees).
- ``max_features``  : fraction of features considered at each split
  (lower → more diversity between trees).

Run
---
    python -m models.random_forest
    # or
    python models/random_forest.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import randint, uniform

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

warnings.filterwarnings("ignore")

# Make sure parent directory is on path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from features.feature_engineering import build_features
from models.evaluate import (
    stratified_split,
    print_report,
    save_confusion_matrix,
    save_results_row,
    VIZ_DIR,
)


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------

# The distributions below are sampled ``N_ITER`` times by RandomizedSearchCV.
# RandomizedSearchCV is preferred over GridSearchCV here because our feature
# space is large (200+ dimensions) and a full grid would be computationally
# prohibitive.
PARAM_DISTRIBUTIONS = {
    # More trees generally help; beyond ~500 the gains plateau.
    "n_estimators": randint(200, 600),
    # None = fully grown trees (low bias, high variance — offset by ensemble).
    # A finite max_depth regularises individual trees.
    "max_depth": [None, 10, 20, 30],
    # A leaf with fewer than ``min_samples_leaf`` samples will not be created,
    # preventing the model from memorising tiny sub-groups.
    "min_samples_leaf": randint(1, 6),
    # sqrt(n_features) is the classic default for classification.
    # "log2" further reduces correlation and speeds up training.
    "max_features": ["sqrt", "log2", 0.3, 0.5],
}

N_ITER = 20          # Number of random hyperparameter combinations to try
CV_FOLDS = 5         # Stratified k-fold cross-validation folds
RANDOM_STATE = 42    # Seed for reproducibility across all random operations


# ---------------------------------------------------------------------------
# Feature importance plot
# ---------------------------------------------------------------------------

def _plot_feature_importances(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 30,
    filename: str = "rf_feature_importances.png",
) -> None:
    """
    Plot and save the top-N feature importances as a horizontal bar chart.

    Feature importances in a Random Forest are computed as the mean decrease
    in Gini impurity (MDI) across all trees and all splits that used a given
    feature.  A feature with high MDI is relied upon by many trees for
    discriminating between pathway classes.

    Parameters
    ----------
    importances:
        Array of shape (n_features,) with MDI importances from the fitted RF.
    feature_names:
        Ordered list of feature name strings matching ``importances``.
    top_n:
        How many top features to display in the chart.
    filename:
        Output PNG filename (saved to ``data/visualizations/``).
    """
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    bars = ax.barh(
        range(top_n), top_vals[::-1],
        color=plt.cm.Blues(np.linspace(0.4, 0.9, top_n)),
        edgecolor="white", linewidth=0.5,
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Mean Decrease in Gini Impurity (MDI)", fontsize=10)
    ax.set_title(
        f"Random Forest — Top {top_n} Feature Importances\n"
        "(Alzheimer's Disease Pathway Prediction)",
        fontsize=12,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(VIZ_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Training cross-validation curve
# ---------------------------------------------------------------------------

def _plot_cv_results(cv_results: dict, filename: str = "rf_cv_results.png") -> None:
    """
    Plot mean cross-validation accuracy across hyperparameter iterations.

    Shows the distribution of 5-fold CV scores across all RandomizedSearchCV
    configurations, giving a sense of hyperparameter sensitivity.

    Parameters
    ----------
    cv_results:
        The ``.cv_results_`` dict from a fitted ``RandomizedSearchCV``.
    filename:
        Output PNG filename.
    """
    means = cv_results["mean_test_score"]
    stds = cv_results["std_test_score"]
    ranks = cv_results["rank_test_score"]
    sorted_idx = np.argsort(ranks)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(
        range(len(means)),
        means[sorted_idx],
        yerr=stds[sorted_idx],
        fmt="o-", color="#2196F3", ecolor="#BBDEFB",
        markersize=5, linewidth=1.5, capsize=3,
    )
    ax.set_xlabel("Hyperparameter configuration (sorted by rank)", fontsize=10)
    ax.set_ylabel("Mean 5-fold CV Accuracy", fontsize=10)
    ax.set_title(
        "Random Forest — RandomizedSearchCV Results\n"
        "Error bars = ±1 std across folds",
        fontsize=12,
    )
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(VIZ_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_random_forest(verbose: bool = True) -> dict:
    """
    Full training pipeline for the Random Forest model.

    Steps
    -----
    1. Build feature matrix and labels via ``feature_engineering.build_features``.
    2. Split into 80% train / 20% test (stratified).
    3. Run ``RandomizedSearchCV`` with 5-fold CV to find the best
       hyperparameters.
    4. Refit the best estimator on the full training set.
    5. Evaluate on the held-out test set.
    6. Save feature-importance plot, CV-results plot, and confusion matrix.
    7. Return a metrics dictionary for the shared results CSV.

    Parameters
    ----------
    verbose:
        If True, print detailed progress and results.

    Returns
    -------
    dict of evaluation metrics (accuracy, macro_f1, weighted_f1, etc.).
    """
    print("\n" + "=" * 65)
    print("  RANDOM FOREST - Alzheimer's Disease Pathway Prediction")
    print("=" * 65)

    # ── Feature engineering ────────────────────────────────────────────────
    X, y, feature_names, le = build_features(verbose=verbose)

    # ── Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = stratified_split(X, y, random_state=RANDOM_STATE)
    if verbose:
        print(f"\nTrain samples : {len(X_train)}")
        print(f"Test  samples : {len(X_test)}")

    # ── Hyperparameter search ──────────────────────────────────────────────
    print("\n[1/3] Running RandomizedSearchCV …")
    base_rf = RandomForestClassifier(
        class_weight="balanced",   # compensate for pathway class imbalance
        n_jobs=-1,                 # use all available CPU cores
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring="accuracy",   # CV metric; we report F1 separately on the test set
        cv=cv,
        verbose=1 if verbose else 0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,           # refit best estimator on full training data
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    best_params = search.best_params_
    best_cv_acc = search.best_score_
    if verbose:
        print(f"\n  Best CV accuracy  : {best_cv_acc:.4f}")
        print(f"  Best parameters   : {best_params}")
        print(f"  Search time       : {elapsed:.1f}s")

    # Save CV results plot
    _plot_cv_results(search.cv_results_)

    # ── Evaluate on test set ───────────────────────────────────────────────
    print("\n[2/3] Evaluating on test set …")
    best_rf = search.best_estimator_
    y_pred = best_rf.predict(X_test)
    metrics = print_report("Random Forest", y_test, y_pred, le)
    metrics.update({
        "best_cv_accuracy": round(best_cv_acc, 4),
        "training_time_s": round(elapsed, 1),
        "best_n_estimators": best_params.get("n_estimators"),
        "best_max_depth": str(best_params.get("max_depth")),
        "best_min_samples_leaf": best_params.get("min_samples_leaf"),
        "best_max_features": str(best_params.get("max_features")),
    })

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n[3/3] Saving visualisations …")
    _plot_feature_importances(
        best_rf.feature_importances_,
        feature_names,
        top_n=30,
    )
    save_confusion_matrix(
        y_test, y_pred, le,
        filename="rf_confusion_matrix.png",
        title="Random Forest — Confusion Matrix (test set, row-normalised)",
    )

    # ── Persist metrics ────────────────────────────────────────────────────
    save_results_row(metrics)

    print("\n[OK] Random Forest training complete.")
    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_random_forest(verbose=True)
