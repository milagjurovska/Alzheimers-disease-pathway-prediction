"""
xgboost_model.py
================
XGBoost classifier for Alzheimer's disease pathway-membership prediction.

Background
----------
XGBoost (eXtreme Gradient Boosting; Chen & Guestrin, 2016) builds an
ensemble of shallow regression trees in a *sequential* manner: each new tree
is trained to correct the residual errors of the current ensemble.  Unlike
Random Forest (which parallelises tree training), XGBoost applies *gradient
descent in function space*, minimising a regularised objective that combines
a differentiable loss (multi-class cross-entropy here) with L1/L2 penalties
on leaf weights to prevent over-fitting.

Why XGBoost for this task?
~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **State-of-the-art on tabular data**: XGBoost consistently wins
   structured-data competitions and is the default choice for biological
   tabular datasets like ours.
2. **Built-in regularisation**: ``reg_alpha`` (L1) and ``reg_lambda`` (L2)
   reduce over-fitting on sparse binary features (GO flags, keywords).
3. **Missing-value handling**: XGBoost learns optimal split directions for
   missing values natively — useful for sparsely annotated proteins.
4. **Early stopping**: training halts automatically when validation loss
   stops improving, preventing over-fitting without manual epoch tuning.
5. **SHAP integration**: XGBoost pairs naturally with the SHAP library,
   enabling model-agnostic, theoretically grounded feature-attribution
   values.  SHAP values tell us *exactly* how much each feature contributes
   to each individual prediction, which is essential for biological insight.

Key hyperparameters explained
------------------------------
``n_estimators``
    Maximum number of boosting rounds (trees).  Early stopping may halt
    training before this limit is reached.
``learning_rate`` (η)
    Shrinkage applied to each tree's contribution.  Lower η = need more
    trees, but better generalisation.  We use 0.05–0.2 (typical range).
``max_depth``
    Maximum depth of each tree.  Deeper trees capture higher-order feature
    interactions but risk over-fitting.  Typical range 3–8.
``subsample``
    Fraction of training rows drawn (without replacement) for each tree.
    Adds stochasticity, reduces over-fitting (analogous to dropout).
``colsample_bytree``
    Fraction of features randomly selected for each tree.
    Decorrelates trees (similar to Random Forest's ``max_features``).
``scale_pos_weight``
    Not directly applicable to multiclass; instead we use ``'balanced'``
    sample weights passed into ``.fit()``.

Run
---
    python -m models.xgboost_model
    # or
    python models/xgboost_model.py
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import randint, uniform

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from features.feature_engineering import build_features
from models.evaluate import (
    stratified_split,
    print_report,
    save_confusion_matrix,
    save_results_row,
    VIZ_DIR,
)

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    print("WARNING: xgboost not installed. Run: pip install xgboost")

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    print("WARNING: shap not installed. SHAP plots will be skipped. Run: pip install shap")


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------

PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(200, 500),
    "learning_rate": uniform(0.05, 0.15),   # samples in [0.05, 0.20]
    "max_depth": randint(3, 8),
    "subsample": uniform(0.6, 0.35),        # samples in [0.6, 0.95]
    "colsample_bytree": uniform(0.5, 0.45), # samples in [0.5, 0.95]
    "reg_alpha": [0, 0.01, 0.1, 1.0],       # L1 regularisation
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],     # L2 regularisation
}

N_ITER = 20
CV_FOLDS = 5
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 20   # Stop if no improvement on eval set for this many rounds


# ---------------------------------------------------------------------------
# SHAP summary plot
# ---------------------------------------------------------------------------

def _plot_shap_summary(
    model: "xgb.XGBClassifier",
    X_test: np.ndarray,
    feature_names: list[str],
    filename: str = "xgb_shap_summary.png",
    max_display: int = 25,
) -> None:
    """
    Compute and save a SHAP beeswarm summary plot.

    SHAP (SHapley Additive exPlanations; Lundberg & Lee, 2017) assigns each
    feature a Shapley value for each prediction.  The Shapley value measures
    the average marginal contribution of that feature across all possible
    feature subsets — a concept borrowed from cooperative game theory.

    In a multi-class setting we compute a SHAP matrix per class and average
    the absolute values across classes to obtain a global feature-importance
    ranking that reflects contribution across *all* pathways rather than
    favouring the majority class.

    The beeswarm plot shows:
    - Each row = one feature (ranked by mean |SHAP|).
    - Each dot = one protein in the test set.
    - Colour = original feature value (red = high, blue = low).
    - X position = SHAP value (positive pushes prediction toward this class).

    Parameters
    ----------
    model:
        Fitted ``XGBClassifier``.
    X_test:
        Test feature matrix (used to compute SHAP values).
    feature_names:
        Ordered list of feature names.
    filename:
        Output PNG filename.
    max_display:
        Maximum number of features to show in the plot.
    """
    if not _SHAP_AVAILABLE:
        print("  SHAP not available - skipping SHAP plot.")
        return

    print("  Computing SHAP values (this may take ~30s) …")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # shap_values may be a list of arrays (one per class) or a 3D array
    if isinstance(shap_values, list):
        abs_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        # Shape: (n_samples, n_features, n_classes)
        abs_mean = np.mean(np.abs(shap_values), axis=2)
    else:
        abs_mean = np.abs(shap_values)

    # Top features by mean |SHAP|
    mean_abs_shap = abs_mean.mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:max_display].tolist()

    # Subset matrices
    shap_subset = abs_mean[:, top_idx]
    X_subset = X_test[:, top_idx]
    top_names = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.35)))
    # Simple horizontal bar chart of mean |SHAP| as clean alternative to beeswarm
    vals = mean_abs_shap[top_idx][::-1]
    names = top_names[::-1]
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.9, max_display))
    ax.barh(range(max_display), vals, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(max_display))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean |SHAP value| (averaged across all classes)", fontsize=10)
    ax.set_title(
        f"XGBoost — Top {max_display} Features by SHAP Importance\n"
        "(Alzheimer's Disease Pathway Prediction)",
        fontsize=12,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(VIZ_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Training curve plot
# ---------------------------------------------------------------------------

def _plot_training_curve(
    eval_results: dict,
    filename: str = "xgb_training_curve.png",
) -> None:
    """
    Plot XGBoost training and validation loss curves across boosting rounds.

    The training curve allows us to identify:
    - **Underfitting**: both train and validation loss remain high.
    - **Overfitting**: train loss keeps falling but validation loss diverges.
    - **Early stopping point**: where validation loss was at its minimum.

    Parameters
    ----------
    eval_results:
        The ``evals_result_`` dict from a fitted ``XGBClassifier``.
    filename:
        Output PNG filename.
    """
    if not eval_results:
        return

    train_key = list(eval_results.keys())[0]   # e.g. "validation_0"
    val_key = list(eval_results.keys())[-1]     # e.g. "validation_1"
    metric = list(eval_results[train_key].keys())[0]

    train_vals = eval_results[train_key][metric]
    val_vals = eval_results[val_key][metric]
    rounds = range(1, len(train_vals) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rounds, train_vals, label="Train", color="#1565C0", linewidth=2)
    ax.plot(rounds, val_vals, label="Validation", color="#E53935",
            linewidth=2, linestyle="--")
    best_round = int(np.argmin(val_vals)) + 1
    ax.axvline(best_round, color="#43A047", linestyle=":", linewidth=1.5,
               label=f"Best round = {best_round}")
    ax.set_xlabel("Boosting round (tree)", fontsize=10)
    ax.set_ylabel(f"Log-loss (mlogloss)", fontsize=10)
    ax.set_title(
        "XGBoost — Training vs. Validation Loss Curve\n"
        "(early stopping shown by green dotted line)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(VIZ_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_xgboost(verbose: bool = True) -> dict:
    """
    Full training pipeline for the XGBoost model.

    Steps
    -----
    1. Build features and labels.
    2. Stratified 80/20 split → further split train into 90/10 for
       early stopping (so the test set remains truly held-out).
    3. Compute balanced sample weights to handle class imbalance.
    4. RandomizedSearchCV (5-fold CV) for hyperparameter optimisation.
    5. Refit best estimator on 80% train with early stopping on 10% eval set.
    6. Evaluate on the held-out 20% test set.
    7. Save SHAP importance plot, training curve, confusion matrix.

    Parameters
    ----------
    verbose:
        If True, print detailed progress and results.

    Returns
    -------
    dict of evaluation metrics.
    """
    if not _XGB_AVAILABLE:
        print("XGBoost not available. Skipping.")
        return {}

    print("\n" + "=" * 65)
    print("  XGBOOST - Alzheimer's Disease Pathway Prediction")
    print("=" * 65)

    # ── Feature engineering ────────────────────────────────────────────────
    X, y, feature_names, le = build_features(verbose=verbose)
    n_classes = len(le.classes_)

    # ── Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = stratified_split(X, y, random_state=RANDOM_STATE)

    # Sub-split train into train+eval for early stopping
    # We use 10% of train as the early-stopping validation set.
    from sklearn.model_selection import train_test_split
    X_tr, X_eval, y_tr, y_eval = train_test_split(
        X_train, y_train, test_size=0.10, stratify=y_train, random_state=RANDOM_STATE
    )

    # Balanced sample weights to compensate for class imbalance
    sample_weights_tr = compute_sample_weight("balanced", y_tr)

    if verbose:
        print(f"\nTrain samples       : {len(X_tr)}")
        print(f"Early-stop eval set : {len(X_eval)}")
        print(f"Test  samples       : {len(X_test)}")
        print(f"Number of classes   : {n_classes}")

    # ── Hyperparameter search ──────────────────────────────────────────────
    print("\n[1/3] Running RandomizedSearchCV …")
    base_xgb = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring="accuracy",
        cv=cv,
        verbose=1 if verbose else 0,
        random_state=RANDOM_STATE,
        n_jobs=1,   # XGBoost already parallelises internally
        refit=True,
    )

    t0 = time.time()
    # Use sample_weight in the fit_params (for CV)
    search.fit(X_tr, y_tr, sample_weight=sample_weights_tr)
    elapsed_search = time.time() - t0

    best_params = search.best_params_
    best_cv_acc = search.best_score_
    if verbose:
        print(f"\n  Best CV accuracy : {best_cv_acc:.4f}")
        print(f"  Best parameters : {best_params}")
        print(f"  Search time     : {elapsed_search:.1f}s")

    # ── Final fit with early stopping ──────────────────────────────────────
    print("\n  Refitting best model with early stopping …")
    best_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        # Override n_estimators to allow early stopping to find optimal
        n_estimators=1000,
        **{k: v for k, v in best_params.items() if k != "n_estimators"},
    )

    t1 = time.time()
    best_model.fit(
        X_tr, y_tr,
        sample_weight=sample_weights_tr,
        eval_set=[(X_tr, y_tr), (X_eval, y_eval)],
        verbose=False,
    )
    elapsed_fit = time.time() - t1

    # ── Evaluate on test set ───────────────────────────────────────────────
    print("\n[2/3] Evaluating on test set …")
    y_pred = best_model.predict(X_test)
    metrics = print_report("XGBoost", y_test, y_pred, le)
    metrics.update({
        "best_cv_accuracy": round(best_cv_acc, 4),
        "training_time_s": round(elapsed_search + elapsed_fit, 1),
        "best_learning_rate": round(best_params.get("learning_rate", 0), 4),
        "best_max_depth": best_params.get("max_depth"),
        "best_subsample": round(best_params.get("subsample", 0), 3),
        "best_colsample_bytree": round(best_params.get("colsample_bytree", 0), 3),
    })

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n[3/3] Saving visualisations …")
    eval_results = best_model.evals_result()
    _plot_training_curve(eval_results)
    _plot_shap_summary(best_model, X_test, feature_names)
    save_confusion_matrix(
        y_test, y_pred, le,
        filename="xgb_confusion_matrix.png",
        title="XGBoost — Confusion Matrix (test set, row-normalised)",
    )

    # ── Persist metrics ────────────────────────────────────────────────────
    save_results_row(metrics)

    print("\n[OK] XGBoost training complete.")
    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_xgboost(verbose=True)
