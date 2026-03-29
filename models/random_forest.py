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
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from features.feature_engineering import build_features
from models.evaluate import (

    stratified_split,
    print_report,
    save_confusion_matrix,
    save_results_row,
    VIZ_DIR,
)
PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(200, 600),
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": randint(1, 6),
    "max_features": ["sqrt", "log2", 0.3, 0.5],
}
N_ITER = 20
CV_FOLDS = 5
RANDOM_STATE = 42


def _plot_feature_importances(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 30,
    filename: str = "rf_feature_importances.png",
) -> None:
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


def _plot_cv_results(cv_results: dict, filename: str = "rf_cv_results.png") -> None:
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


def train_random_forest(verbose: bool = True) -> dict:
    print("\n" + "=" * 65)
    print("  RANDOM FOREST - Alzheimer's Disease Pathway Prediction")
    print("=" * 65)
    X, y, feature_names, le = build_features(verbose=verbose)
    X_train, X_test, y_train, y_test = stratified_split(X, y, random_state=RANDOM_STATE)
    if verbose:
        print(f"\nTrain samples : {len(X_train)}")
        print(f"Test  samples : {len(X_test)}")
    print("\n[1/3] Running RandomizedSearchCV …")
    base_rf = RandomForestClassifier(
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring="accuracy",
        cv=cv,
        verbose=1 if verbose else 0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
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
    _plot_cv_results(search.cv_results_)
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
    save_results_row(metrics)
    print("\n[OK] Random Forest training complete.")
    return metrics
if __name__ == "__main__":
    train_random_forest(verbose=True)
