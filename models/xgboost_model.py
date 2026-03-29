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
PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(200, 500),
    "learning_rate": uniform(0.05, 0.15),
    "max_depth": randint(3, 8),
    "subsample": uniform(0.6, 0.35),
    "colsample_bytree": uniform(0.5, 0.45),
    "reg_alpha": [0, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}
N_ITER = 20
CV_FOLDS = 5
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 20


def _plot_shap_summary(
    model: "xgb.XGBClassifier",
    X_test: np.ndarray,
    feature_names: list[str],
    filename: str = "xgb_shap_summary.png",
    max_display: int = 25,
) -> None:
    if not _SHAP_AVAILABLE:
        print("  SHAP not available - skipping SHAP plot.")
        return
    print("  Computing SHAP values (this may take ~30s) …")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        abs_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        abs_mean = np.mean(np.abs(shap_values), axis=2)
    else:
        abs_mean = np.abs(shap_values)
    mean_abs_shap = abs_mean.mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:max_display].tolist()
    shap_subset = abs_mean[:, top_idx]
    X_subset = X_test[:, top_idx]
    top_names = [feature_names[i] for i in top_idx]
    fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.35)))
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


def _plot_training_curve(
    eval_results: dict,
    filename: str = "xgb_training_curve.png",
) -> None:
    if not eval_results:
        return
    train_key = list(eval_results.keys())[0]
    val_key = list(eval_results.keys())[-1]
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


def train_xgboost(verbose: bool = True) -> dict:
    if not _XGB_AVAILABLE:
        print("XGBoost not available. Skipping.")
        return {}
    print("\n" + "=" * 65)
    print("  XGBOOST - Alzheimer's Disease Pathway Prediction")
    print("=" * 65)
    X, y, feature_names, le = build_features(verbose=verbose)
    n_classes = len(le.classes_)
    X_train, X_test, y_train, y_test = stratified_split(X, y, random_state=RANDOM_STATE)
    from sklearn.model_selection import train_test_split

    X_tr, X_eval, y_tr, y_eval = train_test_split(
        X_train, y_train, test_size=0.10, stratify=y_train, random_state=RANDOM_STATE
    )
    sample_weights_tr = compute_sample_weight("balanced", y_tr)
    if verbose:
        print(f"\nTrain samples       : {len(X_tr)}")
        print(f"Early-stop eval set : {len(X_eval)}")
        print(f"Test  samples       : {len(X_test)}")
        print(f"Number of classes   : {n_classes}")
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
        n_jobs=1,
        refit=True,
    )
    t0 = time.time()
    search.fit(X_tr, y_tr, sample_weight=sample_weights_tr)
    elapsed_search = time.time() - t0
    best_params = search.best_params_
    best_cv_acc = search.best_score_
    if verbose:
        print(f"\n  Best CV accuracy : {best_cv_acc:.4f}")
        print(f"  Best parameters : {best_params}")
        print(f"  Search time     : {elapsed_search:.1f}s")
    print("\n  Refitting best model with early stopping …")
    best_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
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
    print("\n[3/3] Saving visualisations …")
    eval_results = best_model.evals_result()
    _plot_training_curve(eval_results)
    _plot_shap_summary(best_model, X_test, feature_names)
    save_confusion_matrix(
        y_test, y_pred, le,
        filename="xgb_confusion_matrix.png",
        title="XGBoost — Confusion Matrix (test set, row-normalised)",
    )
    save_results_row(metrics)
    print("\n[OK] XGBoost training complete.")
    return metrics
if __name__ == "__main__":
    train_xgboost(verbose=True)
