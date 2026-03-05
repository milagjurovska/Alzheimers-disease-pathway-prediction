"""
run_models.py
=============
Top-level script to train and compare all three ML models for
Alzheimer's disease pathway-membership prediction.

This script orchestrates the full machine-learning pipeline:
  1. Random Forest (Random Forest Classifier, scikit-learn)
  2. XGBoost      (Extreme Gradient Boosting, xgboost)
  3. Neural Net   (Multi-Layer Perceptron, PyTorch)

All three models are trained on the *same* feature matrix and evaluated
on the *same* held-out test set (80/20 stratified split, seed=42), so
their metrics are directly comparable.

Usage
-----
    # Activate your virtual environment first:
    .venv\\Scripts\\activate          # Windows
    source .venv/bin/activate        # Linux/macOS

    # Run all models (default):
    python run_models.py

    # Run a specific model only:
    python run_models.py --model rf
    python run_models.py --model xgb
    python run_models.py --model nn

    # Skip a model:
    python run_models.py --skip nn

Outputs
-------
- data/results/results_summary.csv   : side-by-side metric table
- data/visualizations/               : all plots (see list below)
    rf_feature_importances.png
    rf_cv_results.png
    rf_confusion_matrix.png
    xgb_training_curve.png
    xgb_shap_summary.png
    xgb_confusion_matrix.png
    nn_loss_curves.png
    nn_accuracy_curve.png
    nn_confusion_matrix.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "data", "results", "results_summary.csv"
)


def _print_summary(results: list[dict]) -> None:
    """
    Print a side-by-side comparison table of all model results.

    Parameters
    ----------
    results:
        List of metrics dicts returned by each model's training function.
    """
    if not results:
        print("\nNo results to display.")
        return

    df = pd.DataFrame(results)[["model", "accuracy", "macro_f1", "weighted_f1"]]
    df = df.sort_values("weighted_f1", ascending=False)

    print("\n" + "=" * 65)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 65)
    print(df.to_string(index=False, float_format="{:.4f}".format))
    print("=" * 65)
    print(f"\nFull results saved to: {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate AD pathway prediction models."
    )
    parser.add_argument(
        "--model",
        choices=["rf", "xgb", "nn", "all"],
        default="all",
        help="Which model to run (default: all).",
    )
    parser.add_argument(
        "--skip",
        choices=["rf", "xgb", "nn"],
        nargs="+",
        default=[],
        help="Model(s) to skip.",
    )
    args = parser.parse_args()

    run_all = args.model == "all"
    run_rf = (run_all or args.model == "rf") and "rf" not in args.skip
    run_xgb = (run_all or args.model == "xgb") and "xgb" not in args.skip
    run_nn = (run_all or args.model == "nn") and "nn" not in args.skip

    total_start = time.time()
    results: list[dict] = []

    if run_rf:
        from models.random_forest import train_random_forest
        metrics = train_random_forest(verbose=True)
        if metrics:
            results.append(metrics)

    if run_xgb:
        from models.xgboost_model import train_xgboost
        metrics = train_xgboost(verbose=True)
        if metrics:
            results.append(metrics)

    if run_nn:
        from models.neural_network import train_neural_network
        metrics = train_neural_network(verbose=True)
        if metrics:
            results.append(metrics)

    total_elapsed = time.time() - total_start
    _print_summary(results)
    print(f"\nTotal wall-clock time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
