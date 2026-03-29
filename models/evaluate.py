from __future__ import annotations
import os
from typing import Sequence
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
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

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
VIZ_DIR = os.path.join(BASE_DIR, "data", "visualizations")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def print_report(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    le: LabelEncoder,
) -> dict:
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


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    le: LabelEncoder,
    filename: str,
    title: str = "Confusion Matrix",
) -> None:
    class_names = list(le.classes_)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    np.divide(cm_norm, row_sums, out=cm_norm, where=row_sums != 0)
    n = len(class_names)
    fig_size = max(8, n * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (row-normalised)", fontsize=10)
    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=8)
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


def save_results_row(metrics: dict) -> None:
    path = os.path.join(RESULTS_DIR, "results_summary.csv")
    new_row = pd.DataFrame([metrics])
    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing = existing[existing["model"] != metrics["model"]]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(path, index=False)
    print(f"  Results saved -> {path}")
