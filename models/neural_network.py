from __future__ import annotations
import copy
import os
import sys
import time
import warnings
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed. Run: pip install torch")
from sklearn.preprocessing import StandardScaler

HIDDEN_DIM_1: int = 256
HIDDEN_DIM_2: int = 128
HIDDEN_DIM_3: int = 64
DROPOUT_1: float = 0.30
DROPOUT_2: float = 0.30
DROPOUT_3: float = 0.20
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE: int = 64
MAX_EPOCHS: int = 150
PATIENCE: int = 20
RANDOM_STATE: int = 42


class PathwayMLP(nn.Module):

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dims: tuple[int, ...] = (HIDDEN_DIM_1, HIDDEN_DIM_2, HIDDEN_DIM_3),
        dropout_rates: tuple[float, ...] = (DROPOUT_1, DROPOUT_2, DROPOUT_3),
    ) -> None:
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_features)
        layers: list[nn.Module] = []
        prev_dim = n_features
        for hidden_dim, drop_p in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_p),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.input_bn(x)
        return self.network(x)


def _compute_class_weights(y_train: np.ndarray, n_classes: int) -> "torch.Tensor":
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    best_epoch: int,
    filename: str = "nn_loss_curves.png",
) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, train_losses, label="Train loss", color="#1565C0", linewidth=2)
    ax.plot(epochs, val_losses, label="Validation loss", color="#E53935",
            linewidth=2, linestyle="--")
    ax.axvline(
        best_epoch + 1, color="#43A047", linestyle=":",
        linewidth=1.8, label=f"Best epoch = {best_epoch + 1}",
    )
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Cross-entropy loss", fontsize=10)
    ax.set_title(
        "Neural Network (MLP) — Training vs. Validation Loss\n"
        "(green dotted line = early-stopping checkpoint)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = os.path.join(VIZ_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_accuracy_curve(
    train_accs: list[float],
    val_accs: list[float],
    best_epoch: int,
    filename: str = "nn_accuracy_curve.png",
) -> None:
    epochs = range(1, len(train_accs) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, train_accs, label="Train accuracy", color="#1565C0", linewidth=2)
    ax.plot(epochs, val_accs, label="Validation accuracy", color="#E53935",
            linewidth=2, linestyle="--")
    ax.axvline(
        best_epoch + 1, color="#43A047", linestyle=":",
        linewidth=1.8, label=f"Best epoch = {best_epoch + 1}",
    )
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title(
        "Neural Network (MLP) — Training vs. Validation Accuracy",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = os.path.join(VIZ_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def train_neural_network(verbose: bool = True) -> dict:
    if not _TORCH_AVAILABLE:
        print("PyTorch not available. Skipping.")
        return {}
    print("\n" + "=" * 65)
    print("  NEURAL NETWORK (MLP) - Alzheimer's Disease Pathway Prediction")
    print("=" * 65)
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"\nUsing device: {device}")
    X, y, feature_names, le = build_features(verbose=verbose)
    n_features = X.shape[1]
    n_classes = len(le.classes_)
    X_train_raw, X_test_raw, y_train, y_test = stratified_split(
        X, y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train,
        test_size=0.10, stratify=y_train, random_state=RANDOM_STATE,
    )
    if verbose:
        print(f"\nTrain samples       : {len(X_tr)}")
        print(f"Validation samples  : {len(X_val)}")
        print(f"Test  samples       : {len(X_test_scaled)}")
        print(f"Feature dimensions  : {n_features}")
        print(f"Number of classes   : {n_classes}")

    def to_tensors(X_arr, y_arr):
        return TensorDataset(
            torch.tensor(X_arr, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.long),
        )
    train_loader = DataLoader(
        to_tensors(X_tr, y_tr),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        to_tensors(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    model = PathwayMLP(n_features, n_classes).to(device)
    class_weights = _compute_class_weights(y_tr, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
    )
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel parameters    : {n_params:,}")
        print(f"\nArchitecture:")
        print(model)
        print()
    print(f"\n[1/3] Training for up to {MAX_EPOCHS} epochs "
          f"(patience={PATIENCE}) …")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            ep_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            ep_correct += (preds == y_batch).sum().item()
            ep_total += len(y_batch)
        train_losses.append(ep_loss / ep_total)
        train_accs.append(ep_correct / ep_total)
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)
        mean_val_loss = val_loss / val_total
        val_losses.append(mean_val_loss)
        val_accs.append(val_correct / val_total)
        scheduler.step()
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if verbose and (epoch % 10 == 0 or epoch < 5):
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch+1:>3d}/{MAX_EPOCHS} | "
                f"train_loss={train_losses[-1]:.4f} "
                f"val_loss={val_losses[-1]:.4f} "
                f"val_acc={val_accs[-1]:.3f} "
                f"lr={lr_now:.6f}"
            )
        if patience_counter >= PATIENCE:
            if verbose:
                print(
                    f"\n  Early stopping at epoch {epoch + 1} "
                    f"(best epoch = {best_epoch + 1})"
                )
            break
    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    print("\n[2/3] Evaluating on test set …")
    model.eval()
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        test_logits = model(X_test_t)
        y_pred_t = test_logits.argmax(dim=1).cpu().numpy()
    metrics = print_report("Neural Network (MLP)", y_test, y_pred_t, le)
    metrics.update({
        "best_epoch": best_epoch + 1,
        "training_time_s": round(elapsed, 1),
        "n_parameters": sum(p.numel() for p in model.parameters()),
    })
    print("\n[3/3] Saving visualisations …")
    _plot_loss_curves(train_losses, val_losses, best_epoch)
    _plot_accuracy_curve(train_accs, val_accs, best_epoch)
    save_confusion_matrix(
        y_test, y_pred_t, le,
        filename="nn_confusion_matrix.png",
        title="Neural Network (MLP) — Confusion Matrix (test set, row-normalised)",
    )
    save_results_row(metrics)
    print("\n[OK] Neural Network training complete.")
    return metrics
if __name__ == "__main__":
    train_neural_network(verbose=True)
