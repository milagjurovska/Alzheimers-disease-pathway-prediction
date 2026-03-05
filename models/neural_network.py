"""
neural_network.py
=================
Multi-Layer Perceptron (MLP) classifier for Alzheimer's disease
pathway-membership prediction.

Background
----------
An Artificial Neural Network (ANN) learns a hierarchy of increasingly
abstract representations of the input features through a stack of
parameterised linear transformations followed by non-linear activations.
The MLP used here is a fully-connected, *feed-forward* architecture — the
simplest deep-learning approach and a natural baseline before considering
graph neural networks or transformer models.

Why a Neural Network for this task?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **Non-linear feature interactions**: GO terms and subcellular-location
   flags have complex combinatorial relationships that linear models miss.
   Stacked layers learn these interactions automatically.
2. **Soft probability outputs**: the softmax output gives calibrated
   posterior probabilities P(pathway | protein features), useful for
   thresholding predictions in low-confidence regions.
3. **Regularisation through dropout and weight decay**: our feature matrix
   is moderately high-dimensional (~200 columns) relative to sample size
   (~1 000 proteins), creating a risk of over-fitting.  Dropout (Srivastava
   et al., 2014) randomly zeros a fraction of activations during training,
   acting as an implicit ensemble; AdamW weight decay (Loshchilov & Hutter,
   2019) penalises large weights.
4. **Learning rate scheduling**: a cosine annealing schedule smoothly
   reduces the learning rate from its initial value to near-zero, allowing
   the optimiser to escape local minima early in training and then converge
   to a flat minimum (which generally correlates with better generalisation).

Architecture
------------
    Input (n_features)
       │
    BatchNorm1d               ← normalise input; stabilises training
       │
    Linear(n_features → 256)
    ReLU
    Dropout(0.30)             ← 30 % of neurons zeroed per forward pass
       │
    Linear(256 → 128)
    ReLU
    Dropout(0.30)
       │
    Linear(128 → 64)
    ReLU
    Dropout(0.20)             ← lighter dropout closer to the output head
       │
    Linear(64 → n_classes)    ← raw logits; softmax applied by loss function
       │
    CrossEntropyLoss           ← combines LogSoftmax + NLLLoss for numerical
                                  stability (avoids explicit softmax in forward)

Training loop
-------------
- **Optimiser**: AdamW (adaptive moment + decoupled weight decay).
  Decoupled weight decay (as opposed to standard L2 in Adam) has been shown
  to improve generalisation on many benchmarks.
- **Loss**: multi-class cross-entropy with manually computed class weights to
  compensate for the imbalance between large generic classes (e.g.
  "No Pathway") and small specific Reactome pathway classes.
- **Early stopping**: if the validation loss does not improve for
  ``PATIENCE`` consecutive epochs, training is halted and the checkpoint
  with the lowest validation loss is restored — preventing over-fitting without
  requiring a fixed epoch count.
- **Cosine Annealing LR**: ``CosineAnnealingLR`` decreases the LR following a
  cosine curve from ``lr`` to ``eta_min=1e-6`` over ``T_max`` epochs.

Run
---
    python -m models.neural_network
    # or
    python models/neural_network.py
"""

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


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HIDDEN_DIM_1: int = 256
HIDDEN_DIM_2: int = 128
HIDDEN_DIM_3: int = 64
DROPOUT_1: float = 0.30     # dropout after first hidden layer
DROPOUT_2: float = 0.30     # dropout after second hidden layer
DROPOUT_3: float = 0.20     # dropout after third hidden layer (lighter)

LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4  # AdamW L2 penalty coefficient
BATCH_SIZE: int = 64
MAX_EPOCHS: int = 150
PATIENCE: int = 20          # early-stopping patience (epochs without improvement)
RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class PathwayMLP(nn.Module):
    """
    Feed-forward MLP for multi-class pathway classification.

    Parameters
    ----------
    n_features:
        Number of input features (determined by feature engineering).
    n_classes:
        Number of pathway classes (determined by label encoding).
    hidden_dims:
        Tuple of hidden layer sizes.  Default: (256, 128, 64).
    dropout_rates:
        Dropout probability after each hidden layer.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dims: tuple[int, ...] = (HIDDEN_DIM_1, HIDDEN_DIM_2, HIDDEN_DIM_3),
        dropout_rates: tuple[float, ...] = (DROPOUT_1, DROPOUT_2, DROPOUT_3),
    ) -> None:
        super().__init__()

        # Input batch normalisation: scales each feature to zero mean and unit
        # variance *per batch*, stabilising training when input features have
        # very different scales (e.g. sequence length vs. binary GO flags).
        self.input_bn = nn.BatchNorm1d(n_features)

        layers: list[nn.Module] = []
        prev_dim = n_features
        for hidden_dim, drop_p in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),          # ReLU: simple, fast, avoids vanishing gradient
                nn.Dropout(drop_p),
            ])
            prev_dim = hidden_dim

        # Output layer: no activation here; CrossEntropyLoss applies LogSoftmax
        layers.append(nn.Linear(prev_dim, n_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        Logit tensor of shape (batch_size, n_classes).
        """
        x = self.input_bn(x)
        return self.network(x)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _compute_class_weights(y_train: np.ndarray, n_classes: int) -> "torch.Tensor":
    """
    Compute inverse-frequency class weights for the cross-entropy loss.

    This gives higher weight to under-represented pathway classes, preventing
    the model from ignoring small but biologically specific groups in favour
    of the large "No Pathway" / "Other" buckets.

    Parameters
    ----------
    y_train:
        Training label array.
    n_classes:
        Total number of distinct classes.

    Returns
    -------
    Float tensor of shape (n_classes,) for use with ``nn.CrossEntropyLoss``.
    """
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)   # avoid division by zero
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    best_epoch: int,
    filename: str = "nn_loss_curves.png",
) -> None:
    """
    Plot training and validation loss curves with the early-stopping epoch marked.

    The loss curves provide the most transparent view of the neural network's
    learning dynamics:
    - A rapidly falling training loss with a plateauing validation loss
      indicates over-fitting.
    - Both losses moving together indicates healthy generalisation.
    - The vertical dotted line marks where early stopping halted training
      (i.e. where the best model checkpoint was saved).

    Parameters
    ----------
    train_losses:
        List of mean training loss values, one per epoch.
    val_losses:
        List of mean validation loss values, one per epoch.
    best_epoch:
        0-indexed epoch of the best validation loss (for the dotted line).
    filename:
        Output PNG filename.
    """
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
    """
    Plot per-epoch training and validation accuracy curves.

    Parameters
    ----------
    train_accs, val_accs:
        Per-epoch accuracy lists.
    best_epoch:
        0-indexed epoch of the best validation loss.
    filename:
        Output PNG filename.
    """
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


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_neural_network(verbose: bool = True) -> dict:
    """
    Full training pipeline for the MLP neural network model.

    Steps
    -----
    1. Build features and labels.
    2. Standard-scale features (zero mean, unit variance).  Unlike tree
       models, neural networks are sensitive to feature scale because
       gradient updates are proportional to weight magnitudes.
    3. Split 80 % train / 20 % test; split train further 90/10 for
       validation (early stopping target).
    4. Compute class weights for balanced cross-entropy loss.
    5. Train with AdamW + CosineAnnealingLR + early stopping.
    6. Restore best checkpoint; evaluate on test set.
    7. Save loss curves, accuracy curves, confusion matrix.

    Parameters
    ----------
    verbose:
        If True, print per-epoch progress and results.

    Returns
    -------
    dict of evaluation metrics.
    """
    if not _TORCH_AVAILABLE:
        print("PyTorch not available. Skipping.")
        return {}

    print("\n" + "=" * 65)
    print("  NEURAL NETWORK (MLP) - Alzheimer's Disease Pathway Prediction")
    print("=" * 65)

    # ── Reproducibility ────────────────────────────────────────────────────
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"\nUsing device: {device}")

    # ── Feature engineering ────────────────────────────────────────────────
    X, y, feature_names, le = build_features(verbose=verbose)
    n_features = X.shape[1]
    n_classes = len(le.classes_)

    # ── Feature scaling ────────────────────────────────────────────────────
    # Standard scaling (z-score normalisation) ensures that gradient updates
    # are of comparable magnitude across all input dimensions.  We fit the
    # scaler on training data ONLY to avoid data leakage into the test set.
    X_train_raw, X_test_raw, y_train, y_test = stratified_split(
        X, y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 90/10 split of train for early stopping validation set
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

    # ── Build PyTorch DataLoaders ──────────────────────────────────────────
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

    # ── Model, loss, optimiser, scheduler ─────────────────────────────────
    model = PathwayMLP(n_features, n_classes).to(device)

    class_weights = _compute_class_weights(y_tr, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # CosineAnnealingLR smoothly decays LR from LEARNING_RATE to eta_min
    # over T_max epochs.  This helps both exploration early on and fine
    # convergence towards the end of training.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel parameters    : {n_params:,}")
        print(f"\nArchitecture:")
        print(model)
        print()

    # ── Training loop ──────────────────────────────────────────────────────
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
        # -- Training phase --
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            # Gradient clipping: prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            ep_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            ep_correct += (preds == y_batch).sum().item()
            ep_total += len(y_batch)

        train_losses.append(ep_loss / ep_total)
        train_accs.append(ep_correct / ep_total)

        # -- Validation phase --
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

        # -- Early stopping --
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

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Evaluate on test set ───────────────────────────────────────────────
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

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n[3/3] Saving visualisations …")
    _plot_loss_curves(train_losses, val_losses, best_epoch)
    _plot_accuracy_curve(train_accs, val_accs, best_epoch)
    save_confusion_matrix(
        y_test, y_pred_t, le,
        filename="nn_confusion_matrix.png",
        title="Neural Network (MLP) — Confusion Matrix (test set, row-normalised)",
    )

    # ── Persist metrics ────────────────────────────────────────────────────
    save_results_row(metrics)

    print("\n[OK] Neural Network training complete.")
    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_neural_network(verbose=True)
