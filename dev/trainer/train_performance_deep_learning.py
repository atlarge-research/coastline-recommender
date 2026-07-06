#!/usr/bin/env python3
"""Deep learning predictor: embeddings + residual blocks, Huber loss. Target MdAPE <20%."""

import logging
import time
import warnings
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# The net architecture is the shipped package's — one definition, shared with inference.
from coastline.sdk.predictors.performance.data_driven._nn import EmbeddingNN

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)

from .common import (  # noqa: E402
    PORTFOLIO_DIR,
    SEED,
    as_dataframes,
    calculate_metrics,
    encode_categorical_features,
    inverse_transform_targets,
    load_and_preprocess_data,
    performance_deep_learning_model_dir,
    print_metrics,
    save_deep_learning_bundle_if_better,
    scale_numerical_features,
    split_data,
    transform_targets,
)

HIDDEN_DIMS_OPTIONS = [
    [256, 256, 128],
    [512, 256, 128],
    [512, 512, 256],
    [256, 256, 256, 128],
]
EMBEDDING_DIM_OPTIONS = [48, 64]
DROPOUT_OPTIONS = [0.15, 0.25, 0.35]
LEARNING_RATE_OPTIONS = [0.001, 0.0005]
BATCH_SIZE_OPTIONS = [32, 64]

NUM_EPOCHS = 350
EARLY_STOPPING_PATIENCE = 40
WEIGHT_DECAY = 5e-4
GRAD_CLIP_NORM = 1.0

# Gaussian noise std for input regularisation during training
NOISE_STD = 0.05

# Auto-detect best available device - prioritize MPS for Apple Silicon
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🚀 Using Apple Silicon GPU (MPS) for acceleration")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("🚀 Using NVIDIA GPU (CUDA) for acceleration")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  Using CPU (no GPU acceleration)")


class GPUPerformanceDataset(Dataset):
    """PyTorch Dataset for GPU performance data."""

    def __init__(self, X_cat, X_num, y):
        self.X_cat = torch.LongTensor(X_cat)
        self.X_num = torch.FloatTensor(X_num)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip_norm=1.0):
    """Train for one epoch with gradient clipping."""
    model.train()
    total_loss = 0
    for x_cat, x_num, y in dataloader:
        x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x_cat, x_num)
        loss = criterion(y_pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    predictions, targets = [], []
    with torch.no_grad():
        for x_cat, x_num, y in dataloader:
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
            y_pred = model(x_cat, x_num)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            predictions.append(y_pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    return total_loss / len(dataloader), np.vstack(predictions), np.vstack(targets)


def train_single_config(
    X_cat_train_enc,
    X_num_train_scaled,
    y_log_train,
    X_cat_val_enc,
    X_num_val_scaled,
    y_log_val,
    cat_features,
    num_features,
    vocab_sizes,
    hidden_dims,
    embedding_dim,
    dropout_rate,
    learning_rate,
    batch_size,
):
    """Train a single model configuration and return validation loss."""
    # Per-config seed so grid-search results are reproducible.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_dataset = GPUPerformanceDataset(X_cat_train_enc, X_num_train_scaled, y_log_train)
    val_dataset = GPUPerformanceDataset(X_cat_val_enc, X_num_val_scaled, y_log_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=len(train_dataset) % batch_size == 1
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    embedding_dims_with_vocab = {name: (vocab_sizes[name], embedding_dim) for name in cat_features}

    model = EmbeddingNN(
        embedding_dims=embedding_dims_with_vocab,
        num_numerical_features=len(num_features),
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        noise_std=NOISE_STD,
    ).to(DEVICE)

    # Huber loss is more robust to outliers than L1 or MSE.
    criterion = nn.HuberLoss(delta=0.5)
    optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        train_epoch(model, train_loader, criterion, optimizer, DEVICE, grad_clip_norm=GRAD_CLIP_NORM)
        val_loss, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            break

    return best_val_loss, best_model_state, model


def train_deep_learning_model():
    """Main training pipeline for Deep Learning model with grid search."""
    print("=" * 70)
    print("DEEP LEARNING PREDICTOR TRAINING WITH GRID SEARCH")
    print("=" * 70)

    X_cat, X_num, y, cat_features, num_features = load_and_preprocess_data()
    y_log = transform_targets(y)

    (
        (X_cat_train, X_num_train, y_train, y_log_train),
        (X_cat_val, X_num_val, y_val, y_log_val),
        (X_cat_test, X_num_test, y_test, y_log_test),
    ) = split_data(X_cat, X_num, y, y_log)

    X_cat_train, X_num_train, y_train, y_log_train = as_dataframes(X_cat_train, X_num_train, y_train, y_log_train)
    X_cat_val, X_num_val, y_val, y_log_val = as_dataframes(X_cat_val, X_num_val, y_val, y_log_val)
    X_cat_test, X_num_test, y_test, y_log_test = as_dataframes(X_cat_test, X_num_test, y_test, y_log_test)

    # return_numpy=True: PyTorch embedding layers need integer-coded numpy arrays.
    X_cat_train_enc, X_cat_val_enc, X_cat_test_enc, encoders, vocab_sizes = encode_categorical_features(
        X_cat_train, X_cat_val, X_cat_test, return_numpy=True
    )

    X_num_train_scaled, X_num_val_scaled, X_num_test_scaled, scaler = scale_numerical_features(
        X_num_train, X_num_val, X_num_test
    )

    y_train = y_train.to_numpy(dtype=float)
    y_val = y_val.to_numpy(dtype=float)
    y_test = y_test.to_numpy(dtype=float)
    y_log_train = y_log_train.to_numpy(dtype=float)
    y_log_val = y_log_val.to_numpy(dtype=float)
    y_log_test = y_log_test.to_numpy(dtype=float)

    print(f"\n🔍 Starting grid search on {DEVICE}...")
    print(f"  Hidden dims options: {len(HIDDEN_DIMS_OPTIONS)}")
    print(f"  Embedding dim options: {len(EMBEDDING_DIM_OPTIONS)}")
    print(f"  Dropout options: {len(DROPOUT_OPTIONS)}")
    print(f"  Learning rate options: {len(LEARNING_RATE_OPTIONS)}")
    print(f"  Batch size options: {len(BATCH_SIZE_OPTIONS)}")

    total_configs = (
        len(HIDDEN_DIMS_OPTIONS)
        * len(EMBEDDING_DIM_OPTIONS)
        * len(DROPOUT_OPTIONS)
        * len(LEARNING_RATE_OPTIONS)
        * len(BATCH_SIZE_OPTIONS)
    )
    print(f"  Total configurations: {total_configs}")
    print("  This will take approximately 20-40 minutes...\n")

    best_val_loss = float("inf")
    best_config = None
    best_model_state = None
    best_model_architecture = None

    config_num = 0
    start_time = time.time()

    for hidden_dims in HIDDEN_DIMS_OPTIONS:
        for embedding_dim in EMBEDDING_DIM_OPTIONS:
            for dropout_rate in DROPOUT_OPTIONS:
                for learning_rate in LEARNING_RATE_OPTIONS:
                    for batch_size in BATCH_SIZE_OPTIONS:
                        config_num += 1
                        print(
                            f"[{config_num}/{total_configs}] Testing: hidden={hidden_dims}, emb={embedding_dim}, "
                            f"dropout={dropout_rate}, lr={learning_rate}, batch={batch_size}"
                        )

                        val_loss, model_state, model = train_single_config(
                            X_cat_train_enc,
                            X_num_train_scaled.values,
                            y_log_train,
                            X_cat_val_enc,
                            X_num_val_scaled.values,
                            y_log_val,
                            cat_features,
                            num_features,
                            vocab_sizes,
                            hidden_dims,
                            embedding_dim,
                            dropout_rate,
                            learning_rate,
                            batch_size,
                        )

                        print(f"  → Val Loss: {val_loss:.4f}")

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_config = {
                                "hidden_dims": hidden_dims,
                                "embedding_dim": embedding_dim,
                                "dropout_rate": dropout_rate,
                                "learning_rate": learning_rate,
                                "batch_size": batch_size,
                            }
                            best_model_state = model_state
                            best_model_architecture = model
                            print(f"  ✨ New best configuration! Val Loss: {val_loss:.4f}")

    elapsed_time = time.time() - start_time
    print(f"\n✅ Grid search complete in {elapsed_time / 60:.1f} minutes!")

    if best_config is None or best_model_state is None or best_model_architecture is None:
        raise RuntimeError("Grid search failed to find a valid configuration")

    print("\n🏆 Best configuration:")
    for param, value in best_config.items():
        print(f"  {param}: {value}")
    print(f"  Best validation loss: {best_val_loss:.4f}")

    best_model_architecture.load_state_dict(cast(dict[str, Any], best_model_state))
    model = best_model_architecture

    test_dataset = GPUPerformanceDataset(X_cat_test_enc, X_num_test_scaled.values, y_log_test)
    test_loader = DataLoader(test_dataset, batch_size=best_config["batch_size"], shuffle=False)

    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    criterion = nn.HuberLoss(delta=0.5)
    _, y_log_test_pred, y_log_test_true = evaluate(model, test_loader, criterion, DEVICE)
    y_test_pred = inverse_transform_targets(y_log_test_pred)
    y_test_true = inverse_transform_targets(y_log_test_true)

    test_metrics_throughput = calculate_metrics(
        y_test_true[:, 0],
        y_test_pred[:, 0],
        y_log_test_true[:, 0],
        y_log_test_pred[:, 0],
    )
    test_metrics_runtime = calculate_metrics(
        y_test_true[:, 1],
        y_test_pred[:, 1],
        y_log_test_true[:, 1],
        y_log_test_pred[:, 1],
    )
    print_metrics(test_metrics_throughput, "Test Throughput")
    print_metrics(test_metrics_runtime, "Test Runtime", unit="sec")

    print("\n🔍 Sample throughput predictions (first 10):")
    print(f"{'True':>12} {'Predicted':>12} {'Error %':>10}")
    print("-" * 36)
    for i in range(min(10, len(y_test_true))):
        error_pct = abs(y_test_true[i, 0] - y_test_pred[i, 0]) / y_test_true[i, 0] * 100
        print(f"{y_test_true[i, 0]:>12,.1f} {y_test_pred[i, 0]:>12,.1f} {error_pct:>9.1f}%")

    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = performance_deep_learning_model_dir()
    embedding_dims_with_vocab = {name: (vocab_sizes[name], best_config["embedding_dim"]) for name in cat_features}

    torch_payload = {
        "model_state_dict": model.state_dict(),
        "embedding_dims": embedding_dims_with_vocab,
        "num_numerical_features": len(num_features),
        "hidden_dims": best_config["hidden_dims"],
        "dropout_rate": best_config["dropout_rate"],
        "noise_std": NOISE_STD,
        "output_dim": 2,
        "best_config": best_config,
    }
    sklearn_artifacts = {
        "encoders": encoders,
        "scaler": scaler,
        "cat_features": cat_features,
        "num_features": num_features,
        "best_config": best_config,
        "test_metrics": test_metrics_throughput,
        "test_metrics_by_target": {
            "throughput": test_metrics_throughput,
            "runtime_seconds": test_metrics_runtime,
        },
    }
    new_mdape = float(test_metrics_throughput["original_space"]["mdape"])
    _, save_msg = save_deep_learning_bundle_if_better(
        model_dir,
        new_throughput_mdape=new_mdape,
        torch_save_dict=torch_payload,
        sklearn_artifacts=sklearn_artifacts,
    )

    print(f"💾 {save_msg}")
    print(f"  ✓ Test throughput MdAPE: {test_metrics_throughput['original_space']['mdape']:.2f}%")
    print(f"  ✓ Test runtime MdAPE: {test_metrics_runtime['original_space']['mdape']:.2f}%")
    print(f"  ✓ Test throughput R²: {test_metrics_throughput['original_space']['r2']:.4f}")
    print(f"  ✓ Test runtime R²: {test_metrics_runtime['original_space']['r2']:.4f}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    mdape = test_metrics_throughput["original_space"]["mdape"]
    target_threshold = 20.0

    if mdape < target_threshold:
        status = "✅ SUCCESS"
        emoji = "🎉"
    else:
        status = "⚠️  NEEDS IMPROVEMENT"
        emoji = "🔧"

    print(f"\n{emoji} {status}")
    print(f"  Target MdAPE: <{target_threshold}%")
    print(f"  Achieved MdAPE: {mdape:.2f}%")
    print(f"  Test throughput R²: {test_metrics_throughput['original_space']['r2']:.4f}")
    print(f"  Test runtime R²: {test_metrics_runtime['original_space']['r2']:.4f}")
    print(f"  Throughput within 20%: {test_metrics_throughput['original_space']['within_20_pct']:.1f}%")
    print(f"  Runtime within 20%: {test_metrics_runtime['original_space']['within_20_pct']:.1f}%")
    print(f"  Training time: {elapsed_time / 60:.1f} minutes")
    print("\n🚀 Deep Learning predictor ready for inference!")

    return model, encoders, scaler


if __name__ == "__main__":
    train_deep_learning_model()
