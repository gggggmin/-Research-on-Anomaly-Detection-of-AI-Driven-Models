from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch

from .config import ModelConfig, dataset_defaults
from .data import load_dataset
from .evaluation import write_run_outputs
from .models import HybridAnomalyDetector


def choose_config(dataset: str, input_dim: int, fast: bool) -> ModelConfig:
    hidden_dim = 64 if fast else 128
    n_heads = 4
    if hidden_dim % n_heads != 0:
        n_heads = 1
    return ModelConfig(
        dataset=dataset,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=1 if fast else 2,
        dim_feedforward=128,
        epochs=4 if fast else 15,
        patience=2 if fast else 5,
        batch_size=64 if fast else 128,
        iforest_estimators=80 if fast else 200,
        contamination=0.1,
    )


def train_pipeline(
    dataset: str,
    data_path: str | None,
    output_dir: str = "outputs",
    feature_mode: str = "importance95",
    fast: bool = False,
) -> Path:
    np.random.seed(42)
    torch.manual_seed(42)

    data_config = dataset_defaults(dataset, data_path)
    data_config.feature_mode = feature_mode
    bundle = load_dataset(data_config)
    model_config = choose_config(dataset, input_dim=bundle.x_train.shape[-1], fast=fast)

    run_name = f"{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    detector = HybridAnomalyDetector(input_dim=bundle.x_train.shape[-1], config=model_config)
    losses = detector.fit(bundle.x_train)
    scores = detector.anomaly_scores(bundle.x_test)

    config = {
        "data": {
            "dataset": data_config.dataset,
            "data_path": str(data_config.data_path) if data_config.data_path else None,
            "window_size": data_config.window_size,
            "stride": data_config.stride,
            "test_size": data_config.test_size,
            "feature_mode": data_config.feature_mode,
            "feature_names": bundle.feature_names,
            "x_train_shape": list(bundle.x_train.shape),
            "x_test_shape": list(bundle.x_test.shape),
        },
        "model": model_config.to_dict(),
    }
    metrics = write_run_outputs(run_dir, bundle.y_test, scores, losses, config)

    torch.save(detector.transformer.state_dict(), run_dir / "model.pt")
    joblib.dump(detector.iforest, run_dir / "isolation_forest.joblib")
    joblib.dump(bundle.scaler, run_dir / "scaler.joblib")

    print(f"Run directory: {run_dir}")
    print(
        "Metrics: "
        + ", ".join(
            f"{key}={value:.4f}" for key, value in metrics.items() if key in {"accuracy", "precision", "recall", "f1", "auc"}
        )
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer + IsolationForest anomaly detector.")
    parser.add_argument("--dataset", choices=["fall", "cicids", "synthetic"], required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--feature-mode", choices=["importance95", "top7"], default="importance95")
    parser.add_argument("--fast", action="store_true", help="Use a small config for quick validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_pipeline(args.dataset, args.data_path, args.output_dir, args.feature_mode, args.fast)


if __name__ == "__main__":
    main()
