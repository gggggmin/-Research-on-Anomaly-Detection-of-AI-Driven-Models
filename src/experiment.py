from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .baselines import iforest_scores, one_class_svm_scores, zscore_scores
from .config import dataset_defaults
from .data import load_dataset
from .evaluation import compute_metrics, plot_confusion_matrix, plot_losses, plot_pr_curve, save_json, save_predictions
from .models import AutoencoderIsolationForest, HybridAnomalyDetector
from .train import choose_config


def format_metrics(metrics: dict[str, float]) -> dict[str, float]:
    keys = ("accuracy", "precision", "recall", "f1", "auc")
    return {key: float(metrics.get(key, 0.0)) for key in keys}


def score_and_record(
    run_dir: Path,
    name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    losses: list[float] | None = None,
) -> dict[str, float]:
    metrics = compute_metrics(y_true, scores)
    method_dir = run_dir / name
    method_dir.mkdir(parents=True, exist_ok=True)
    save_json(method_dir / "metrics.json", {"method": name, "metrics": metrics})
    save_predictions(method_dir / "predictions.csv", y_true, scores, metrics["threshold"])
    plot_pr_curve(method_dir / "pr_curve.png", y_true, scores)
    plot_confusion_matrix(method_dir / "confusion_matrix.png", y_true, scores, metrics["threshold"])
    if losses:
        plot_losses(method_dir / "train_loss.png", losses)
    return metrics


def score_with_threshold_and_record(
    run_dir: Path,
    name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    metrics: dict[str, float],
    losses: list[float] | None = None,
) -> dict[str, float]:
    method_dir = run_dir / name
    method_dir.mkdir(parents=True, exist_ok=True)
    save_json(method_dir / "metrics.json", {"method": name, "metrics": metrics})
    save_predictions(method_dir / "predictions.csv", y_true, scores, threshold)
    plot_pr_curve(method_dir / "pr_curve.png", y_true, scores)
    plot_confusion_matrix(method_dir / "confusion_matrix.png", y_true, scores, threshold)
    if losses:
        plot_losses(method_dir / "train_loss.png", losses)
    return metrics


def record_with_validation_threshold(
    run_dir: Path,
    name: str,
    y_val: np.ndarray,
    val_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    losses: list[float] | None = None,
) -> dict[str, float]:
    threshold = compute_metrics(y_val, val_scores)["threshold"]
    metrics = compute_metrics(y_test, test_scores, threshold)
    return score_with_threshold_and_record(run_dir, name, y_test, test_scores, threshold, metrics, losses)


def write_summary(run_dir: Path, dataset: str, results: dict[str, dict[str, float]], note: str) -> None:
    csv_path = run_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "accuracy", "precision", "recall", "f1", "auc"])
        for method, metrics in results.items():
            row = [method] + [f"{metrics[key]:.6f}" for key in ("accuracy", "precision", "recall", "f1", "auc")]
            writer.writerow(row)

    lines = [
        f"# Experiment Report: {dataset}",
        "",
        note,
        "",
        "| Method | Accuracy | Precision | Recall | F1 | AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in results.items():
        lines.append(
            "| "
            + method
            + " | "
            + " | ".join(f"{metrics[key]:.4f}" for key in ("accuracy", "precision", "recall", "f1", "auc"))
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These values are produced by the current local run. If the dataset is synthetic, they only verify the experiment pipeline and cannot be cited as real paper evidence.",
            "For thesis-grade evidence, rerun the same command with the real UCI Localization and CIC-IDS2017 files under `data/raw/`.",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_full_experiment(
    dataset: str,
    data_path: str | None = None,
    output_dir: str = "outputs",
    feature_mode: str = "importance95",
    fast: bool = False,
) -> Path:
    data_config = dataset_defaults(dataset, data_path)
    data_config.feature_mode = feature_mode
    bundle = load_dataset(data_config)
    model_config = choose_config(dataset, bundle.x_train.shape[-1], fast=fast)
    x_dev, x_val, y_dev, y_val = train_test_split(
        bundle.x_train,
        bundle.y_train,
        test_size=0.2,
        stratify=bundle.y_train,
        random_state=42,
    )
    run_dir = Path(output_dir) / f"experiment_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fit_x = x_dev[y_dev == 0]
    if len(fit_x) == 0:
        fit_x = x_dev

    results: dict[str, dict[str, float]] = {}

    hybrid = HybridAnomalyDetector(bundle.x_train.shape[-1], model_config)
    hybrid_losses = hybrid.fit(fit_x)
    val_scores = hybrid.anomaly_scores(x_val)
    test_scores = hybrid.anomaly_scores(bundle.x_test)
    results["full_transformer_iforest"] = format_metrics(
        record_with_validation_threshold(
            run_dir,
            "full_transformer_iforest",
            y_val,
            val_scores,
            bundle.y_test,
            test_scores,
            hybrid_losses,
        )
    )

    supervised = HybridAnomalyDetector(bundle.x_train.shape[-1], model_config)
    supervised_losses = supervised.fit_supervised(x_dev, y_dev)
    sup_val_scores = supervised.fused_scores(x_val, alpha=0.65)
    sup_test_scores = supervised.fused_scores(bundle.x_test, alpha=0.65)
    results["supervised_transformer_iforest_fusion"] = format_metrics(
        record_with_validation_threshold(
            run_dir,
            "supervised_transformer_iforest_fusion",
            y_val,
            sup_val_scores,
            bundle.y_test,
            sup_test_scores,
            supervised_losses,
        )
    )

    z_val = zscore_scores(fit_x, x_val)
    z_test = zscore_scores(fit_x, bundle.x_test)
    results["zscore"] = format_metrics(
        record_with_validation_threshold(run_dir, "zscore", y_val, z_val, bundle.y_test, z_test)
    )
    if_val = iforest_scores(fit_x, x_val)
    if_test = iforest_scores(fit_x, bundle.x_test)
    results["iforest_without_transformer"] = format_metrics(
        record_with_validation_threshold(run_dir, "iforest_without_transformer", y_val, if_val, bundle.y_test, if_test)
    )
    svm_val = one_class_svm_scores(fit_x, x_val)
    svm_test = one_class_svm_scores(fit_x, bundle.x_test)
    results["one_class_svm"] = format_metrics(
        record_with_validation_threshold(run_dir, "one_class_svm", y_val, svm_val, bundle.y_test, svm_test)
    )
    recon_val = hybrid.reconstruction_scores(x_val)
    recon_test = hybrid.reconstruction_scores(bundle.x_test)
    results["transformer_without_iforest"] = format_metrics(
        record_with_validation_threshold(run_dir, "transformer_without_iforest", y_val, recon_val, bundle.y_test, recon_test)
    )

    ae_iforest = AutoencoderIsolationForest(bundle.x_train.shape[1:], model_config)
    ae_losses = ae_iforest.fit(fit_x)
    ae_val_scores = ae_iforest.anomaly_scores(x_val)
    ae_test_scores = ae_iforest.anomaly_scores(bundle.x_test)
    results["autoencoder_iforest"] = format_metrics(
        record_with_validation_threshold(run_dir, "autoencoder_iforest", y_val, ae_val_scores, bundle.y_test, ae_test_scores, ae_losses)
    )

    note = (
        "Dataset source: synthetic smoke data."
        if dataset == "synthetic"
        else f"Dataset source: `{data_path}`."
    )
    write_summary(run_dir, dataset, results, note)
    save_json(
        run_dir / "experiment_config.json",
        {
            "dataset": dataset,
            "data_path": data_path,
            "feature_mode": feature_mode,
            "fast": fast,
            "x_train_shape": list(bundle.x_train.shape),
            "x_fit_normal_shape": list(fit_x.shape),
            "x_test_shape": list(bundle.x_test.shape),
            "feature_names": bundle.feature_names,
            "results": results,
        },
    )
    print(f"Experiment directory: {run_dir}")
    print((run_dir / "summary.csv").read_text(encoding="utf-8"))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete thesis experiment suite.")
    parser.add_argument("--dataset", choices=["fall", "cicids", "synthetic"], required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--feature-mode", choices=["importance95", "top7"], default="importance95")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    run_full_experiment(args.dataset, args.data_path, args.output_dir, args.feature_mode, args.fast)


if __name__ == "__main__":
    main()
