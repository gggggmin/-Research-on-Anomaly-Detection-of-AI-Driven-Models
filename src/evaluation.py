from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)


def best_threshold_from_pr(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.nanargmax(f1))
    if best_idx >= len(thresholds):
        threshold = float(thresholds[-1]) if len(thresholds) else float(np.median(scores))
    else:
        threshold = float(thresholds[best_idx])
    pr_auc = float(auc(recall, precision))
    return threshold, {"best_pr_f1": float(f1[best_idx]), "pr_auc": pr_auc}


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float | None = None) -> dict[str, float]:
    if threshold is None:
        threshold, extra = best_threshold_from_pr(y_true, scores)
    else:
        extra = {}
    y_pred = (scores >= threshold).astype(int)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0,
    }
    metrics.update(extra)
    return metrics


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_predictions(path: Path, y_true: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_pred = (scores >= threshold).astype(int)
    lines = ["y_true,score,y_pred"]
    lines.extend(f"{int(t)},{float(s):.10f},{int(p)}" for t, s, p in zip(y_true, scores, y_pred))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_pr_curve(path: Path, y_true: np.ndarray, scores: np.ndarray) -> None:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_confusion_matrix(path: Path, y_true: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["Normal", "Anomaly"])
    plt.yticks([0, 1], ["Normal", "Anomaly"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_losses(path: Path, losses: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Transformer Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_run_outputs(run_dir: Path, y_true: np.ndarray, scores: np.ndarray, losses: list[float], config: dict) -> dict[str, float]:
    threshold, _ = best_threshold_from_pr(y_true, scores)
    metrics = compute_metrics(y_true, scores, threshold)
    save_json(run_dir / "metrics.json", {"metrics": metrics, "config": config})
    save_predictions(run_dir / "predictions.csv", y_true, scores, threshold)
    plot_pr_curve(run_dir / "pr_curve.png", y_true, scores)
    plot_confusion_matrix(run_dir / "confusion_matrix.png", y_true, scores, threshold)
    plot_losses(run_dir / "train_loss.png", losses)
    return metrics
