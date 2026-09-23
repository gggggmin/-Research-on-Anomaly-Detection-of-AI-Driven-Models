from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluation import compute_metrics, plot_confusion_matrix, plot_pr_curve, save_json


def evaluate_run(run_dir: str) -> dict[str, float]:
    path = Path(run_dir)
    pred_file = path / "predictions.csv"
    if not pred_file.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_file}")
    data = np.genfromtxt(pred_file, delimiter=",", names=True)
    y_true = np.asarray(data["y_true"], dtype=int)
    scores = np.asarray(data["score"], dtype=float)
    metrics_file = path / "metrics.json"
    threshold = None
    if metrics_file.exists():
        payload = json.loads(metrics_file.read_text(encoding="utf-8"))
        threshold = payload.get("metrics", {}).get("threshold")
    metrics = compute_metrics(y_true, scores, threshold)
    save_json(path / "metrics_recomputed.json", {"metrics": metrics})
    plot_pr_curve(path / "pr_curve_recomputed.png", y_true, scores)
    plot_confusion_matrix(path / "confusion_matrix_recomputed.png", y_true, scores, metrics["threshold"])
    print("Metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute metrics for a run directory.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    evaluate_run(args.run_dir)


if __name__ == "__main__":
    main()
