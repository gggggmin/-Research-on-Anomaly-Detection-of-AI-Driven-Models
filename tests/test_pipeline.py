from __future__ import annotations

import unittest

import numpy as np

from src.config import dataset_defaults
from src.data import load_dataset, make_fall_windows
from src.evaluation import best_threshold_from_pr, compute_metrics
from src.models import HybridAnomalyDetector
from src.train import choose_config
from src.experiment import run_full_experiment


class PipelineTests(unittest.TestCase):
    def test_fall_window_shape(self) -> None:
        xyz = np.random.default_rng(42).normal(size=(80, 3)).astype(np.float32)
        labels = np.zeros(80, dtype=int)
        labels[40:60] = 1
        x, y = make_fall_windows(xyz, labels, window_size=20, stride=10)
        self.assertEqual(x.shape[1:], (20, 11))
        self.assertEqual(len(x), len(y))

    def test_synthetic_loader(self) -> None:
        bundle = load_dataset(dataset_defaults("synthetic"))
        self.assertEqual(bundle.x_train.ndim, 3)
        self.assertEqual(bundle.x_train.shape[-1], len(bundle.feature_names))
        self.assertGreater(bundle.y_train.sum(), 0)

    def test_metrics(self) -> None:
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        threshold, _ = best_threshold_from_pr(y, scores)
        metrics = compute_metrics(y, scores, threshold)
        self.assertEqual(metrics["f1"], 1.0)

    def test_model_feature_shape(self) -> None:
        bundle = load_dataset(dataset_defaults("synthetic"))
        cfg = choose_config("synthetic", bundle.x_train.shape[-1], fast=True)
        cfg.epochs = 1
        detector = HybridAnomalyDetector(bundle.x_train.shape[-1], cfg, device="cpu")
        detector.fit(bundle.x_train[:64])
        features = detector.extract_features(bundle.x_test[:10])
        self.assertEqual(features.shape, (10, cfg.hidden_dim))

    def test_full_experiment_smoke(self) -> None:
        run_dir = run_full_experiment("synthetic", output_dir="outputs/test_runs", fast=True)
        self.assertTrue((run_dir / "summary.csv").exists())
        self.assertTrue((run_dir / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
