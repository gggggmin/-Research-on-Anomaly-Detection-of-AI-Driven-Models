from __future__ import annotations

from pathlib import Path

from .train import train_pipeline


def main() -> None:
    run_dir = train_pipeline("synthetic", None, output_dir="outputs", fast=True)
    required = ["metrics.json", "predictions.csv", "pr_curve.png", "confusion_matrix.png", "train_loss.png"]
    missing = [name for name in required if not (Path(run_dir) / name).exists()]
    if missing:
        raise RuntimeError("Smoke test missing outputs: " + ", ".join(missing))
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
