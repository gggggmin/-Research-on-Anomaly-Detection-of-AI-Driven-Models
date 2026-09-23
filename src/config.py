from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


RANDOM_STATE = 42
NORMAL_LABELS = {"0", "benign", "normal", "false", "negative", "nonfall", "non-fall", "walk", "standing"}
LABEL_CANDIDATES = ("label", "Label", "class", "Class", "target", "Target", "activity", "Activity")


@dataclass
class ModelConfig:
    dataset: str
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    epochs: int = 15
    patience: int = 5
    batch_size: int = 128
    learning_rate: float = 1e-3
    iforest_estimators: int = 200
    iforest_max_samples: float = 0.8
    contamination: float = 0.1
    random_state: int = RANDOM_STATE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataConfig:
    dataset: str
    data_path: Path | None = None
    window_size: int = 50
    stride: int = 25
    test_size: float = 0.2
    feature_mode: str = "importance95"
    random_state: int = RANDOM_STATE


def dataset_defaults(dataset: str, data_path: str | None = None) -> DataConfig:
    dataset = dataset.lower()
    if dataset == "fall":
        return DataConfig(dataset=dataset, data_path=Path(data_path) if data_path else None, test_size=0.2)
    if dataset == "cicids":
        return DataConfig(dataset=dataset, data_path=Path(data_path) if data_path else None, test_size=0.3)
    if dataset == "synthetic":
        return DataConfig(dataset=dataset, data_path=None, test_size=0.25)
    raise ValueError(f"Unsupported dataset: {dataset}")
