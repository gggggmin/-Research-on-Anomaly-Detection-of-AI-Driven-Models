from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import LABEL_CANDIDATES, NORMAL_LABELS, DataConfig, RANDOM_STATE


@dataclass
class DatasetBundle:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler | MinMaxScaler


def load_dataset(config: DataConfig) -> DatasetBundle:
    if config.dataset == "fall":
        return load_fall_dataset(config)
    if config.dataset == "cicids":
        return load_cicids_dataset(config)
    if config.dataset == "synthetic":
        return synthetic_dataset(config)
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def read_data_files(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {path}")
    files: list[Path]
    if path.is_file():
        files = [path]
    else:
        files = []
        for pattern in ("*.csv", "*.xlsx", "*.xls", "*.parquet"):
            files.extend(path.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No supported data files found under: {path}")

    frames = []
    for file in sorted(files):
        suffix = file.suffix.lower()
        if suffix == ".csv":
            frames.append(pd.read_csv(file))
        elif suffix in {".xlsx", ".xls"}:
            frames.append(pd.read_excel(file))
        elif suffix == ".parquet":
            frames.append(pd.read_parquet(file))
    return pd.concat(frames, ignore_index=True)


def find_label_column(df: pd.DataFrame) -> str:
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        lowered = str(col).lower()
        if "label" in lowered or "class" in lowered or "activity" in lowered:
            return col
    raise ValueError("Could not infer label column. Rename it to one of: " + ", ".join(LABEL_CANDIDATES))


def labels_to_binary(values: pd.Series) -> np.ndarray:
    text = values.astype(str).str.strip().str.lower()
    return (~text.isin(NORMAL_LABELS)).astype(int).to_numpy()


def numeric_frame(df: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    out = df.drop(columns=list(exclude), errors="ignore").replace([np.inf, -np.inf], np.nan)
    out = out.select_dtypes(include=[np.number])
    if out.empty:
        raise ValueError("No numeric features found after excluding labels.")
    return out.fillna(out.median(numeric_only=True)).fillna(0)


def infer_xyz_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    numeric_cols = list(numeric_frame(df, {label_col}).columns)
    lowered = {str(c).lower(): c for c in numeric_cols}
    choices = []
    for key in ("x", "acc_x", "accx", "accelerometer_x"):
        if key in lowered:
            choices.append(lowered[key])
            break
    for key in ("y", "acc_y", "accy", "accelerometer_y"):
        if key in lowered:
            choices.append(lowered[key])
            break
    for key in ("z", "acc_z", "accz", "accelerometer_z"):
        if key in lowered:
            choices.append(lowered[key])
            break
    if len(choices) == 3:
        return choices
    if len(numeric_cols) < 3:
        raise ValueError("Fall dataset requires at least three numeric acceleration columns.")
    return numeric_cols[:3]


def make_fall_windows(xyz: np.ndarray, labels: np.ndarray, window_size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    windows = []
    y = []
    for start in range(0, len(xyz) - window_size + 1, stride):
        segment = xyz[start : start + window_size].astype(np.float32)
        seg_y = labels[start : start + window_size]
        total = np.sqrt(np.sum(segment**2, axis=1, keepdims=True))
        tilt = np.arctan2(segment[:, 1:2], np.sqrt(segment[:, 0:1] ** 2 + segment[:, 2:3] ** 2) + 1e-8)
        means = np.repeat(segment.mean(axis=0, keepdims=True), window_size, axis=0)
        stds = np.repeat(segment.std(axis=0, keepdims=True), window_size, axis=0)
        features = np.concatenate([segment, total, tilt, means, stds], axis=1)
        windows.append(features)
        y.append(int(seg_y.mean() >= 0.5))
    if not windows:
        raise ValueError("Not enough fall rows to form one sliding window.")
    return np.stack(windows), np.asarray(y, dtype=np.int64)


def load_fall_dataset(config: DataConfig) -> DatasetBundle:
    if config.data_path is None:
        raise ValueError("Fall dataset requires --data-path.")
    df = read_data_files(config.data_path)
    label_col = find_label_column(df)
    labels = labels_to_binary(df[label_col])
    xyz_cols = infer_xyz_columns(df, label_col)
    xyz = numeric_frame(df[xyz_cols], set()).to_numpy(dtype=np.float32)
    x, y = make_fall_windows(xyz, labels, config.window_size, config.stride)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.test_size, stratify=y, random_state=config.random_state
    )
    scaler = StandardScaler()
    x_train_2d = x_train.reshape(-1, x_train.shape[-1])
    scaler.fit(x_train_2d)
    x_train = scaler.transform(x_train_2d).reshape(x_train.shape)
    x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
    feature_names = ["x", "y", "z", "acc_total", "tilt", "mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]
    return DatasetBundle(x_train, x_test, y_train, y_test, feature_names, scaler)


def remove_correlated_features(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=drop_cols)


def select_network_features(x: pd.DataFrame, y: np.ndarray, mode: str) -> pd.DataFrame:
    if x.shape[1] <= 1:
        return x
    forest = RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced")
    forest.fit(x, y)
    importances = pd.Series(forest.feature_importances_, index=x.columns).sort_values(ascending=False)
    if mode == "top7":
        selected = list(importances.head(min(7, len(importances))).index)
    else:
        cumulative = importances.cumsum()
        selected = list(cumulative[cumulative <= 0.95].index)
        if not selected:
            selected = [importances.index[0]]
        if cumulative.iloc[len(selected) - 1] < 0.95 and len(selected) < len(importances):
            selected.append(importances.index[len(selected)])
    return x[selected]


def load_cicids_dataset(config: DataConfig) -> DatasetBundle:
    if config.data_path is None:
        raise ValueError("CICIDS dataset requires --data-path.")
    df = read_data_files(config.data_path)
    label_col = find_label_column(df)
    y = labels_to_binary(df[label_col])
    x = numeric_frame(df, {label_col})
    x = remove_correlated_features(x, threshold=0.9)
    x = select_network_features(x, y, config.feature_mode)

    x_train, x_test, y_train, y_test = train_test_split(
        x.to_numpy(dtype=np.float32), y, test_size=config.test_size, stratify=y, random_state=config.random_state
    )
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    x_train = x_train[:, :, None]
    x_test = x_test[:, :, None]
    return DatasetBundle(x_train, x_test, y_train, y_test, list(x.columns), scaler)


def synthetic_dataset(config: DataConfig) -> DatasetBundle:
    rng = np.random.default_rng(config.random_state)
    n = 420
    seq_len = 24
    dims = 6
    x = rng.normal(0, 1, size=(n, seq_len, dims)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    anomaly_idx = rng.choice(n, size=70, replace=False)
    y[anomaly_idx] = 1
    x[anomaly_idx, 8:14, :2] += rng.normal(4.0, 0.7, size=(len(anomaly_idx), 6, 2)).astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.test_size, stratify=y, random_state=config.random_state
    )
    scaler = StandardScaler()
    scaler.fit(x_train.reshape(-1, dims))
    x_train = scaler.transform(x_train.reshape(-1, dims)).reshape(x_train.shape)
    x_test = scaler.transform(x_test.reshape(-1, dims)).reshape(x_test.shape)
    return DatasetBundle(x_train, x_test, y_train, y_test, [f"f{i}" for i in range(dims)], scaler)
