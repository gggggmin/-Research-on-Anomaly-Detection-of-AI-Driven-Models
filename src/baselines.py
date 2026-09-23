from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from .evaluation import compute_metrics


def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(len(x), -1)


def zscore_scores(x_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    train = flatten(x_train)
    test = flatten(x_test)
    mu = train.mean(axis=0)
    sigma = train.std(axis=0) + 1e-8
    return np.abs((test - mu) / sigma).max(axis=1)


def iforest_scores(x_train: np.ndarray, x_test: np.ndarray, random_state: int = 42) -> np.ndarray:
    model = IsolationForest(n_estimators=200, contamination=0.1, random_state=random_state, n_jobs=1)
    model.fit(flatten(x_train))
    return -model.score_samples(flatten(x_test))


def one_class_svm_scores(x_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    train = flatten(x_train)
    test = flatten(x_test)
    if len(train) > 5000:
        train = train[:5000]
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    model.fit(train)
    return -model.score_samples(test)


def run_baselines(x_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, dict[str, float]]:
    scores = {
        "zscore": zscore_scores(x_train, x_test),
        "isolation_forest_raw": iforest_scores(x_train, x_test),
        "one_class_svm": one_class_svm_scores(x_train, x_test),
    }
    return {name: compute_metrics(y_test, value) for name, value in scores.items()}
