from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest

from .config import ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 2048) -> None:
        super().__init__()
        positions = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-np.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads.")
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.position = PositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        z = self.position(z)
        return self.encoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_proj(self.encode_sequence(x))

    def pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_sequence(x).mean(dim=1)


class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(max(8, hidden_dim // 2), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        return self.decoder(self.encoder(flat))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        return self.encoder(flat)


class HybridAnomalyDetector:
    def __init__(self, input_dim: int, config: ModelConfig, device: str | None = None) -> None:
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.transformer = TransformerAutoencoder(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        ).to(self.device)
        self.iforest = IsolationForest(
            n_estimators=config.iforest_estimators,
            max_samples=config.iforest_max_samples,
            contamination=config.contamination,
            random_state=config.random_state,
            n_jobs=1,
        )

    def fit_transformer(self, x_train: np.ndarray) -> list[float]:
        torch.manual_seed(self.config.random_state)
        data = torch.tensor(x_train, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(data),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.config.random_state),
        )
        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()
        best_loss = float("inf")
        wait = 0
        losses: list[float] = []
        best_state = None

        for _ in range(self.config.epochs):
            self.transformer.train()
            total = 0.0
            count = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                recon = self.transformer(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu()) * len(batch)
                count += len(batch)
            epoch_loss = total / max(count, 1)
            losses.append(epoch_loss)
            if epoch_loss < best_loss - 1e-5:
                best_loss = epoch_loss
                wait = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.transformer.state_dict().items()}
            else:
                wait += 1
                if wait >= self.config.patience:
                    break

        if best_state is not None:
            self.transformer.load_state_dict(best_state)
        return losses

    def extract_features(self, x: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        self.transformer.eval()
        batch_size = batch_size or self.config.batch_size
        features = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                batch = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=self.device)
                features.append(self.transformer.pooled_features(batch).cpu().numpy())
        return np.concatenate(features, axis=0)

    def fit(self, x_train: np.ndarray) -> list[float]:
        losses = self.fit_transformer(x_train)
        features = self.extract_features(x_train)
        self.iforest.fit(features)
        return losses

    def anomaly_scores(self, x: np.ndarray) -> np.ndarray:
        features = self.extract_features(x)
        return -self.iforest.score_samples(features)

    def reconstruction_scores(self, x: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        self.transformer.eval()
        batch_size = batch_size or self.config.batch_size
        scores = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                batch = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=self.device)
                recon = self.transformer(batch)
                err = torch.mean((recon - batch) ** 2, dim=tuple(range(1, recon.ndim)))
                scores.append(err.cpu().numpy())
        return np.concatenate(scores, axis=0)


class AutoencoderIsolationForest:
    def __init__(self, input_shape: tuple[int, ...], config: ModelConfig, device: str | None = None) -> None:
        self.config = config
        self.input_shape = input_shape
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.autoencoder = DenseAutoencoder(int(np.prod(input_shape)), hidden_dim=min(128, max(32, int(np.prod(input_shape)) // 2))).to(self.device)
        self.iforest = IsolationForest(
            n_estimators=config.iforest_estimators,
            max_samples=config.iforest_max_samples,
            contamination=config.contamination,
            random_state=config.random_state,
            n_jobs=1,
        )

    def fit_autoencoder(self, x_train: np.ndarray) -> list[float]:
        torch.manual_seed(self.config.random_state)
        data = torch.tensor(x_train, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(data),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.config.random_state),
        )
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()
        losses: list[float] = []
        for _ in range(self.config.epochs):
            total = 0.0
            count = 0
            self.autoencoder.train()
            for (batch,) in loader:
                batch = batch.to(self.device)
                flat = batch.reshape(batch.shape[0], -1)
                optimizer.zero_grad(set_to_none=True)
                recon = self.autoencoder(batch)
                loss = loss_fn(recon, flat)
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu()) * len(batch)
                count += len(batch)
            losses.append(total / max(count, 1))
        return losses

    def extract_features(self, x: np.ndarray) -> np.ndarray:
        self.autoencoder.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(x), self.config.batch_size):
                batch = torch.tensor(x[start : start + self.config.batch_size], dtype=torch.float32, device=self.device)
                out.append(self.autoencoder.features(batch).cpu().numpy())
        return np.concatenate(out, axis=0)

    def fit(self, x_train: np.ndarray) -> list[float]:
        losses = self.fit_autoencoder(x_train)
        self.iforest.fit(self.extract_features(x_train))
        return losses

    def anomaly_scores(self, x: np.ndarray) -> np.ndarray:
        return -self.iforest.score_samples(self.extract_features(x))
