from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainResult:
    model: MLPRegressor
    scaler: StandardScaler
    history: list[dict[str, float]]


def _arrays(df: pd.DataFrame, feature_cols: list[str], fit_scaler: StandardScaler | None = None):
    x = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["target"].to_numpy(dtype=np.float32)
    scaler = fit_scaler or StandardScaler()
    x = scaler.fit_transform(x).astype(np.float32) if fit_scaler is None else scaler.transform(x).astype(np.float32)
    return x, y, scaler


def train_mlp(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    epochs: int = 10,
    batch_size: int = 4096,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    dropout: float = 0.20,
    seed: int = 2026,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, y_train, scaler = _arrays(train_df, feature_cols)
    x_val, y_val, _ = _arrays(val_df, feature_cols, scaler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(len(feature_cols), hidden_dim=hidden_dim, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    history: list[dict[str, float]] = []
    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_t), y_val_t).detach().cpu())
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss}
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model=model, scaler=scaler, history=history)


def predict(model: MLPRegressor, scaler: StandardScaler, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    device = next(model.parameters()).device
    x = scaler.transform(df[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), 65536):
            xb = torch.from_numpy(x[start : start + 65536]).to(device)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.float32)


def save_checkpoint(path: str, result: TrainResult, feature_cols: list[str]) -> None:
    torch.save(
        {
            "model_state": result.model.state_dict(),
            "input_dim": len(feature_cols),
            "feature_cols": feature_cols,
            "scaler_mean": result.scaler.mean_,
            "scaler_scale": result.scaler.scale_,
            "history": result.history,
        },
        path,
    )


def load_checkpoint(path: str) -> tuple[MLPRegressor, StandardScaler, list[str], list[dict[str, float]]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    feature_cols = list(checkpoint["feature_cols"])
    model = MLPRegressor(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    scaler = StandardScaler()
    scaler.mean_ = checkpoint["scaler_mean"]
    scaler.scale_ = checkpoint["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(feature_cols)
    return model, scaler, feature_cols, checkpoint.get("history", [])

