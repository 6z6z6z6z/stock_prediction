from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare.backtest import rebalance_topk_backtest, topn_backtest
from ashare.evaluate import daily_ic, summarize_ic, top_group_returns
from ashare.model import MLPRegressor, predict, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", default="outputs/features_2023.pkl")
    parser.add_argument("--model-out", default="outputs/rank_mlp_checkpoint.pt")
    parser.add_argument("--pred-out", default="outputs/val_predictions_rank_mlp.csv")
    parser.add_argument("--metrics-out", default="outputs/metrics_rank_mlp.json")
    parser.add_argument("--train-end", default="20251231")
    parser.add_argument("--val-start", default="20260101")
    parser.add_argument("--val-end", default="20260420")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--top-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=1)
    return parser.parse_args()


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = pred.norm() * target.norm() + 1e-8
    return 1.0 - torch.sum(pred * target) / denom


def top_decile_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Aggressive objective: separate the top target decile from the rest.
    threshold = torch.quantile(target.detach(), 0.90)
    label = (target >= threshold).float()
    if label.sum() < 2:
        return pred.new_tensor(0.0)
    return nn.functional.binary_cross_entropy_with_logits(pred, label)


def fit_rank_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    val_x = scaler.transform(val_df[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    train_y = train_df["target"].to_numpy(dtype=np.float32)
    val_y = val_df["target"].to_numpy(dtype=np.float32)

    train_dates = train_df["trade_date"].to_numpy()
    groups = []
    for date in pd.unique(train_dates):
        idx = np.flatnonzero(train_dates == date)
        if len(idx) >= 100:
            groups.append(idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(len(feature_cols), hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    huber = nn.SmoothL1Loss()

    x_val_t = torch.from_numpy(val_x).to(device)
    y_val_t = torch.from_numpy(val_y).to(device)
    best_state = None
    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(groups)
        losses = []
        for idx in groups:
            xb = torch.from_numpy(train_x[idx]).to(device)
            yb = torch.from_numpy(train_y[idx]).to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = huber(pred, yb)
            loss = loss + args.rank_weight * pearson_loss(pred, yb)
            loss = loss + args.top_weight * top_decile_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_loss = float((huber(val_pred, y_val_t) + pearson_loss(val_pred, y_val_t)).detach().cpu())
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss}
        history.append(row)
        print(row)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    class Result:
        pass

    result = Result()
    result.model = model
    result.scaler = scaler
    result.history = history
    return result


def main() -> None:
    args = parse_args()
    with Path(args.feature_file).open("rb") as f:
        payload = pickle.load(f)
    df = payload["features"].copy()
    feature_cols = payload["feature_cols"]
    ret_col = f"future_ret_{payload['horizon']}"

    usable = df.loc[df["is_tradable"] & df["target"].notna() & df[ret_col].notna()].copy()
    train_df = usable.loc[usable["trade_date"] <= args.train_end].copy()
    val_df = usable.loc[(usable["trade_date"] >= args.val_start) & (usable["trade_date"] <= args.val_end)].copy()
    if train_df.empty or val_df.empty:
        raise ValueError(f"empty train or validation split: train={len(train_df)}, val={len(val_df)}")

    result = fit_rank_model(train_df, val_df, feature_cols, args)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.model_out, result, feature_cols)

    val_df["score"] = predict(result.model, result.scaler, val_df, feature_cols)
    pred_cols = ["trade_date", "ts_code", "name", "industry", "close", "amount", "score", ret_col, "target"]
    Path(args.pred_out).parent.mkdir(parents=True, exist_ok=True)
    val_df[pred_cols].sort_values(["trade_date", "score"], ascending=[True, False]).to_csv(
        args.pred_out, index=False, encoding="utf-8-sig"
    )

    ic_df = daily_ic(val_df, "score", ret_col)
    top_curve, top_metrics = topn_backtest(val_df, "score", ret_col, n=args.top_n)
    reb_curve, reb_metrics = rebalance_topk_backtest(val_df, "score", ret_col, n=args.top_n, k=args.top_k)
    metrics = {
        "rows": {"train": len(train_df), "val": len(val_df)},
        "loss_history": result.history,
        "ic": summarize_ic(ic_df),
        "top_group_returns": top_group_returns(val_df, "score", ret_col, [5, 10, 20, 50]),
        "topn_backtest": top_metrics,
        "rebalance_topk_backtest": reb_metrics,
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if not top_curve.empty:
        top_curve.to_csv(Path(args.metrics_out).with_name("rank_topn_curve.csv"), index=False, encoding="utf-8-sig")
    if not reb_curve.empty:
        reb_curve.to_csv(Path(args.metrics_out).with_name("rank_rebalance_curve.csv"), index=False, encoding="utf-8-sig")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

