from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare.backtest import rebalance_topk_backtest, topn_backtest
from ashare.evaluate import daily_ic, summarize_ic, top_group_returns
from ashare.model import predict, save_checkpoint, train_mlp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", default="outputs/features.pkl")
    parser.add_argument("--model-out", default="outputs/mlp_checkpoint.pt")
    parser.add_argument("--pred-out", default="outputs/val_predictions.csv")
    parser.add_argument("--metrics-out", default="outputs/metrics.json")
    parser.add_argument("--train-end", default="20251231")
    parser.add_argument("--val-start", default="20260101")
    parser.add_argument("--val-end", default="20260420")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


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

    result = train_mlp(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        seed=args.seed,
    )
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
        "top_group_returns": top_group_returns(val_df, "score", ret_col, [5, 10, 20]),
        "topn_backtest": top_metrics,
        "rebalance_topk_backtest": reb_metrics,
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if not top_curve.empty:
        top_curve.to_csv(Path(args.metrics_out).with_name("topn_curve.csv"), index=False, encoding="utf-8-sig")
    if not reb_curve.empty:
        reb_curve.to_csv(Path(args.metrics_out).with_name("rebalance_curve.csv"), index=False, encoding="utf-8-sig")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
