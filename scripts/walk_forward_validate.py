from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare.backtest import rebalance_topk_backtest, rebalance_topk_backtest_diversified
from ashare.evaluate import daily_ic, summarize_ic, top_group_returns
from ashare.model import predict, train_mlp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--output-dir", default="outputs/walk_forward")
    parser.add_argument("--val-start", default="20250101")
    parser.add_argument("--val-end", default="20260420")
    parser.add_argument("--train-start", default="")
    parser.add_argument("--train-window-months", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--market-mode", choices=["all", "no_star", "main_only"], default="no_star")
    parser.add_argument("--min-amount", type=float, default=50000.0)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-per-industry", type=int, default=1)
    parser.add_argument("--cost", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.feature_file).open("rb") as f:
        payload = pickle.load(f)

    df = payload["features"].copy()
    feature_cols = payload["feature_cols"]
    ret_col = f"future_ret_{payload['horizon']}"

    usable = df.loc[df["is_tradable"] & df["target"].notna() & df[ret_col].notna()].copy()
    usable["trade_date"] = usable["trade_date"].astype(str)
    fold_months = validation_months(usable["trade_date"], args.val_start, args.val_end)
    if args.max_folds > 0:
        fold_months = fold_months[: args.max_folds]
    if not fold_months:
        raise ValueError("no validation months found")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    pred_frames = []
    for fold_idx, month in enumerate(fold_months, start=1):
        val_mask = usable["trade_date"].str.startswith(month)
        val_mask &= usable["trade_date"].between(args.val_start, args.val_end)
        val_df = usable.loc[val_mask].copy()
        if val_df.empty:
            continue

        fold_val_start = val_df["trade_date"].min()
        prev_dates = usable.loc[usable["trade_date"] < fold_val_start, "trade_date"]
        if prev_dates.empty:
            continue
        train_end = prev_dates.max()
        train_mask = usable["trade_date"] <= train_end
        if args.train_start:
            train_mask &= usable["trade_date"] >= args.train_start
        if args.train_window_months > 0:
            min_train_date = month_offset(fold_val_start, -args.train_window_months)
            train_mask &= usable["trade_date"] >= min_train_date
        train_df = usable.loc[train_mask].copy()
        if train_df.empty:
            continue

        print(
            f"fold {fold_idx}/{len(fold_months)} month={month} "
            f"train={train_df['trade_date'].min()}..{train_end} rows={len(train_df)} "
            f"val={val_df['trade_date'].min()}..{val_df['trade_date'].max()} rows={len(val_df)}"
        )
        result = train_mlp(
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            seed=args.seed + fold_idx - 1,
        )
        val_df["score"] = predict(result.model, result.scaler, val_df, feature_cols)
        fold_ic = summarize_ic(daily_ic(val_df, "score", ret_col))
        fold_top = top_group_returns(val_df, "score", ret_col, [5, 8, 10, 20])
        fold_rows.append(
            {
                "fold": fold_idx,
                "month": month,
                "train_start": train_df["trade_date"].min(),
                "train_end": train_end,
                "val_start": val_df["trade_date"].min(),
                "val_end": val_df["trade_date"].max(),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                **fold_ic,
                **fold_top,
                "best_val_loss": min(row["val_loss"] for row in result.history),
            }
        )
        pred_cols = [
            "trade_date",
            "ts_code",
            "name",
            "industry",
            "market",
            "close",
            "amount",
            "ret_1",
            "score",
            ret_col,
            "target",
        ]
        pred_frames.append(val_df[[c for c in pred_cols if c in val_df.columns]].copy())

    if not pred_frames:
        raise ValueError("no fold predictions produced")

    preds = pd.concat(pred_frames, ignore_index=True)
    preds = preds.sort_values(["trade_date", "score"], ascending=[True, False])
    fold_metrics = pd.DataFrame(fold_rows)

    strategy_df = apply_live_filters(preds, args.market_mode, args.min_amount)
    if args.max_per_industry > 0:
        curve, strategy_metrics = rebalance_topk_backtest_diversified(
            strategy_df,
            "score",
            ret_col,
            n=args.n,
            k=args.k,
            max_per_industry=args.max_per_industry,
            cost=args.cost,
        )
    else:
        curve, strategy_metrics = rebalance_topk_backtest(
            strategy_df,
            "score",
            ret_col,
            n=args.n,
            k=args.k,
            cost=args.cost,
        )

    overall = {
        "feature_file": str(Path(args.feature_file).resolve()),
        "target_mode": payload.get("target_mode", ""),
        "horizon": payload.get("horizon", 1),
        "folds": len(fold_metrics),
        "rows": {"predictions": len(preds), "strategy_rows": len(strategy_df)},
        "ic": summarize_ic(daily_ic(preds, "score", ret_col)),
        "top_group_returns": top_group_returns(preds, "score", ret_col, [5, 8, 10, 20]),
        "strategy": {
            "market_mode": args.market_mode,
            "min_amount": args.min_amount,
            "n": args.n,
            "k": args.k,
            "max_per_industry": args.max_per_industry,
            **strategy_metrics,
            "avg_turnover": float(curve["turnover"].mean()) if len(curve) else None,
        },
    }

    pred_path = out_dir / "walk_forward_predictions.csv"
    fold_path = out_dir / "walk_forward_fold_metrics.csv"
    curve_path = out_dir / "walk_forward_strategy_curve.csv"
    summary_path = out_dir / "walk_forward_summary.json"
    preds.to_csv(pred_path, index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(fold_path, index=False, encoding="utf-8-sig")
    curve.to_csv(curve_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2))


def validation_months(dates: pd.Series, start: str, end: str) -> list[str]:
    valid = dates.loc[dates.between(start, end)]
    return sorted(valid.str.slice(0, 6).unique().tolist())


def month_offset(date: str, months: int) -> str:
    ts = pd.Timestamp(year=int(date[:4]), month=int(date[4:6]), day=1)
    shifted = ts + pd.DateOffset(months=months)
    return shifted.strftime("%Y%m%d")


def apply_live_filters(df: pd.DataFrame, market_mode: str, min_amount: float) -> pd.DataFrame:
    out = df.loc[df["amount"].fillna(0) >= min_amount].copy()
    if market_mode == "no_star":
        out = out.loc[~out["market"].eq("科创板")].copy()
    elif market_mode == "main_only":
        out = out.loc[out["market"].eq("主板")].copy()
    return out


if __name__ == "__main__":
    main()
