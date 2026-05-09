from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare.backtest import rebalance_topk_backtest, rebalance_topk_backtest_diversified
from ashare.evaluate import daily_ic, summarize_ic, top_group_returns
from ashare.model import predict, save_checkpoint, train_mlp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a latest-data MLP ensemble for live contest prediction."
    )
    parser.add_argument("--feature-file", default="outputs/features_live.pkl")
    parser.add_argument("--output-dir", default="outputs/live_ensemble")
    parser.add_argument("--prefix", default="live_mlp")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 7, 42])
    parser.add_argument("--val-days", type=int, default=20)
    parser.add_argument("--train-start", default="")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--eval-ret-mode", choices=["payload", "close", "open2open", "intraday"], default="open2open")
    parser.add_argument("--market-mode", choices=["all", "no_star", "main_only"], default="no_star")
    parser.add_argument("--min-amount", type=float, default=50000.0)
    parser.add_argument("--latest-top-n", type=int, default=120)
    parser.add_argument("--plan-n", type=int, default=8)
    parser.add_argument("--plan-k", type=int, default=1)
    parser.add_argument("--max-per-industry", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.feature_file).open("rb") as f:
        payload = pickle.load(f)

    df = payload["features"].copy()
    feature_cols = list(payload["feature_cols"])
    horizon = int(payload["horizon"])
    target_ret_col = f"future_ret_{horizon}"
    eval_ret_col = resolve_eval_ret_col(df, horizon, args.eval_ret_mode, target_ret_col)

    usable = df.loc[
        df["is_tradable"] & df["target"].notna() & df[target_ret_col].notna()
    ].copy()
    if args.train_start:
        usable = usable.loc[usable["trade_date"] >= args.train_start].copy()
    if usable.empty:
        raise ValueError("No labeled rows available for live training.")

    train_df, val_df, split_info = split_latest_tail(usable, args.val_days)
    latest_date = str(df["trade_date"].max())
    latest_df = df.loc[(df["trade_date"] == latest_date) & df["is_tradable"]].copy()
    if latest_df.empty:
        raise ValueError(f"No tradable latest rows for {latest_date}.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_scores: list[pd.Series] = []
    latest_scores: list[pd.Series] = []
    histories = {}
    checkpoint_paths = []

    for seed in args.seeds:
        result = train_mlp(
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            seed=seed,
        )
        checkpoint = output_dir / f"{args.prefix}_seed{seed}.pt"
        save_checkpoint(str(checkpoint), result, feature_cols)
        checkpoint_paths.append(str(checkpoint))
        histories[str(seed)] = result.history
        val_scores.append(pd.Series(predict(result.model, result.scaler, val_df, feature_cols), index=val_df.index))
        latest_scores.append(pd.Series(predict(result.model, result.scaler, latest_df, feature_cols), index=latest_df.index))

    val_out = val_df.copy()
    latest_out = latest_df.copy()
    score_cols = []
    for seed, val_score, latest_score in zip(args.seeds, val_scores, latest_scores):
        col = f"score_seed{seed}"
        score_cols.append(col)
        val_out[col] = val_score
        latest_out[col] = latest_score
    val_out["score"] = val_out[score_cols].mean(axis=1)
    latest_out["score"] = latest_out[score_cols].mean(axis=1)

    val_pred_path = output_dir / f"{args.prefix}_validation.csv"
    latest_path = output_dir / f"{args.prefix}_latest_candidates.csv"
    filtered_path = output_dir / f"{args.prefix}_latest_filtered.csv"
    plan_path = output_dir / f"{args.prefix}_rebalance_plan.csv"
    metrics_path = output_dir / f"{args.prefix}_metrics.json"

    val_cols = [
        "trade_date",
        "ts_code",
        "name",
        "industry",
        "market",
        "close",
        "amount",
        "score",
        *score_cols,
        target_ret_col,
        eval_ret_col,
        "target",
    ]
    val_cols = dedupe_existing_cols(val_out, val_cols)
    val_out[val_cols].sort_values(["trade_date", "score"], ascending=[True, False]).to_csv(
        val_pred_path, index=False, encoding="utf-8-sig"
    )

    latest_cols = [
        "trade_date",
        "ts_code",
        "name",
        "industry",
        "market",
        "close",
        "amount",
        "ret_1",
        "score",
        *score_cols,
    ]
    latest_cols = dedupe_existing_cols(latest_out, latest_cols)
    latest_candidates = latest_out.sort_values("score", ascending=False).head(args.latest_top_n)
    latest_candidates[latest_cols].to_csv(latest_path, index=False, encoding="utf-8-sig")

    filtered = apply_live_filters(latest_candidates, args.market_mode, args.min_amount)
    filtered[latest_cols].to_csv(filtered_path, index=False, encoding="utf-8-sig")
    plan = make_initial_plan(filtered, args.plan_n, args.max_per_industry)
    plan.insert(0, "action", "buy_initial")
    plan.to_csv(plan_path, index=False, encoding="utf-8-sig")

    eval_df = val_out.loc[val_out[eval_ret_col].notna()].copy()
    ic_df = daily_ic(eval_df, "score", eval_ret_col)
    if args.max_per_industry > 0:
        curve, backtest_metrics = rebalance_topk_backtest_diversified(
            apply_live_filters(eval_df, args.market_mode, args.min_amount),
            "score",
            eval_ret_col,
            n=args.plan_n,
            k=args.plan_k,
            max_per_industry=args.max_per_industry,
            cost=0.001,
        )
    else:
        curve, backtest_metrics = rebalance_topk_backtest(
            apply_live_filters(eval_df, args.market_mode, args.min_amount),
            "score",
            eval_ret_col,
            n=args.plan_n,
            k=args.plan_k,
            cost=0.001,
        )

    metrics = {
        "feature_file": str(Path(args.feature_file).resolve()),
        "latest_date": latest_date,
        "checkpoint_paths": checkpoint_paths,
        "target_mode": payload.get("target_mode", "unknown"),
        "target_ret_col": target_ret_col,
        "eval_ret_col": eval_ret_col,
        "split": split_info,
        "rows": {"train": int(len(train_df)), "val": int(len(val_df)), "latest": int(len(latest_df))},
        "seeds": args.seeds,
        "loss_history": histories,
        "ic": summarize_ic(ic_df),
        "top_group_returns": top_group_returns(eval_df, "score", eval_ret_col, [5, 8, 10, 20]),
        "rebalance_backtest": backtest_metrics,
        "avg_turnover": float(curve["turnover"].mean()) if len(curve) else None,
        "live_filters": {
            "market_mode": args.market_mode,
            "min_amount": args.min_amount,
            "plan_n": args.plan_n,
            "plan_k": args.plan_k,
            "max_per_industry": args.max_per_industry,
        },
        "outputs": {
            "validation": str(val_pred_path),
            "latest_candidates": str(latest_path),
            "filtered_candidates": str(filtered_path),
            "rebalance_plan": str(plan_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(curve):
        curve.to_csv(output_dir / f"{args.prefix}_validation_curve.csv", index=False, encoding="utf-8-sig")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print()
    print(plan[["action", "trade_date", "ts_code", "name", "industry", "market", "close", "amount", "score"]].to_string(index=False))


def resolve_eval_ret_col(df: pd.DataFrame, horizon: int, mode: str, target_ret_col: str) -> str:
    if mode == "payload":
        return target_ret_col
    col = f"future_ret_{horizon}_{mode}"
    if col not in df.columns:
        raise ValueError(f"Requested eval return column is missing: {col}")
    return col


def split_latest_tail(df: pd.DataFrame, val_days: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dates = sorted(df["trade_date"].dropna().astype(str).unique())
    if len(dates) <= val_days:
        raise ValueError(f"Need more than {val_days} labeled dates, got {len(dates)}.")
    val_dates = dates[-val_days:]
    train_dates = dates[:-val_days]
    train_df = df.loc[df["trade_date"].isin(train_dates)].copy()
    val_df = df.loc[df["trade_date"].isin(val_dates)].copy()
    return train_df, val_df, {
        "train_start": train_dates[0],
        "train_end": train_dates[-1],
        "val_start": val_dates[0],
        "val_end": val_dates[-1],
        "val_days": val_days,
    }


def apply_live_filters(df: pd.DataFrame, market_mode: str, min_amount: float) -> pd.DataFrame:
    out = df.loc[df["amount"].fillna(0) >= min_amount].copy()
    if market_mode == "no_star":
        out = out.loc[~out["market"].eq("科创板")].copy()
    elif market_mode == "main_only":
        out = out.loc[out["market"].eq("主板")].copy()
    return out.sort_values("score", ascending=False)


def make_initial_plan(df: pd.DataFrame, n: int, max_per_industry: int) -> pd.DataFrame:
    if max_per_industry <= 0:
        return df.head(n).copy()
    rows = []
    counts: Counter[str] = Counter()
    for _, row in df.sort_values("score", ascending=False).iterrows():
        industry = str(row.get("industry", ""))
        if counts[industry] >= max_per_industry:
            continue
        rows.append(row)
        counts[industry] += 1
        if len(rows) >= n:
            break
    if len(rows) < n:
        raise ValueError(f"Only selected {len(rows)} rows with industry cap {max_per_industry}; need {n}.")
    return pd.DataFrame(rows)


def dedupe_existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    result = []
    seen = set()
    for col in cols:
        if col in df.columns and col not in seen:
            result.append(col)
            seen.add(col)
    return result


if __name__ == "__main__":
    main()
