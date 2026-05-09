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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved scores against a selected realized-return column.")
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--score-file", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--eval-ret-mode", choices=["payload", "close", "open2open", "intraday"], default="open2open")
    parser.add_argument("--market-mode", choices=["all", "no_star", "main_only"], default="no_star")
    parser.add_argument("--ret-mode", choices=["all", "no_large_drop", "no_large_move", "not_red_hot"], default="all")
    parser.add_argument("--min-amount", type=float, default=50000.0)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-per-industry", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.feature_file).open("rb") as f:
        payload = pickle.load(f)

    features = payload["features"]
    horizon = int(payload["horizon"])
    ret_col = resolve_ret_col(features, horizon, args.eval_ret_mode)
    scores = pd.read_csv(args.score_file, dtype={"trade_date": str, "ts_code": str})
    if "score" not in scores.columns:
        raise ValueError("score file must contain a score column")

    base_cols = [
        "trade_date",
        "ts_code",
        "name",
        "industry",
        "market",
        "amount",
        "ret_1",
        "is_tradable",
        ret_col,
    ]
    df = features.loc[
        (features["trade_date"] >= args.start_date)
        & (features["trade_date"] <= args.end_date)
        & features["is_tradable"]
        & features[ret_col].notna(),
        base_cols,
    ].merge(scores[["trade_date", "ts_code", "score"]], on=["trade_date", "ts_code"], how="inner")
    df = apply_filters(df, args.market_mode, args.ret_mode, args.min_amount)

    ic_df = daily_ic(df, "score", ret_col)
    if args.max_per_industry > 0:
        curve, backtest = rebalance_topk_backtest_diversified(
            df,
            "score",
            ret_col,
            n=args.n,
            k=args.k,
            max_per_industry=args.max_per_industry,
            cost=0.001,
        )
    else:
        curve, backtest = rebalance_topk_backtest(df, "score", ret_col, n=args.n, k=args.k, cost=0.001)

    metrics = {
        "feature_file": str(Path(args.feature_file).resolve()),
        "score_file": str(Path(args.score_file).resolve()),
        "date_range": {"start": args.start_date, "end": args.end_date},
        "eval_ret_col": ret_col,
        "rows": int(len(df)),
        "dates": int(df["trade_date"].nunique()),
        "ic": summarize_ic(ic_df),
        "top_group_returns": top_group_returns(df, "score", ret_col, [5, 8, 10, 20]),
        "rebalance_backtest": backtest,
        "avg_turnover": float(curve["turnover"].mean()) if len(curve) else None,
        "filters": {
            "market_mode": args.market_mode,
            "ret_mode": args.ret_mode,
            "min_amount": args.min_amount,
            "n": args.n,
            "k": args.k,
            "max_per_industry": args.max_per_industry,
        },
    }
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        if len(curve):
            curve.to_csv(out.with_name(out.stem + "_curve.csv"), index=False, encoding="utf-8-sig")
    print(text)


def resolve_ret_col(features: pd.DataFrame, horizon: int, mode: str) -> str:
    col = f"future_ret_{horizon}" if mode == "payload" else f"future_ret_{horizon}_{mode}"
    if col not in features.columns:
        raise ValueError(f"Missing return column: {col}")
    return col


def apply_filters(df: pd.DataFrame, market_mode: str, ret_mode: str, min_amount: float) -> pd.DataFrame:
    out = df.loc[df["amount"].fillna(0) >= min_amount].copy()
    if market_mode == "no_star":
        out = out.loc[~out["market"].eq("科创板")].copy()
    elif market_mode == "main_only":
        out = out.loc[out["market"].eq("主板")].copy()
    if "ret_1" in out.columns:
        if ret_mode == "no_large_drop":
            out = out.loc[out["ret_1"].fillna(0) > -0.07].copy()
        elif ret_mode == "no_large_move":
            out = out.loc[out["ret_1"].abs().fillna(0) < 0.07].copy()
        elif ret_mode == "not_red_hot":
            out = out.loc[out["ret_1"].fillna(0) < 0.05].copy()
    return out.sort_values(["trade_date", "score"], ascending=[True, False])


if __name__ == "__main__":
    main()
