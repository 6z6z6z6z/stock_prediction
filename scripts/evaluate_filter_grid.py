from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare.backtest import rebalance_topk_backtest
from ashare.evaluate import daily_ic, summarize_ic, top_group_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--score-file", required=True)
    parser.add_argument("--output", default="outputs/filter_grid.csv")
    parser.add_argument("--start-date", default="20260101")
    parser.add_argument("--end-date", default="20260420")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--k", type=int, default=1)
    return parser.parse_args()


def build_mask(df: pd.DataFrame, min_amount: float, market_mode: str, ret_mode: str) -> pd.Series:
    mask = df["amount"].fillna(0) >= min_amount

    if market_mode == "main_only":
        mask &= df["market"].eq("主板")
    elif market_mode == "no_star":
        mask &= ~df["market"].eq("科创板")
    elif market_mode == "all":
        pass
    else:
        raise ValueError(f"unknown market_mode: {market_mode}")

    if ret_mode == "all":
        pass
    elif ret_mode == "no_large_drop":
        mask &= df["ret_1"].fillna(0) > -0.07
    elif ret_mode == "no_large_move":
        mask &= df["ret_1"].abs().fillna(0) < 0.07
    elif ret_mode == "not_red_hot":
        mask &= df["ret_1"].fillna(0) < 0.05
    else:
        raise ValueError(f"unknown ret_mode: {ret_mode}")

    return mask


def main() -> None:
    args = parse_args()
    with Path(args.feature_file).open("rb") as f:
        payload = pickle.load(f)
    features = payload["features"]
    ret_col = f"future_ret_{payload['horizon']}"
    scores = pd.read_csv(args.score_file, dtype={"trade_date": str, "ts_code": str})
    scores = scores[["trade_date", "ts_code", args.score_col]].rename(columns={args.score_col: "score"})

    base = features.loc[
        (features["trade_date"] >= args.start_date)
        & (features["trade_date"] <= args.end_date)
        & features["is_tradable"]
        & features[ret_col].notna()
    ].merge(scores, on=["trade_date", "ts_code"], how="inner")

    rows = []
    for min_amount in [50000.0, 100000.0, 200000.0, 500000.0, 1000000.0]:
        for market_mode in ["all", "no_star", "main_only"]:
            for ret_mode in ["all", "no_large_drop", "no_large_move", "not_red_hot"]:
                df = base.loc[build_mask(base, min_amount, market_mode, ret_mode)].copy()
                if df.empty:
                    continue
                per_day = df.groupby("trade_date")["ts_code"].size()
                if per_day.min() < args.n:
                    continue
                curve, metrics = rebalance_topk_backtest(df, "score", ret_col, n=args.n, k=args.k, cost=0.001)
                ic = summarize_ic(daily_ic(df, "score", ret_col))
                top = top_group_returns(df, "score", ret_col, [5, 10, 20])
                rows.append(
                    {
                        "min_amount": min_amount,
                        "market_mode": market_mode,
                        "ret_mode": ret_mode,
                        "min_daily_candidates": int(per_day.min()),
                        "avg_daily_candidates": float(per_day.mean()),
                        **ic,
                        **top,
                        **metrics,
                        "avg_turnover": float(curve["turnover"].mean()) if len(curve) else None,
                    }
                )

    out = pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()

