from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare.backtest import rebalance_topk_backtest, rebalance_topk_backtest_diversified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--score-file", required=True)
    parser.add_argument("--output", default="outputs/diversified_eval.csv")
    parser.add_argument("--start-date", default="20260101")
    parser.add_argument("--end-date", default="20260420")
    parser.add_argument("--market-mode", choices=["all", "no_star", "main_only"], default="no_star")
    parser.add_argument("--min-amount", type=float, default=50000.0)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--k", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.feature_file).open("rb") as f:
        payload = pickle.load(f)
    features = payload["features"]
    ret_col = f"future_ret_{payload['horizon']}"
    scores = pd.read_csv(args.score_file, dtype={"trade_date": str, "ts_code": str})
    scores = scores[["trade_date", "ts_code", "score"]]
    df = features.loc[
        (features["trade_date"] >= args.start_date)
        & (features["trade_date"] <= args.end_date)
        & features["is_tradable"]
        & features[ret_col].notna()
        & (features["amount"] >= args.min_amount)
    ].merge(scores, on=["trade_date", "ts_code"], how="inner")
    if args.market_mode == "no_star":
        df = df.loc[~df["market"].eq("科创板")].copy()
    elif args.market_mode == "main_only":
        df = df.loc[df["market"].eq("主板")].copy()

    rows = []
    curve, metrics = rebalance_topk_backtest(df, "score", ret_col, n=args.n, k=args.k, cost=0.001)
    rows.append({"mode": "plain", "max_per_industry": None, **metrics, "avg_turnover": curve["turnover"].mean()})
    for cap in [1, 2, 3]:
        curve, metrics = rebalance_topk_backtest_diversified(
            df, "score", ret_col, n=args.n, k=args.k, max_per_industry=cap, cost=0.001
        )
        rows.append({"mode": "industry_cap", "max_per_industry": cap, **metrics, "avg_turnover": curve["turnover"].mean()})
    out = pd.DataFrame(rows).sort_values("annual_return", ascending=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

