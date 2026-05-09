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
    parser.add_argument("--output", default="outputs/final_strategy_grid.csv")
    parser.add_argument("--start-date", default="20260101")
    parser.add_argument("--end-date", default="20260420")
    parser.add_argument("--min-amount", type=float, default=50000.0)
    parser.add_argument("--markets", nargs="+", default=["no_star", "main_only"])
    parser.add_argument("--ns", nargs="+", type=int, default=[5, 8, 10, 12, 15])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--caps", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.feature_file).open("rb") as f:
        payload = pickle.load(f)
    features = payload["features"]
    ret_col = f"future_ret_{payload['horizon']}"
    scores = pd.read_csv(args.score_file, dtype={"trade_date": str, "ts_code": str})
    scores = scores[["trade_date", "ts_code", "score"]]

    base = features.loc[
        (features["trade_date"] >= args.start_date)
        & (features["trade_date"] <= args.end_date)
        & features["is_tradable"]
        & features[ret_col].notna()
        & (features["amount"] >= args.min_amount)
    ].merge(scores, on=["trade_date", "ts_code"], how="inner")

    rows = []
    for market_mode in args.markets:
        if market_mode == "all":
            df = base.copy()
        elif market_mode == "no_star":
            df = base.loc[~base["market"].eq("科创板")].copy()
        else:
            df = base.loc[base["market"].eq("主板")].copy()

        per_day = df.groupby("trade_date")["ts_code"].size()
        if per_day.empty:
            continue

        for n in args.ns:
            for k in args.ks:
                if k >= n:
                    continue
                if per_day.min() < n:
                    continue

                curve, metrics = rebalance_topk_backtest(df, "score", ret_col, n=n, k=k, cost=0.001)
                rows.append(
                    {
                        "market_mode": market_mode,
                        "industry_cap": 0,
                        "n": n,
                        "k": k,
                        **metrics,
                        "avg_turnover": curve["turnover"].mean() if len(curve) else None,
                    }
                )

                for cap in args.caps:
                    if cap == 0:
                        continue
                    if cap * len(df["industry"].dropna().unique()) < n:
                        continue
                    curve, metrics = rebalance_topk_backtest_diversified(
                        df, "score", ret_col, n=n, k=k, max_per_industry=cap, cost=0.001
                    )
                    rows.append(
                        {
                            "market_mode": market_mode,
                            "industry_cap": cap,
                            "n": n,
                            "k": k,
                            **metrics,
                            "avg_turnover": curve["turnover"].mean() if len(curve) else None,
                        }
                    )

    out = pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(out.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
