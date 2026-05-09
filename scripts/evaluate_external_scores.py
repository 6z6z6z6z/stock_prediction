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
    parser.add_argument("--output", default="outputs/external_score_eval.csv")
    parser.add_argument("--start-date", default="20260101")
    parser.add_argument("--end-date", default="20260420")
    parser.add_argument("--score-col", default="score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.feature_file).open("rb") as f:
        payload = pickle.load(f)
    features = payload["features"]
    ret_col = f"future_ret_{payload['horizon']}"
    scores = pd.read_csv(args.score_file, dtype={"trade_date": str, "ts_code": str})
    scores = scores[["trade_date", "ts_code", args.score_col]].rename(columns={args.score_col: "score"})

    df = features.loc[
        (features["trade_date"] >= args.start_date)
        & (features["trade_date"] <= args.end_date)
        & features["is_tradable"]
        & features[ret_col].notna()
    ].merge(scores, on=["trade_date", "ts_code"], how="inner")

    rows = []
    ic = summarize_ic(daily_ic(df, "score", ret_col))
    top = top_group_returns(df, "score", ret_col, [5, 10, 20, 50])
    for n, k in [(5, 1), (10, 1), (20, 1), (20, 2)]:
        curve, metrics = rebalance_topk_backtest(df, "score", ret_col, n=n, k=k, cost=0.001)
        rows.append({"n": n, "k": k, **ic, **top, **metrics, "avg_turnover": curve["turnover"].mean()})
    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

