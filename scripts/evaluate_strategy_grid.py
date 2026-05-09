from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare.backtest import rebalance_topk_backtest, topn_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file", default="outputs/val_predictions.csv")
    parser.add_argument("--output", default="outputs/strategy_grid.csv")
    parser.add_argument("--cost", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred = pd.read_csv(args.pred_file)
    ret_cols = [c for c in pred.columns if c.startswith("future_ret_")]
    if not ret_cols:
        raise ValueError("prediction file must contain a future_ret_* column")
    ret_col = ret_cols[0]

    rows = []
    for n in [5, 10, 15, 20, 30, 50]:
        curve, metrics = topn_backtest(pred, "score", ret_col, n=n, cost=args.cost)
        rows.append(
            {
                "strategy": "topn",
                "n": n,
                "k": None,
                **metrics,
                "avg_turnover": curve["turnover"].mean() if len(curve) else None,
            }
        )

    for n in [10, 15, 20, 30, 50]:
        for k in [1, 2, 3, 5, 10]:
            if k > n:
                continue
            curve, metrics = rebalance_topk_backtest(pred, "score", ret_col, n=n, k=k, cost=args.cost)
            rows.append(
                {
                    "strategy": "rebalance",
                    "n": n,
                    "k": k,
                    **metrics,
                    "avg_turnover": curve["turnover"].mean() if len(curve) else None,
                }
            )

    result = pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

