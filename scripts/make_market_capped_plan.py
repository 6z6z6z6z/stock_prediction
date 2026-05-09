from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="outputs/latest_candidates_ensemble_no_star.csv")
    parser.add_argument("--output", default="outputs/rebalance_plan_market_capped.csv")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-per-industry", type=int, default=1)
    parser.add_argument("--max-chinext", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.candidates, dtype={"trade_date": str, "ts_code": str})
    df = df.sort_values("score", ascending=False)

    selected = []
    industry_counts: Counter[str] = Counter()
    chinext_count = 0
    for _, row in df.iterrows():
        industry = str(row.get("industry", ""))
        market = str(row.get("market", ""))
        if industry_counts[industry] >= args.max_per_industry:
            continue
        if market == "创业板" and chinext_count >= args.max_chinext:
            continue
        selected.append(row)
        industry_counts[industry] += 1
        if market == "创业板":
            chinext_count += 1
        if len(selected) >= args.n:
            break

    out = pd.DataFrame(selected)
    out.insert(0, "action", "buy_initial")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved market-capped plan -> {args.output}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

