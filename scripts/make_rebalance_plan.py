from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="outputs/latest_candidates_2023.csv")
    parser.add_argument("--holdings", default="")
    parser.add_argument("--output", default="outputs/rebalance_plan.csv")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--k", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates, dtype={"trade_date": str, "ts_code": str})
    candidates = candidates.sort_values("score", ascending=False).reset_index(drop=True)

    if not args.holdings:
        plan = candidates.head(args.n).copy()
        plan.insert(0, "action", "buy_initial")
    else:
        holdings = pd.read_csv(args.holdings, dtype={"ts_code": str})
        if "ts_code" not in holdings.columns:
            raise ValueError("holdings file must contain a ts_code column")

        held_codes = holdings["ts_code"].dropna().astype(str).tolist()
        score_map = candidates.set_index("ts_code")["score"].to_dict()
        sell_codes = sorted(held_codes, key=lambda code: score_map.get(code, float("-inf")))[: args.k]
        keep_codes = [code for code in held_codes if code not in set(sell_codes)]
        buy_count = max(args.k, args.n - len(keep_codes))
        buy_rows = candidates.loc[~candidates["ts_code"].isin(keep_codes)].head(buy_count).copy()

        sell_rows = holdings.loc[holdings["ts_code"].isin(sell_codes)].copy()
        sell_rows["score"] = sell_rows["ts_code"].map(score_map).fillna(float("-inf"))
        sell_rows.insert(0, "action", "sell")
        buy_rows.insert(0, "action", "buy")
        plan = pd.concat([sell_rows, buy_rows], ignore_index=True, sort=False)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved rebalance plan -> {args.output}")
    print(plan.to_string(index=False))


if __name__ == "__main__":
    main()
