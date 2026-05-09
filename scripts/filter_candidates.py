from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="outputs/latest_candidates_ensemble.csv")
    parser.add_argument("--output", default="outputs/latest_candidates_filtered.csv")
    parser.add_argument("--min-amount", type=float, default=100000.0)
    parser.add_argument("--market-mode", choices=["all", "no_star", "main_only"], default="all")
    parser.add_argument("--ret-mode", choices=["all", "no_large_drop", "no_large_move", "not_red_hot"], default="all")
    parser.add_argument("--top-n", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.candidates, dtype={"trade_date": str, "ts_code": str})
    mask = df["amount"].fillna(0) >= args.min_amount
    if args.market_mode == "main_only":
        mask &= df["market"].eq("主板")
    elif args.market_mode == "no_star":
        mask &= ~df["market"].eq("科创板")

    if args.ret_mode == "no_large_drop":
        mask &= df["ret_1"].fillna(0) > -0.07
    elif args.ret_mode == "no_large_move":
        mask &= df["ret_1"].abs().fillna(0) < 0.07
    elif args.ret_mode == "not_red_hot":
        mask &= df["ret_1"].fillna(0) < 0.05

    out = df.loc[mask].sort_values("score", ascending=False).head(args.top_n)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved {len(out)} filtered candidates -> {args.output}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

