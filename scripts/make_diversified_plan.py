from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="outputs/latest_candidates_ensemble_no_star.csv")
    parser.add_argument("--holdings", default="")
    parser.add_argument("--output", default="outputs/rebalance_plan_diversified.csv")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-per-industry", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.candidates, dtype={"trade_date": str, "ts_code": str})
    df = df.sort_values("score", ascending=False)

    if not args.holdings:
        selected = _select_rows(df, args.n, args.max_per_industry)
        out = pd.DataFrame(selected)
        out.insert(0, "action", "buy_initial")
    else:
        holdings = pd.read_csv(args.holdings, dtype={"ts_code": str})
        if "ts_code" not in holdings.columns:
            raise ValueError("holdings file must contain a ts_code column")
        score_map = df.set_index("ts_code")["score"].to_dict()
        held_codes = holdings["ts_code"].dropna().astype(str).tolist()
        sell_codes = sorted(held_codes, key=lambda code: score_map.get(code, float("-inf")))[: args.k]
        keep_codes = [code for code in held_codes if code not in set(sell_codes)]
        keep_industries = []
        if "industry" in holdings.columns:
            keep_industries = holdings.loc[holdings["ts_code"].isin(keep_codes), "industry"].astype(str).tolist()
        else:
            industry_map = df.set_index("ts_code")["industry"].astype(str).to_dict()
            keep_industries = [industry_map.get(code, "") for code in keep_codes]

        buy_count = max(args.k, args.n - len(keep_codes))
        buy_rows = _select_rows(
            df.loc[~df["ts_code"].isin(held_codes)],
            buy_count,
            args.max_per_industry,
            existing_industries=keep_industries,
        )
        sell_rows = holdings.loc[holdings["ts_code"].isin(sell_codes)].copy()
        if "action" in sell_rows.columns:
            sell_rows = sell_rows.drop(columns=["action"])
        sell_rows["score"] = sell_rows["ts_code"].map(score_map).fillna(float("-inf"))
        sell_rows.insert(0, "action", "sell")
        buy_df = pd.DataFrame(buy_rows)
        if "action" in buy_df.columns:
            buy_df = buy_df.drop(columns=["action"])
        buy_df.insert(0, "action", "buy")
        out = pd.concat([sell_rows, buy_df], ignore_index=True, sort=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved diversified plan -> {args.output}")
    print(out.to_string(index=False))


def _select_rows(
    df: pd.DataFrame,
    n: int,
    max_per_industry: int,
    existing_industries: list[str] | None = None,
) -> list[pd.Series]:
    selected = []
    counts: Counter[str] = Counter(existing_industries or [])
    for _, row in df.iterrows():
        industry = str(row.get("industry", ""))
        if counts[industry] >= max_per_industry:
            continue
        selected.append(row)
        counts[industry] += 1
        if len(selected) >= n:
            break
    return selected


if __name__ == "__main__":
    main()
