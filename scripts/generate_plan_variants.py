from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", default="outputs/daily/plan_variants.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates, dtype={"trade_date": str, "ts_code": str})
    candidates = candidates.sort_values("score", ascending=False)
    variants = [
        {
            "name": "aggressive_n8_industry1",
            "description": "默认进攻版：排除科创板，8只，每行业最多1只，不额外限制创业板数量。",
            "n": 8,
            "max_per_industry": 1,
            "max_chinext": 8,
        },
        {
            "name": "balanced_n8_chinext4",
            "description": "创业板上限版：8只，每行业最多1只，创业板最多4只。",
            "n": 8,
            "max_per_industry": 1,
            "max_chinext": 4,
        },
        {
            "name": "defensive_n8_main_bias",
            "description": "偏防守版：8只，每行业最多1只，创业板最多3只。",
            "n": 8,
            "max_per_industry": 1,
            "max_chinext": 3,
        },
        {
            "name": "concentrated_n5",
            "description": "高集中度冲排名版：5只，每行业最多1只。",
            "n": 5,
            "max_per_industry": 1,
            "max_chinext": 5,
        },
    ]
    output = []
    for spec in variants:
        rows = select(candidates, spec["n"], spec["max_per_industry"], spec["max_chinext"])
        output.append(
            {
                **spec,
                "selected": [row["ts_code"] for row in rows],
                "details": rows,
                "avg_score": sum(float(row.get("score", 0)) for row in rows) / max(len(rows), 1),
                "chinext_count": sum(1 for row in rows if row.get("market") == "创业板"),
            }
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(Path(args.output).read_text(encoding="utf-8"))


def select(df: pd.DataFrame, n: int, max_per_industry: int, max_chinext: int) -> list[dict]:
    industry_counts: Counter[str] = Counter()
    chinext_count = 0
    rows = []
    for _, row in df.iterrows():
        industry = str(row.get("industry", ""))
        market = str(row.get("market", ""))
        if industry_counts[industry] >= max_per_industry:
            continue
        if market == "创业板" and chinext_count >= max_chinext:
            continue
        record = {
            "ts_code": str(row.get("ts_code", "")),
            "name": str(row.get("name", "")),
            "industry": industry,
            "market": market,
            "score": float(row.get("score", 0)),
            "amount": float(row.get("amount", 0)),
            "ret_1": float(row.get("ret_1", 0)),
        }
        rows.append(record)
        industry_counts[industry] += 1
        if market == "创业板":
            chinext_count += 1
        if len(rows) >= n:
            break
    return rows


if __name__ == "__main__":
    main()
