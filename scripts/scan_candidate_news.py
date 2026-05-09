from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


POSITIVE_WORDS = [
    "增长",
    "同比增长",
    "预增",
    "中标",
    "签订",
    "合作",
    "回购",
    "增持",
    "突破",
    "创新高",
    "涨停",
]

NEGATIVE_WORDS = [
    "亏损",
    "同比下降",
    "减持",
    "立案",
    "调查",
    "处罚",
    "风险警示",
    "退市",
    "下跌",
    "跌停",
    "终止",
    "违约",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../A股数据")
    parser.add_argument("--candidates", default="outputs/rebalance_plan_final_n8_no_star_industry1.csv")
    parser.add_argument("--date", default="")
    parser.add_argument("--output", default="outputs/candidate_news_scan.csv")
    return parser.parse_args()


def keyword_score(text: str) -> tuple[int, int, str]:
    pos = [w for w in POSITIVE_WORDS if w in text]
    neg = [w for w in NEGATIVE_WORDS if w in text]
    return len(pos), len(neg), ",".join(pos + neg)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    candidates = pd.read_csv(args.candidates, dtype={"ts_code": str})
    if args.date:
        date = args.date
    elif "trade_date" in candidates.columns and candidates["trade_date"].notna().any():
        date = str(candidates["trade_date"].dropna().iloc[0])
    else:
        daily_files = sorted((data_dir / "daily").glob("*.csv"))
        date = daily_files[-1].stem

    news_path = data_dir / "news" / f"{date}.csv"
    if not news_path.exists():
        raise FileNotFoundError(f"news file not found: {news_path}")
    news = pd.read_csv(news_path)
    news["text"] = news["title"].fillna("") + " " + news["content"].fillna("")

    basic = pd.read_csv(data_dir / "basic.csv", dtype={"ts_code": str, "symbol": str})
    lookup = candidates[["ts_code"]].merge(basic[["ts_code", "symbol", "name"]], on="ts_code", how="left")

    rows = []
    for _, stock in lookup.iterrows():
        code = str(stock["ts_code"])
        symbol = str(stock["symbol"])
        name = str(stock["name"])
        # Require exact code/symbol/name mention. Name-only matches can be ambiguous; keep titles visible for review.
        mask = (
            news["text"].str.contains(code, regex=False)
            | news["text"].str.contains(symbol, regex=False)
            | news["text"].str.contains(name, regex=False)
        )
        matched = news.loc[mask].copy()
        if matched.empty:
            rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "name": name,
                    "mentions": 0,
                    "positive_hits": 0,
                    "negative_hits": 0,
                    "keywords": "",
                    "titles": "",
                }
            )
            continue
        pos_total = 0
        neg_total = 0
        keywords = []
        for text in matched["text"]:
            pos, neg, words = keyword_score(str(text))
            pos_total += pos
            neg_total += neg
            if words:
                keywords.append(words)
        rows.append(
            {
                "trade_date": date,
                "ts_code": code,
                "name": name,
                "mentions": len(matched),
                "positive_hits": pos_total,
                "negative_hits": neg_total,
                "keywords": "|".join(keywords),
                "titles": " | ".join(matched["title"].head(5).astype(str).tolist()),
            }
        )

    out = pd.DataFrame(rows).sort_values(["negative_hits", "mentions"], ascending=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

