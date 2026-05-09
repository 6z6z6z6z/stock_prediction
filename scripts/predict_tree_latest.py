from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", default="outputs/features_2023.pkl")
    parser.add_argument("--model", default="outputs/tree_model.joblib")
    parser.add_argument("--output", default="outputs/latest_candidates_tree.csv")
    parser.add_argument("--top-n", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.feature_file).open("rb") as f:
        payload = pickle.load(f)
    saved = joblib.load(args.model)
    model = saved["model"]
    feature_cols = saved["feature_cols"]

    df = payload["features"].copy()
    latest_date = str(df["trade_date"].max())
    latest = df.loc[(df["trade_date"] == latest_date) & df["is_tradable"]].copy()
    latest["score"] = model.predict(latest[feature_cols])
    cols = ["trade_date", "ts_code", "name", "industry", "market", "close", "amount", "ret_1", "score"]
    latest = latest.sort_values("score", ascending=False).head(args.top_n)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    latest[cols].to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved top {len(latest)} candidates for {latest_date} -> {args.output}")
    print(latest[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

