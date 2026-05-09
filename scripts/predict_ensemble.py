from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare.model import load_checkpoint, predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", default="outputs/features_2023.pkl")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--start-date", default="20260101")
    parser.add_argument("--end-date", default="20260420")
    parser.add_argument("--output", default="outputs/val_predictions_ensemble.csv")
    parser.add_argument("--latest-output", default="outputs/latest_candidates_ensemble.csv")
    parser.add_argument("--latest-top-n", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.feature_file).open("rb") as f:
        payload = pickle.load(f)

    df = payload["features"].copy()
    ret_col = f"future_ret_{payload['horizon']}"

    val = df.loc[
        (df["trade_date"] >= args.start_date)
        & (df["trade_date"] <= args.end_date)
        & df["is_tradable"]
        & df[ret_col].notna()
        & df["target"].notna()
    ].copy()
    latest_date = str(df["trade_date"].max())
    latest = df.loc[(df["trade_date"] == latest_date) & df["is_tradable"]].copy()

    val_scores = []
    latest_scores = []
    expected_features = None
    for checkpoint_path in args.checkpoints:
        model, scaler, feature_cols, _ = load_checkpoint(checkpoint_path)
        if expected_features is None:
            expected_features = feature_cols
        elif expected_features != feature_cols:
            raise ValueError(f"feature columns mismatch in {checkpoint_path}")
        val_scores.append(predict(model, scaler, val, feature_cols))
        latest_scores.append(predict(model, scaler, latest, feature_cols))

    for i, score in enumerate(val_scores, start=1):
        val[f"score_{i}"] = score
    for i, score in enumerate(latest_scores, start=1):
        latest[f"score_{i}"] = score

    score_cols = [f"score_{i}" for i in range(1, len(args.checkpoints) + 1)]
    val["score"] = val[score_cols].mean(axis=1)
    latest["score"] = latest[score_cols].mean(axis=1)

    val_cols = ["trade_date", "ts_code", "name", "industry", "close", "amount", "score", ret_col, "target"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    val[val_cols].sort_values(["trade_date", "score"], ascending=[True, False]).to_csv(
        args.output, index=False, encoding="utf-8-sig"
    )

    latest_cols = [
        "trade_date",
        "ts_code",
        "name",
        "industry",
        "market",
        "close",
        "amount",
        "ret_1",
        "score",
    ]
    latest = latest.sort_values("score", ascending=False).head(args.latest_top_n)
    latest[latest_cols].to_csv(args.latest_output, index=False, encoding="utf-8-sig")

    print(f"saved validation ensemble -> {args.output}")
    print(f"saved latest ensemble candidates for {latest_date} -> {args.latest_output}")
    print(latest[latest_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

