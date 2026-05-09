from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare.model import load_checkpoint, predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", default="outputs/features.pkl")
    parser.add_argument("--checkpoint", default="outputs/mlp_checkpoint.pt")
    parser.add_argument("--date", default="")
    parser.add_argument("--output", default="outputs/latest_candidates.csv")
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.feature_file).open("rb") as f:
        payload = pickle.load(f)

    df = payload["features"].copy()
    model, scaler, feature_cols, _ = load_checkpoint(args.checkpoint)

    date = args.date or str(df["trade_date"].max())
    latest = df.loc[(df["trade_date"] == date) & df["is_tradable"]].copy()
    if latest.empty:
        raise ValueError(f"no tradable rows for date {date}")

    latest["score"] = predict(model, scaler, latest, feature_cols)
    output_cols = [
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
    latest = latest.sort_values("score", ascending=False).head(args.top_n)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    latest[output_cols].to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved top {len(latest)} candidates for {date} -> {args.output}")
    print(latest[output_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

