from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target-mode", choices=["close", "open2open", "intraday"], required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.input).open("rb") as f:
        payload = pickle.load(f)

    df = payload["features"]
    horizon = int(payload.get("horizon", 1))
    source_col = f"future_ret_{horizon}_{args.target_mode}"
    target_col = f"future_ret_{horizon}"
    if source_col not in df.columns:
        raise ValueError(f"missing source target column: {source_col}")

    df[target_col] = df[source_col]
    target_raw = df[source_col] - df.groupby("trade_date")[source_col].transform("mean")
    lower = target_raw.groupby(df["trade_date"]).transform(lambda s: s.quantile(0.01))
    upper = target_raw.groupby(df["trade_date"]).transform(lambda s: s.quantile(0.99))
    clipped = target_raw.clip(lower=lower, upper=upper)
    mean = clipped.groupby(df["trade_date"]).transform("mean")
    std = clipped.groupby(df["trade_date"]).transform("std")
    df["target"] = ((clipped - mean) / (std + EPS)).replace([np.inf, -np.inf], np.nan).astype(np.float32)

    payload["features"] = df
    payload["target_mode"] = args.target_mode
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"retargeted {len(df):,} rows to {args.target_mode} -> {output}")


if __name__ == "__main__":
    main()
