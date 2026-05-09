from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fills", required=True)
    parser.add_argument("--log", default="outputs/fill_log.csv")
    parser.add_argument("--summary", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fills = pd.read_csv(args.fills, dtype={"trade_date": str, "ts_code": str})
    required = {"trade_date", "ts_code", "action", "filled_price", "filled_shares", "status"}
    missing = sorted(required - set(fills.columns))
    if missing:
        raise ValueError(f"fills file missing columns: {missing}")

    fills["filled_amount"] = pd.to_numeric(fills.get("filled_amount", 0), errors="coerce").fillna(
        pd.to_numeric(fills["filled_price"], errors="coerce").fillna(0)
        * pd.to_numeric(fills["filled_shares"], errors="coerce").fillna(0)
    )
    if args.summary:
        fills["agent_summary"] = args.summary

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        old = pd.read_csv(log_path, dtype=str)
        for col in fills.columns:
            if col not in old.columns:
                old[col] = ""
        for col in old.columns:
            if col not in fills.columns:
                fills[col] = ""
        out = pd.concat([old, fills[old.columns]], ignore_index=True)
    else:
        out = fills
    out.to_csv(log_path, index=False, encoding="utf-8-sig")
    print(f"recorded {len(fills)} fill rows -> {log_path}")
    print(fills.to_string(index=False))


if __name__ == "__main__":
    main()
