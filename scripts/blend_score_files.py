from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend score CSV files with daily rank normalization.")
    parser.add_argument("--score-files", nargs="+", required=True, help="Use name=path entries.")
    parser.add_argument("--weights", required=True, help="Comma-separated weights, e.g. old:0.8,live:0.2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--top-n", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names_paths = [parse_named_path(item) for item in args.score_files]
    weights = parse_weights(args.weights, [name for name, _ in names_paths])

    merged = None
    base_cols = []
    rank_cols = []
    raw_score_cols = []
    for name, path in names_paths:
        df = pd.read_csv(path, dtype={"trade_date": str, "ts_code": str})
        if args.score_col not in df.columns:
            raise ValueError(f"{path} missing score column: {args.score_col}")
        raw_col = f"score_{name}"
        rank_col = f"{raw_col}_rank"
        if merged is None:
            base_cols = [c for c in df.columns if c != args.score_col and not c.startswith("score_")]
            merged = df[base_cols + [args.score_col]].rename(columns={args.score_col: raw_col})
        else:
            score = df[["trade_date", "ts_code", args.score_col]].rename(columns={args.score_col: raw_col})
            merged = merged.merge(score, on=["trade_date", "ts_code"], how="inner")
        merged[rank_col] = merged.groupby("trade_date")[raw_col].rank(pct=True) - 0.5
        raw_score_cols.append(raw_col)
        rank_cols.append(rank_col)

    if merged is None or merged.empty:
        raise ValueError("No rows after score merge.")

    merged["score"] = 0.0
    for name, _ in names_paths:
        merged["score"] += weights[name] * merged[f"score_{name}_rank"]

    keep_cols = dedupe([*base_cols, "score", *raw_score_cols])
    out = merged[keep_cols].sort_values(["trade_date", "score"], ascending=[True, False])
    if args.top_n > 0:
        out = out.groupby("trade_date", group_keys=False).head(args.top_n)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"saved blended scores -> {output}")
    print(json.dumps({"weights": weights, "rows": len(out), "dates": int(out['trade_date'].nunique())}, ensure_ascii=False))
    print(out.head(20).to_string(index=False))


def parse_named_path(item: str) -> tuple[str, str]:
    if "=" not in item:
        path = Path(item)
        return path.stem, item
    name, path = item.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty score name in {item}")
    return name, path


def parse_weights(spec: str, names: list[str]) -> dict[str, float]:
    weights = {name: 0.0 for name in names}
    for part in spec.split(","):
        if not part.strip():
            continue
        if ":" not in part:
            raise ValueError(f"invalid weight part: {part}")
        name, value = part.split(":", 1)
        name = name.strip()
        if name not in weights:
            raise ValueError(f"unknown score name in weights: {name}")
        weights[name] = float(value)
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"weights must sum to 1.0, got {total}")
    return weights


def dedupe(cols: list[str]) -> list[str]:
    result = []
    seen = set()
    for col in cols:
        if col not in seen:
            result.append(col)
            seen.add(col)
    return result


if __name__ == "__main__":
    main()
