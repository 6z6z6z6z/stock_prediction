from __future__ import annotations

import argparse
import json
import pickle
import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare.backtest import rebalance_topk_backtest, rebalance_topk_backtest_diversified
from ashare.evaluate import daily_ic, summarize_ic, top_group_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--score-files", nargs="+", required=True, help="Use name=path entries.")
    parser.add_argument("--output", default="outputs/score_blend_grid.csv")
    parser.add_argument("--start-date", default="20260101")
    parser.add_argument("--end-date", default="20260420")
    parser.add_argument("--eval-ret-mode", choices=["payload", "close", "open2open", "intraday"], default="payload")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument(
        "--weight-specs",
        nargs="*",
        default=[],
        help="Optional explicit blends, e.g. ensemble:0.8,open:0.2 tree:1.0. If set, skips grid generation.",
    )
    parser.add_argument("--market-mode", choices=["all", "no_star", "main_only"], default="no_star")
    parser.add_argument("--ret-mode", choices=["all", "no_large_drop", "no_large_move", "not_red_hot"], default="all")
    parser.add_argument("--min-amount", type=float, default=50000.0)
    parser.add_argument("--ns", nargs="+", type=int, default=[5, 8, 10])
    parser.add_argument("--ks", nargs="+", type=int, default=[1])
    parser.add_argument("--caps", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--cost", type=float, default=0.001)
    parser.add_argument("--top-results", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names_paths = [parse_named_path(item) for item in args.score_files]
    if len(names_paths) < 2:
        raise ValueError("at least two score files are required")

    with Path(args.feature_file).open("rb") as f:
        payload = pickle.load(f)
    features = payload["features"]
    ret_col = resolve_ret_col(features, int(payload["horizon"]), args.eval_ret_mode)
    base_cols = [
        "trade_date",
        "ts_code",
        "name",
        "industry",
        "market",
        "amount",
        "ret_1",
        "is_tradable",
        ret_col,
    ]
    df = features.loc[
        (features["trade_date"] >= args.start_date)
        & (features["trade_date"] <= args.end_date)
        & features["is_tradable"]
        & features[ret_col].notna(),
        [c for c in base_cols if c in features.columns],
    ].copy()

    score_cols = []
    for name, path in names_paths:
        col = f"score_{name}"
        score = pd.read_csv(path, dtype={"trade_date": str, "ts_code": str})
        score = score[["trade_date", "ts_code", args.score_col]].rename(columns={args.score_col: col})
        df = df.merge(score, on=["trade_date", "ts_code"], how="inner")
        rank_col = f"{col}_rank"
        df[rank_col] = df.groupby("trade_date")[col].rank(pct=True) - 0.5
        score_cols.append(rank_col)

    df = apply_live_filters(df, args.market_mode, args.ret_mode, args.min_amount)
    if df.empty:
        raise ValueError("empty evaluation frame after score merge and filters")

    rows = []
    all_weights = explicit_weight_specs(args.weight_specs, [name for name, _ in names_paths])
    if not all_weights:
        all_weights = weight_grid(len(score_cols), args.weight_step)
    for weights in all_weights:
        blend_name = "+".join(f"{name}:{weight:.2f}" for (name, _), weight in zip(names_paths, weights))
        score = sum(weight * df[col] for col, weight in zip(score_cols, weights))
        eval_df = df.copy()
        eval_df["blend_score"] = score
        ic = summarize_ic(daily_ic(eval_df, "blend_score", ret_col))
        top = top_group_returns(eval_df, "blend_score", ret_col, [5, 8, 10, 20])
        for n in args.ns:
            for k in args.ks:
                if k >= n:
                    continue
                for cap in args.caps:
                    if cap > 0:
                        curve, metrics = rebalance_topk_backtest_diversified(
                            eval_df,
                            "blend_score",
                            ret_col,
                            n=n,
                            k=k,
                            max_per_industry=cap,
                            cost=args.cost,
                        )
                    else:
                        curve, metrics = rebalance_topk_backtest(
                            eval_df,
                            "blend_score",
                            ret_col,
                            n=n,
                            k=k,
                            cost=args.cost,
                        )
                    row = {
                        "blend": blend_name,
                        "weights_json": json.dumps(
                            {name: weight for (name, _), weight in zip(names_paths, weights)},
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "market_mode": args.market_mode,
                        "ret_mode": args.ret_mode,
                        "industry_cap": cap,
                        "n": n,
                        "k": k,
                        **ic,
                        **top,
                        **metrics,
                        "avg_turnover": float(curve["turnover"].mean()) if len(curve) else None,
                    }
                    rows.append(row)

    out = pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(out.head(args.top_results).to_string(index=False))


def parse_named_path(item: str) -> tuple[str, str]:
    if "=" not in item:
        path = Path(item)
        return path.stem, item
    name, path = item.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty score name in {item}")
    return name, path


def resolve_ret_col(features: pd.DataFrame, horizon: int, mode: str) -> str:
    col = f"future_ret_{horizon}" if mode == "payload" else f"future_ret_{horizon}_{mode}"
    if col not in features.columns:
        raise ValueError(f"Missing return column: {col}")
    return col


def weight_grid(n: int, step: float) -> list[tuple[float, ...]]:
    if step <= 0 or step > 1:
        raise ValueError("--weight-step must be in (0, 1]")
    denom = round(1.0 / step)
    if abs(denom * step - 1.0) > 1e-8:
        raise ValueError("--weight-step must divide 1.0 exactly, e.g. 0.1, 0.2, 0.25")
    combos = []
    for parts in product(range(denom + 1), repeat=n):
        if sum(parts) == denom:
            combos.append(tuple(part / denom for part in parts))
    return combos


def explicit_weight_specs(specs: list[str], names: list[str]) -> list[tuple[float, ...]]:
    if not specs:
        return []
    rows = []
    for spec in specs:
        mapping = {name: 0.0 for name in names}
        for part in spec.split(","):
            if not part.strip():
                continue
            if ":" not in part:
                raise ValueError(f"invalid weight spec part: {part}")
            name, value = part.split(":", 1)
            name = name.strip()
            if name not in mapping:
                raise ValueError(f"unknown model name in weight spec: {name}")
            mapping[name] = float(value)
        total = sum(mapping.values())
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"weights must sum to 1.0, got {total}: {spec}")
        rows.append(tuple(mapping[name] for name in names))
    return rows


def apply_live_filters(df: pd.DataFrame, market_mode: str, ret_mode: str, min_amount: float) -> pd.DataFrame:
    out = df.loc[df["amount"].fillna(0) >= min_amount].copy()
    if market_mode == "no_star":
        out = out.loc[~out["market"].eq("科创板")].copy()
    elif market_mode == "main_only":
        out = out.loc[out["market"].eq("主板")].copy()
    if "ret_1" in out.columns:
        if ret_mode == "no_large_drop":
            out = out.loc[out["ret_1"].fillna(0) > -0.07].copy()
        elif ret_mode == "no_large_move":
            out = out.loc[out["ret_1"].abs().fillna(0) < 0.07].copy()
        elif ret_mode == "not_red_hot":
            out = out.loc[out["ret_1"].fillna(0) < 0.05].copy()
    return out


if __name__ == "__main__":
    main()
