from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare.backtest import rebalance_topk_backtest, rebalance_topk_backtest_diversified
from ashare.evaluate import daily_ic, summarize_ic, top_group_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--base-score-file", required=True)
    parser.add_argument("--aux-score-file", required=True)
    parser.add_argument("--output", default="outputs/aux_filter_grid.csv")
    parser.add_argument("--start-date", default="20260101")
    parser.add_argument("--end-date", default="20260420")
    parser.add_argument("--base-score-col", default="score")
    parser.add_argument("--aux-score-col", default="score")
    parser.add_argument("--aux-min-ranks", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--market-mode", choices=["all", "no_star", "main_only"], default="no_star")
    parser.add_argument("--min-amount", type=float, default=50000.0)
    parser.add_argument("--ns", nargs="+", type=int, default=[5, 8, 10])
    parser.add_argument("--ks", nargs="+", type=int, default=[1])
    parser.add_argument("--caps", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--cost", type=float, default=0.001)
    parser.add_argument("--top-results", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.feature_file).open("rb") as f:
        payload = pickle.load(f)
    features = payload["features"]
    ret_col = f"future_ret_{payload['horizon']}"
    cols = ["trade_date", "ts_code", "name", "industry", "market", "amount", "is_tradable", ret_col]
    df = features.loc[
        (features["trade_date"] >= args.start_date)
        & (features["trade_date"] <= args.end_date)
        & features["is_tradable"]
        & features[ret_col].notna(),
        [c for c in cols if c in features.columns],
    ].copy()

    base = pd.read_csv(args.base_score_file, dtype={"trade_date": str, "ts_code": str})
    base = base[["trade_date", "ts_code", args.base_score_col]].rename(columns={args.base_score_col: "base_score"})
    aux = pd.read_csv(args.aux_score_file, dtype={"trade_date": str, "ts_code": str})
    aux = aux[["trade_date", "ts_code", args.aux_score_col]].rename(columns={args.aux_score_col: "aux_score"})
    df = df.merge(base, on=["trade_date", "ts_code"], how="inner").merge(aux, on=["trade_date", "ts_code"], how="inner")
    df["aux_rank"] = df.groupby("trade_date")["aux_score"].rank(pct=True)
    df = apply_live_filters(df, args.market_mode, args.min_amount)

    rows = []
    for aux_min_rank in args.aux_min_ranks:
        filtered = df.loc[df["aux_rank"] >= aux_min_rank].copy()
        if filtered.empty:
            continue
        ic = summarize_ic(daily_ic(filtered, "base_score", ret_col))
        top = top_group_returns(filtered, "base_score", ret_col, [5, 8, 10, 20])
        per_day_min = filtered.groupby("trade_date")["ts_code"].size().min()
        for n in args.ns:
            if per_day_min < n:
                continue
            for k in args.ks:
                if k >= n:
                    continue
                for cap in args.caps:
                    if cap > 0:
                        curve, metrics = rebalance_topk_backtest_diversified(
                            filtered,
                            "base_score",
                            ret_col,
                            n=n,
                            k=k,
                            max_per_industry=cap,
                            cost=args.cost,
                        )
                    else:
                        curve, metrics = rebalance_topk_backtest(
                            filtered,
                            "base_score",
                            ret_col,
                            n=n,
                            k=k,
                            cost=args.cost,
                        )
                    rows.append(
                        {
                            "aux_min_rank": aux_min_rank,
                            "market_mode": args.market_mode,
                            "industry_cap": cap,
                            "n": n,
                            "k": k,
                            "rows": len(filtered),
                            "min_candidates_per_day": int(per_day_min),
                            **ic,
                            **top,
                            **metrics,
                            "avg_turnover": float(curve["turnover"].mean()) if len(curve) else None,
                        }
                    )

    out = pd.DataFrame(rows).sort_values(["annual_return", "sharpe"], ascending=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    meta = {
        "feature_file": str(Path(args.feature_file).resolve()),
        "base_score_file": str(Path(args.base_score_file).resolve()),
        "aux_score_file": str(Path(args.aux_score_file).resolve()),
        "ret_col": ret_col,
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out.head(args.top_results).to_string(index=False))


def apply_live_filters(df: pd.DataFrame, market_mode: str, min_amount: float) -> pd.DataFrame:
    out = df.loc[df["amount"].fillna(0) >= min_amount].copy()
    if market_mode == "no_star":
        out = out.loc[~out["market"].eq("科创板")].copy()
    elif market_mode == "main_only":
        out = out.loc[out["market"].eq("主板")].copy()
    return out


if __name__ == "__main__":
    main()
