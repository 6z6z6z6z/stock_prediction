from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare.backtest import rebalance_topk_backtest, rebalance_topk_backtest_diversified
from ashare.evaluate import daily_ic, summarize_ic, top_group_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--model-out", default="outputs/gbdt_sidecar.joblib")
    parser.add_argument("--pred-out", default="outputs/val_predictions_gbdt_sidecar.csv")
    parser.add_argument("--latest-output", default="outputs/latest_candidates_gbdt_sidecar.csv")
    parser.add_argument("--metrics-out", default="outputs/metrics_gbdt_sidecar.json")
    parser.add_argument("--backend", choices=["auto", "lightgbm", "sklearn"], default="auto")
    parser.add_argument("--train-end", default="20251231")
    parser.add_argument("--val-start", default="20260101")
    parser.add_argument("--val-end", default="20260420")
    parser.add_argument("--sample-size", type=int, default=1200000)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tail-weight", type=float, default=1.5)
    parser.add_argument("--top-quantile", type=float, default=0.90)
    parser.add_argument("--latest-top-n", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.feature_file).open("rb") as f:
        payload = pickle.load(f)
    df = payload["features"].copy()
    feature_cols = payload["feature_cols"]
    ret_col = f"future_ret_{payload['horizon']}"

    usable = df.loc[df["is_tradable"] & df["target"].notna() & df[ret_col].notna()].copy()
    train_df = usable.loc[usable["trade_date"] <= args.train_end].copy()
    val_df = usable.loc[(usable["trade_date"] >= args.val_start) & (usable["trade_date"] <= args.val_end)].copy()
    latest_date = str(df["trade_date"].max())
    latest_df = df.loc[(df["trade_date"] == latest_date) & df["is_tradable"]].copy()
    if args.sample_size > 0 and len(train_df) > args.sample_size:
        train_df = train_df.sample(args.sample_size, random_state=args.seed)

    backend = choose_backend(args.backend)
    if backend == "lightgbm":
        model = train_lightgbm_ranker(args, train_df, val_df, feature_cols)
    else:
        model = train_sklearn_gbdt(args, train_df, feature_cols)

    val_df["score"] = predict_model(model, val_df, feature_cols, backend)
    latest_df["score"] = predict_model(model, latest_df, feature_cols, backend)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "backend": backend,
            "model": model,
            "feature_cols": feature_cols,
            "horizon": payload["horizon"],
            "target_mode": payload.get("target_mode", ""),
        },
        args.model_out,
    )

    pred_cols = ["trade_date", "ts_code", "name", "industry", "market", "close", "amount", "ret_1", "score", ret_col, "target"]
    Path(args.pred_out).parent.mkdir(parents=True, exist_ok=True)
    val_df[[c for c in pred_cols if c in val_df.columns]].sort_values(
        ["trade_date", "score"], ascending=[True, False]
    ).to_csv(args.pred_out, index=False, encoding="utf-8-sig")

    latest_cols = ["trade_date", "ts_code", "name", "industry", "market", "close", "amount", "ret_1", "score"]
    latest_df = latest_df.sort_values("score", ascending=False).head(args.latest_top_n)
    latest_df[[c for c in latest_cols if c in latest_df.columns]].to_csv(
        args.latest_output, index=False, encoding="utf-8-sig"
    )

    strategy_rows = []
    for market_mode in ["no_star", "main_only"]:
        eval_df = val_df.loc[~val_df["market"].eq("科创板")].copy() if market_mode == "no_star" else val_df.loc[val_df["market"].eq("主板")].copy()
        eval_df = eval_df.loc[eval_df["amount"].fillna(0) >= 50000].copy()
        for n, k, cap in [(5, 1, 0), (5, 1, 1), (8, 1, 1), (10, 1, 1), (15, 1, 1)]:
            if cap:
                curve, metrics = rebalance_topk_backtest_diversified(
                    eval_df, "score", ret_col, n=n, k=k, max_per_industry=cap, cost=0.001
                )
            else:
                curve, metrics = rebalance_topk_backtest(eval_df, "score", ret_col, n=n, k=k, cost=0.001)
            strategy_rows.append(
                {
                    "market_mode": market_mode,
                    "n": n,
                    "k": k,
                    "industry_cap": cap,
                    **metrics,
                    "avg_turnover": float(curve["turnover"].mean()) if len(curve) else None,
                }
            )

    metrics = {
        "backend": backend,
        "rows": {"train": len(train_df), "val": len(val_df), "latest": len(latest_df)},
        "feature_file": str(Path(args.feature_file).resolve()),
        "target_mode": payload.get("target_mode", ""),
        "ic": summarize_ic(daily_ic(val_df, "score", ret_col)),
        "top_group_returns": top_group_returns(val_df, "score", ret_col, [5, 8, 10, 20, 50]),
        "strategy_grid": sorted(strategy_rows, key=lambda row: (row["total_return"], row["sharpe"]), reverse=True),
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(latest_df[[c for c in latest_cols if c in latest_df.columns]].head(10).to_string(index=False))


def choose_backend(requested: str) -> str:
    if requested in {"auto", "lightgbm"}:
        try:
            import lightgbm  # noqa: F401

            return "lightgbm"
        except ModuleNotFoundError:
            if requested == "lightgbm":
                raise
    return "sklearn"


def train_lightgbm_ranker(args: argparse.Namespace, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]):
    import lightgbm as lgb

    train_df = train_df.sort_values("trade_date")
    val_df = val_df.sort_values("trade_date")
    train_label = rank_labels(train_df)
    val_label = rank_labels(val_df)
    train_group = train_df.groupby("trade_date", sort=False).size().to_list()
    val_group = val_df.groupby("trade_date", sort=False).size().to_list()
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=args.max_iter,
        learning_rate=args.learning_rate,
        num_leaves=args.max_leaf_nodes,
        reg_lambda=args.l2,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(
        train_df[feature_cols],
        train_label,
        group=train_group,
        eval_set=[(val_df[feature_cols], val_label)],
        eval_group=[val_group],
        eval_at=[5, 8, 10],
    )
    return model


def train_sklearn_gbdt(args: argparse.Namespace, train_df: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    weights = sample_weights(train_df, top_quantile=args.top_quantile, tail_weight=args.tail_weight)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "gbdt",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=args.learning_rate,
                    max_iter=args.max_iter,
                    max_leaf_nodes=args.max_leaf_nodes,
                    l2_regularization=args.l2,
                    random_state=args.seed,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=30,
                ),
            ),
        ]
    )
    model.fit(train_df[feature_cols], train_df["target"], gbdt__sample_weight=weights)
    return model


def predict_model(model, df: pd.DataFrame, feature_cols: list[str], backend: str) -> np.ndarray:
    return model.predict(df[feature_cols])


def rank_labels(df: pd.DataFrame, bins: int = 31) -> pd.Series:
    rank = df.groupby("trade_date")["target"].rank(pct=True)
    return np.floor(rank * bins).clip(0, bins - 1).astype(int)


def sample_weights(df: pd.DataFrame, top_quantile: float, tail_weight: float) -> np.ndarray:
    ranks = df.groupby("trade_date")["target"].rank(pct=True)
    top = ranks >= top_quantile
    bottom = ranks <= (1.0 - top_quantile)
    weights = np.ones(len(df), dtype=np.float32)
    weights[top.to_numpy() | bottom.to_numpy()] = float(tail_weight)
    return weights


if __name__ == "__main__":
    main()
