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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare.backtest import rebalance_topk_backtest
from ashare.evaluate import daily_ic, summarize_ic, top_group_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", default="outputs/features_2023.pkl")
    parser.add_argument("--model-out", default="outputs/tree_model.joblib")
    parser.add_argument("--pred-out", default="outputs/val_predictions_tree.csv")
    parser.add_argument("--metrics-out", default="outputs/metrics_tree.json")
    parser.add_argument("--train-end", default="20251231")
    parser.add_argument("--val-start", default="20260101")
    parser.add_argument("--val-end", default="20260420")
    parser.add_argument("--sample-size", type=int, default=800000)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--seed", type=int, default=2026)
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
    if args.sample_size > 0 and len(train_df) > args.sample_size:
        train_df = train_df.sample(args.sample_size, random_state=args.seed)

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "gbrt",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=args.learning_rate,
                    max_iter=args.max_iter,
                    max_leaf_nodes=args.max_leaf_nodes,
                    l2_regularization=0.05,
                    random_state=args.seed,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=25,
                ),
            ),
        ]
    )
    model.fit(train_df[feature_cols], train_df["target"])
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols, "horizon": payload["horizon"]}, args.model_out)

    val_df["score"] = model.predict(val_df[feature_cols])
    pred_cols = ["trade_date", "ts_code", "name", "industry", "close", "amount", "score", ret_col, "target"]
    Path(args.pred_out).parent.mkdir(parents=True, exist_ok=True)
    val_df[pred_cols].sort_values(["trade_date", "score"], ascending=[True, False]).to_csv(
        args.pred_out, index=False, encoding="utf-8-sig"
    )

    rows = []
    for n, k in [(5, 1), (10, 1), (20, 1)]:
        curve, m = rebalance_topk_backtest(val_df, "score", ret_col, n=n, k=k, cost=0.001)
        rows.append({"n": n, "k": k, **m, "avg_turnover": curve["turnover"].mean() if len(curve) else np.nan})

    metrics = {
        "rows": {"train": len(train_df), "val": len(val_df)},
        "ic": summarize_ic(daily_ic(val_df, "score", ret_col)),
        "top_group_returns": top_group_returns(val_df, "score", ret_col, [5, 10, 20, 50]),
        "strategy_grid": rows,
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

