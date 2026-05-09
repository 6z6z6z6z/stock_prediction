from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../A股数据")
    parser.add_argument("--start-date", default="20230101")
    parser.add_argument("--output-dir", default="outputs/daily")
    parser.add_argument("--checkpoints", nargs="+", default=[
        "outputs/mlp_checkpoint_2023.pt",
        "outputs/mlp_checkpoint_2023_seed7.pt",
        "outputs/mlp_checkpoint_2023_seed42.pt",
    ])
    parser.add_argument("--holdings", default="")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-per-industry", type=int, default=1)
    parser.add_argument("--min-amount", type=float, default=50000.0)
    parser.add_argument("--market-mode", choices=["all", "no_star", "main_only"], default="no_star")
    return parser.parse_args()


def latest_daily_date(data_dir: Path) -> str:
    daily_dir = data_dir / "daily"
    files = sorted(daily_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No daily csv files found in {daily_dir}")
    return files[-1].stem


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    end_date = latest_daily_date(data_dir)

    feature_file = out_dir / f"features_{end_date}.pkl"
    val_pred = out_dir / f"val_predictions_ensemble_{end_date}.csv"
    latest_candidates = out_dir / f"latest_candidates_ensemble_{end_date}.csv"
    filtered_candidates = out_dir / f"latest_candidates_filtered_{end_date}.csv"
    plan = out_dir / f"rebalance_plan_{end_date}.csv"

    run([
        sys.executable,
        "scripts/prepare_features.py",
        "--data-dir",
        str(data_dir),
        "--start-date",
        args.start_date,
        "--end-date",
        end_date,
        "--target-mode",
        "close",
        "--output",
        str(feature_file),
    ])

    run([
        sys.executable,
        "scripts/predict_ensemble.py",
        "--feature-file",
        str(feature_file),
        "--checkpoints",
        *args.checkpoints,
        "--output",
        str(val_pred),
        "--latest-output",
        str(latest_candidates),
        "--latest-top-n",
        "120",
    ])

    run([
        sys.executable,
        "scripts/filter_candidates.py",
        "--candidates",
        str(latest_candidates),
        "--output",
        str(filtered_candidates),
        "--min-amount",
        str(args.min_amount),
        "--market-mode",
        args.market_mode,
        "--ret-mode",
        "all",
        "--top-n",
        "120",
    ])

    plan_cmd = [
        sys.executable,
        "scripts/make_diversified_plan.py",
        "--candidates",
        str(filtered_candidates),
        "--output",
        str(plan),
        "--n",
        str(args.n),
        "--k",
        str(args.k),
        "--max-per-industry",
        str(args.max_per_industry),
    ]
    if args.holdings:
        plan_cmd.extend(["--holdings", args.holdings])
    run(plan_cmd)
    print(f"daily pipeline complete for {end_date}: {plan}")


if __name__ == "__main__":
    main()

