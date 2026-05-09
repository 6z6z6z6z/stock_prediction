from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily pipeline for old/live blended ensemble.")
    parser.add_argument("--data-dir", default="../A股数据")
    parser.add_argument("--start-date", default="20230101")
    parser.add_argument("--output-dir", default="outputs/daily_blend")
    parser.add_argument("--old-checkpoints", nargs="+", default=[
        "outputs/mlp_checkpoint_2023.pt",
        "outputs/mlp_checkpoint_2023_seed7.pt",
        "outputs/mlp_checkpoint_2023_seed42.pt",
    ])
    parser.add_argument(
        "--live-checkpoints",
        nargs="+",
        default=None,
        help="Live ensemble checkpoints. Defaults to the newest outputs/live_ensemble_YYYYMMDD directory.",
    )
    parser.add_argument("--weights", default="old:0.9,live:0.1")
    parser.add_argument("--val-start", default="20260402")
    parser.add_argument("--val-end", default="20260430")
    parser.add_argument("--holdings", default="")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-per-industry", type=int, default=1)
    parser.add_argument("--min-amount", type=float, default=50000.0)
    parser.add_argument("--market-mode", choices=["all", "no_star", "main_only"], default="no_star")
    parser.add_argument("--ret-mode", choices=["all", "no_large_drop", "no_large_move", "not_red_hot"], default="all")
    parser.add_argument("--latest-top-n", type=int, default=5000)
    return parser.parse_args()


def latest_daily_date(data_dir: Path) -> str:
    daily_dir = data_dir / "daily"
    files = sorted(daily_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No daily csv files found in {daily_dir}")
    return files[-1].stem


def discover_latest_live_checkpoints(outputs_root: Path = Path("outputs")) -> list[str]:
    live_dirs = []
    for path in outputs_root.glob("live_ensemble_*"):
        suffix = path.name.removeprefix("live_ensemble_")
        if path.is_dir() and suffix.isdigit():
            live_dirs.append(path)
    live_dirs.sort(key=lambda path: path.name, reverse=True)

    preferred_seeds = [2026, 7, 42]
    for live_dir in live_dirs:
        preferred = [live_dir / f"live_close_seed{seed}.pt" for seed in preferred_seeds]
        if all(path.exists() for path in preferred):
            return [str(path) for path in preferred]

        found = sorted(live_dir.glob("live_close_seed*.pt"))
        if found:
            return [str(path) for path in found]

    raise FileNotFoundError(
        "No live ensemble checkpoints found. Run scripts/train_live_ensemble.py first "
        "or pass --live-checkpoints explicitly."
    )


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    end_date = latest_daily_date(data_dir)
    live_checkpoints = args.live_checkpoints or discover_latest_live_checkpoints()

    feature_file = out_dir / f"features_{end_date}.pkl"
    old_val = out_dir / f"old_validation_{end_date}.csv"
    old_latest = out_dir / f"old_latest_{end_date}.csv"
    live_val = out_dir / f"live_validation_{end_date}.csv"
    live_latest = out_dir / f"live_latest_{end_date}.csv"
    blend_latest = out_dir / f"blend_latest_{end_date}.csv"
    filtered = out_dir / f"blend_filtered_{end_date}.csv"
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

    common_predict_args = [
        "--feature-file",
        str(feature_file),
        "--start-date",
        args.val_start,
        "--end-date",
        args.val_end,
        "--latest-top-n",
        str(args.latest_top_n),
    ]
    run([
        sys.executable,
        "scripts/predict_ensemble.py",
        *common_predict_args,
        "--checkpoints",
        *args.old_checkpoints,
        "--output",
        str(old_val),
        "--latest-output",
        str(old_latest),
    ])
    run([
        sys.executable,
        "scripts/predict_ensemble.py",
        *common_predict_args,
        "--checkpoints",
        *live_checkpoints,
        "--output",
        str(live_val),
        "--latest-output",
        str(live_latest),
    ])
    run([
        sys.executable,
        "scripts/blend_score_files.py",
        "--score-files",
        f"old={old_latest}",
        f"live={live_latest}",
        "--weights",
        args.weights,
        "--output",
        str(blend_latest),
        "--top-n",
        str(args.latest_top_n),
    ])
    run([
        sys.executable,
        "scripts/filter_candidates.py",
        "--candidates",
        str(blend_latest),
        "--output",
        str(filtered),
        "--min-amount",
        str(args.min_amount),
        "--market-mode",
        args.market_mode,
        "--ret-mode",
        args.ret_mode,
        "--top-n",
        "120",
    ])
    plan_cmd = [
        sys.executable,
        "scripts/make_diversified_plan.py",
        "--candidates",
        str(filtered),
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
    print(f"blended daily pipeline complete for {end_date}: {plan}")


if __name__ == "__main__":
    main()
