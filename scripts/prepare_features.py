from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare.data import DataConfig, read_panel
from ashare.features import add_features, tradable_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../A股数据")
    parser.add_argument("--start-date", default="20240101")
    parser.add_argument("--end-date", default="20260428")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--target-mode", choices=["close", "open2open", "intraday"], default="close")
    parser.add_argument("--output", default="outputs/features.pkl")
    parser.add_argument("--no-metric", action="store_true")
    parser.add_argument("--no-moneyflow", action="store_true")
    parser.add_argument("--no-market", action="store_true")
    parser.add_argument("--min-amount", type=float, default=50000.0)
    parser.add_argument("--min-price", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DataConfig(
        data_dir=Path(args.data_dir),
        start_date=args.start_date,
        end_date=args.end_date,
        use_metric=not args.no_metric,
        use_moneyflow=not args.no_moneyflow,
        use_market=not args.no_market,
    )
    panel = read_panel(config)
    features, feature_cols = add_features(panel, horizon=args.horizon, target_mode=args.target_mode)
    features["is_tradable"] = tradable_filter(features, min_amount=args.min_amount, min_price=args.min_price)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(
            {
                "features": features,
                "feature_cols": feature_cols,
                "horizon": args.horizon,
                "target_mode": args.target_mode,
                "data_dir": str(Path(args.data_dir).resolve()),
                "start_date": args.start_date,
                "end_date": args.end_date,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"saved {len(features):,} rows, {len(feature_cols)} features -> {output}")


if __name__ == "__main__":
    main()
