from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path
    start_date: str
    end_date: str
    use_metric: bool = True
    use_moneyflow: bool = True
    use_market: bool = True


def list_date_files(folder: Path, start_date: str, end_date: str) -> list[Path]:
    if not folder.exists():
        return []
    files: list[Path] = []
    for path in folder.glob("*.csv"):
        date = path.stem
        if start_date <= date <= end_date:
            files.append(path)
    return sorted(files, key=lambda p: p.stem)


def load_basic(data_dir: Path) -> pd.DataFrame:
    basic = pd.read_csv(data_dir / "basic.csv", dtype={"ts_code": str, "list_date": str})
    basic["is_st"] = basic["name"].astype(str).str.contains("ST", case=False, na=False)
    basic["is_bj"] = basic["market"].astype(str).eq("北交所")
    return basic


def filtered_universe(data_dir: Path) -> pd.DataFrame:
    basic = load_basic(data_dir)
    return basic.loc[~basic["is_st"] & ~basic["is_bj"]].copy()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"ts_code": str, "trade_date": str})


def read_panel(config: DataConfig) -> pd.DataFrame:
    data_dir = Path(config.data_dir)
    daily_files = list_date_files(data_dir / "daily", config.start_date, config.end_date)
    if not daily_files:
        raise FileNotFoundError(f"No daily csv files found in {data_dir / 'daily'}")

    universe = filtered_universe(data_dir)[["ts_code", "name", "industry", "market", "list_date"]]
    market_features = load_market_features(data_dir, config.start_date, config.end_date) if config.use_market else None
    frames: list[pd.DataFrame] = []

    for daily_path in daily_files:
        df = _read_csv(daily_path)
        df = df.merge(universe, on="ts_code", how="inner")

        if config.use_metric:
            metric_path = data_dir / "metric" / daily_path.name
            if metric_path.exists():
                metric = _read_csv(metric_path)
                drop_cols = [c for c in ["close"] if c in metric.columns]
                metric = metric.drop(columns=drop_cols)
                df = df.merge(metric, on=["ts_code", "trade_date"], how="left")

        if config.use_moneyflow:
            money_path = data_dir / "moneyflow" / daily_path.name
            if money_path.exists():
                money = _read_csv(money_path)
                df = df.merge(money, on=["ts_code", "trade_date"], how="left")

        if market_features is not None:
            df = df.merge(market_features, on="trade_date", how="left")

        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel["trade_date"] = panel["trade_date"].astype(str)
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return panel


def load_market_features(data_dir: Path, start_date: str, end_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    mapping = {
        "000001.SH.csv": "sse",
        "000300.SH.csv": "hs300",
        "399006.SZ.csv": "chinext",
    }
    for filename, prefix in mapping.items():
        path = data_dir / "market" / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype={"trade_date": str})
        df = df.loc[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)].copy()
        df = df.sort_values("trade_date")
        for col in ["open", "high", "low", "close", "pre_close", "vol", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"{prefix}_ret_1"] = df["close"] / df["pre_close"] - 1.0
        for window in [3, 5, 10, 20]:
            df[f"{prefix}_ret_{window}"] = df["close"] / df["close"].shift(window) - 1.0
        df[f"{prefix}_volatility_20"] = df[f"{prefix}_ret_1"].rolling(20, min_periods=20).std()
        df[f"{prefix}_amount_ratio_5"] = df["amount"] / df["amount"].rolling(5, min_periods=5).mean() - 1.0
        keep = ["trade_date"] + [c for c in df.columns if c.startswith(prefix + "_")]
        frames.append(df[keep])
    if not frames:
        return pd.DataFrame({"trade_date": []})
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="trade_date", how="outer")
    return result.sort_values("trade_date")
