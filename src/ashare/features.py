from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-12


def _roll(grouped: pd.core.groupby.SeriesGroupBy, window: int, func: str) -> pd.Series:
    return grouped.transform(lambda s: getattr(s.rolling(window, min_periods=window), func)())


def _future_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


def add_features(panel: pd.DataFrame, horizon: int = 1, target_mode: str = "close") -> tuple[pd.DataFrame, list[str]]:
    df = panel.copy()
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "vol",
        "amount",
        "vwap",
    ]
    optional_numeric = [
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "total_mv",
        "circ_mv",
        "net_mf_amount",
        "buy_lg_amount",
        "sell_lg_amount",
        "buy_elg_amount",
        "sell_elg_amount",
        "buy_sm_amount",
        "sell_sm_amount",
    ]
    optional_numeric += [
        col
        for col in df.columns
        if col.startswith(("sse_", "hs300_", "chinext_"))
    ]
    for col in numeric_cols + optional_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    g = df.groupby("ts_code", group_keys=False)
    close_g = g["close"]
    ret = df["close"] / df["pre_close"] - 1.0
    df["ret_1"] = ret.replace([np.inf, -np.inf], np.nan)

    for window in [2, 3, 5, 10, 20, 60]:
        df[f"ret_{window}"] = close_g.transform(lambda s, w=window: s / s.shift(w) - 1.0)

    for window in [5, 10, 20, 60]:
        ma = _roll(close_g, window, "mean")
        df[f"ma_dev_{window}"] = df["close"] / (ma + EPS) - 1.0

    ret_g = df.groupby("ts_code", group_keys=False)["ret_1"]
    for window in [5, 10, 20]:
        df[f"volatility_{window}"] = _roll(ret_g, window, "std")

    vol_g = g["vol"]
    amount_g = g["amount"]
    for window in [5, 20]:
        vol_ma = _roll(vol_g, window, "mean")
        amount_ma = _roll(amount_g, window, "mean")
        df[f"vol_ratio_{window}"] = df["vol"] / (vol_ma + EPS) - 1.0
        df[f"amount_ratio_{window}"] = df["amount"] / (amount_ma + EPS) - 1.0

    df["amplitude"] = df["high"] / (df["low"] + EPS) - 1.0
    df["gap_ret"] = df["open"] / (df["pre_close"] + EPS) - 1.0
    df["gap_abs"] = df["gap_ret"].abs()
    df["oc_ret"] = df["close"] / (df["open"] + EPS) - 1.0
    df["vwap_dev"] = df["close"] / (df["vwap"] + EPS) - 1.0
    df["close_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + EPS)
    df["candle_body"] = (df["close"] - df["open"]).abs() / (df["pre_close"] + EPS)
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["pre_close"] + EPS)
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["pre_close"] + EPS)
    df["high_close_gap"] = df["high"] / (df["close"] + EPS) - 1.0
    df["close_low_gap"] = df["close"] / (df["low"] + EPS) - 1.0
    df["log_amount"] = np.log1p(df["amount"].clip(lower=0))
    df["log_vol"] = np.log1p(df["vol"].clip(lower=0))

    for window in [3, 5, 10, 20]:
        df[f"amplitude_mean_{window}"] = _roll(g["amplitude"], window, "mean")
        df[f"close_pos_mean_{window}"] = _roll(g["close_pos"], window, "mean")
        df[f"candle_body_mean_{window}"] = _roll(g["candle_body"], window, "mean")
        df[f"gap_abs_mean_{window}"] = _roll(g["gap_abs"], window, "mean")

    if "net_mf_amount" in df.columns:
        amount_10k = df["amount"] / 10.0
        df["net_mf_ratio"] = df["net_mf_amount"] / (amount_10k.abs() + EPS)
        net_mf_g = df.groupby("ts_code", group_keys=False)["net_mf_ratio"]
        for window in [3, 5, 10, 20]:
            df[f"net_mf_ratio_sum_{window}"] = _roll(net_mf_g, window, "sum")
            df[f"net_mf_ratio_mean_{window}"] = _roll(net_mf_g, window, "mean")

    if {"buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount"}.issubset(df.columns):
        amount_10k = df["amount"] / 10.0
        big_net = (
            df["buy_lg_amount"].fillna(0)
            + df["buy_elg_amount"].fillna(0)
            - df["sell_lg_amount"].fillna(0)
            - df["sell_elg_amount"].fillna(0)
        )
        df["big_order_net_ratio"] = big_net / (amount_10k.abs() + EPS)
        big_g = df.groupby("ts_code", group_keys=False)["big_order_net_ratio"]
        for window in [3, 5, 10, 20]:
            df[f"big_order_net_ratio_sum_{window}"] = _roll(big_g, window, "sum")
            df[f"big_order_net_ratio_mean_{window}"] = _roll(big_g, window, "mean")

    if {"buy_sm_amount", "sell_sm_amount"}.issubset(df.columns):
        amount_10k = df["amount"] / 10.0
        df["small_order_net_ratio"] = (
            df["buy_sm_amount"].fillna(0) - df["sell_sm_amount"].fillna(0)
        ) / (amount_10k.abs() + EPS)
        small_g = df.groupby("ts_code", group_keys=False)["small_order_net_ratio"]
        for window in [3, 5, 10]:
            df[f"small_order_net_ratio_sum_{window}"] = _roll(small_g, window, "sum")

    feature_cols = [
        c
        for c in df.columns
        if c.startswith(
            (
                "ret_",
                "ma_dev_",
                "volatility_",
                "vol_ratio_",
                "amount_ratio_",
            )
        )
    ]
    feature_cols += [
        c
        for c in [
            "amplitude",
            "gap_ret",
            "gap_abs",
            "oc_ret",
            "vwap_dev",
            "close_pos",
            "candle_body",
            "upper_shadow",
            "lower_shadow",
            "high_close_gap",
            "close_low_gap",
            "log_amount",
            "log_vol",
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "pe",
            "pe_ttm",
            "pb",
            "ps",
            "ps_ttm",
            "total_mv",
            "circ_mv",
            "net_mf_ratio",
            "big_order_net_ratio",
            "small_order_net_ratio",
        ]
        if c in df.columns
    ]
    feature_cols += [
        c
        for c in df.columns
        if c.startswith(
            (
                "amplitude_mean_",
                "close_pos_mean_",
                "candle_body_mean_",
                "gap_abs_mean_",
                "net_mf_ratio_sum_",
                "net_mf_ratio_mean_",
                "big_order_net_ratio_sum_",
                "big_order_net_ratio_mean_",
                "small_order_net_ratio_sum_",
            )
        )
    ]
    feature_cols += [
        c
        for c in df.columns
        if c.startswith(("sse_", "hs300_", "chinext_"))
    ]

    rank_base = [
        c
        for c in [
            "ret_1",
            "gap_ret",
            "amplitude",
            "close_pos",
            "ret_5",
            "ret_20",
            "volatility_20",
            "amount_ratio_20",
            "turnover_rate",
            "pb",
            "total_mv",
            "net_mf_ratio",
            "big_order_net_ratio",
            "net_mf_ratio_sum_5",
            "big_order_net_ratio_sum_5",
        ]
        if c in df.columns
    ]
    for col in rank_base:
        rank_col = f"{col}_cs_rank"
        df[rank_col] = df.groupby("trade_date")[col].rank(pct=True)
        feature_cols.append(rank_col)

    df[f"future_ret_{horizon}_close"] = g["close"].transform(lambda s: _future_return(s, horizon))
    df[f"future_ret_{horizon}_open2open"] = g["open"].transform(
        lambda s: s.shift(-(horizon + 1)) / (s.shift(-1) + EPS) - 1.0
    )
    future_close = g["close"].transform(lambda s: s.shift(-horizon))
    future_open = g["open"].transform(lambda s: s.shift(-horizon))
    df[f"future_ret_{horizon}_intraday"] = future_close / (future_open + EPS) - 1.0

    target_map = {
        "close": f"future_ret_{horizon}_close",
        "open2open": f"future_ret_{horizon}_open2open",
        "intraday": f"future_ret_{horizon}_intraday",
    }
    if target_mode not in target_map:
        raise ValueError(f"target_mode must be one of {sorted(target_map)}, got {target_mode}")
    selected_ret_col = target_map[target_mode]
    df[f"future_ret_{horizon}"] = df[selected_ret_col]
    target_raw = df[selected_ret_col] - df.groupby("trade_date")[selected_ret_col].transform("mean")
    lower = target_raw.groupby(df["trade_date"]).transform(lambda s: s.quantile(0.01))
    upper = target_raw.groupby(df["trade_date"]).transform(lambda s: s.quantile(0.99))
    clipped = target_raw.clip(lower=lower, upper=upper)
    mean = clipped.groupby(df["trade_date"]).transform("mean")
    std = clipped.groupby(df["trade_date"]).transform("std")
    df["target"] = (clipped - mean) / (std + EPS)

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = df.groupby("trade_date")[col].transform("median")
        df[col] = df[col].fillna(med)
        df[col] = df[col].fillna(0.0).astype(np.float32)
    return df, feature_cols


def tradable_filter(df: pd.DataFrame, min_amount: float = 50000.0, min_price: float = 3.0) -> pd.Series:
    amount_ok = df["amount"].fillna(0) >= min_amount
    price_ok = df["close"].fillna(0) >= min_price
    not_limit_like = df["ret_1"].abs().fillna(0) < 0.095
    return amount_ok & price_ok & not_limit_like
