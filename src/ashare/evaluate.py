from __future__ import annotations

import numpy as np
import pandas as pd


def daily_ic(df: pd.DataFrame, score_col: str, ret_col: str) -> pd.DataFrame:
    rows = []
    for date, day in df.dropna(subset=[score_col, ret_col]).groupby("trade_date"):
        if day[score_col].nunique() < 2 or day[ret_col].nunique() < 2:
            continue
        rows.append(
            {
                "trade_date": date,
                "ic": day[score_col].corr(day[ret_col], method="pearson"),
                "rank_ic": day[score_col].corr(day[ret_col], method="spearman"),
            }
        )
    return pd.DataFrame(rows)


def summarize_ic(ic_df: pd.DataFrame) -> dict[str, float]:
    if ic_df.empty:
        return {"ic_mean": np.nan, "icir": np.nan, "rank_ic_mean": np.nan, "rank_icir": np.nan}
    ic_std = ic_df["ic"].std(ddof=1)
    rank_std = ic_df["rank_ic"].std(ddof=1)
    return {
        "ic_mean": float(ic_df["ic"].mean()),
        "icir": float(ic_df["ic"].mean() / ic_std) if ic_std and not np.isnan(ic_std) else np.nan,
        "rank_ic_mean": float(ic_df["rank_ic"].mean()),
        "rank_icir": float(ic_df["rank_ic"].mean() / rank_std) if rank_std and not np.isnan(rank_std) else np.nan,
    }


def top_group_returns(df: pd.DataFrame, score_col: str, ret_col: str, top_ns: list[int]) -> dict[str, float]:
    result: dict[str, float] = {}
    for n in top_ns:
        daily = []
        for _, day in df.dropna(subset=[score_col, ret_col]).groupby("trade_date"):
            top = day.nlargest(n, score_col)
            if len(top) > 0:
                daily.append(top[ret_col].mean())
        result[f"top{n}_mean_ret"] = float(np.mean(daily)) if daily else np.nan
    return result

