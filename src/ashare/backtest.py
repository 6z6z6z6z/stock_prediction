from __future__ import annotations

import numpy as np
import pandas as pd


def performance(daily_ret: pd.Series) -> dict[str, float]:
    daily_ret = daily_ret.dropna()
    if daily_ret.empty:
        return {
            "total_return": np.nan,
            "annual_return": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "trading_days": 0,
        }
    equity = (1.0 + daily_ret).cumprod()
    total_return = equity.iloc[-1] - 1.0
    trading_days = len(daily_ret)
    annual_return = equity.iloc[-1] ** (252.0 / trading_days) - 1.0
    vol = daily_ret.std(ddof=1)
    sharpe = daily_ret.mean() / vol * np.sqrt(252.0) if vol and not np.isnan(vol) else np.nan
    drawdown = equity / equity.cummax() - 1.0
    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "trading_days": int(trading_days),
    }


def topn_backtest(
    df: pd.DataFrame,
    score_col: str,
    ret_col: str,
    n: int = 15,
    cost: float = 0.001,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    prev_holdings: set[str] = set()
    for date, day in df.dropna(subset=[score_col, ret_col]).groupby("trade_date"):
        selected = day.nlargest(n, score_col)
        holdings = set(selected["ts_code"])
        turnover = len(holdings.symmetric_difference(prev_holdings)) / max(n, 1) / 2.0 if prev_holdings else 1.0
        gross = selected[ret_col].mean() if len(selected) else 0.0
        net = gross - cost * turnover
        rows.append({"trade_date": date, "return": net, "turnover": turnover, "holding_count": len(holdings)})
        prev_holdings = holdings
    curve = pd.DataFrame(rows)
    if not curve.empty:
        curve["equity"] = (1.0 + curve["return"]).cumprod()
    return curve, performance(curve["return"] if not curve.empty else pd.Series(dtype=float))


def rebalance_topk_backtest(
    df: pd.DataFrame,
    score_col: str,
    ret_col: str,
    n: int = 15,
    k: int = 3,
    cost: float = 0.001,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    holdings: list[str] = []
    data = df.dropna(subset=[score_col, ret_col]).copy()
    for date, day in data.groupby("trade_date"):
        day = day.sort_values(score_col, ascending=False)
        score_map = day.set_index("ts_code")[score_col].to_dict()
        if not holdings:
            holdings = day.head(n)["ts_code"].tolist()
            turnover = 1.0
        else:
            valid_holdings = [code for code in holdings if code in score_map]
            sell = sorted(valid_holdings, key=lambda code: score_map.get(code, -np.inf))[:k]
            remaining = [code for code in valid_holdings if code not in sell]
            buy = [code for code in day["ts_code"].tolist() if code not in remaining][: max(0, n - len(remaining))]
            holdings = (remaining + buy)[:n]
            turnover = len(buy) / max(n, 1)

        held = day.loc[day["ts_code"].isin(holdings)]
        gross = held[ret_col].mean() if len(held) else 0.0
        net = gross - cost * turnover
        rows.append({"trade_date": date, "return": net, "turnover": turnover, "holding_count": len(holdings)})

    curve = pd.DataFrame(rows)
    if not curve.empty:
        curve["equity"] = (1.0 + curve["return"]).cumprod()
    return curve, performance(curve["return"] if not curve.empty else pd.Series(dtype=float))


def rebalance_topk_backtest_diversified(
    df: pd.DataFrame,
    score_col: str,
    ret_col: str,
    n: int = 10,
    k: int = 1,
    max_per_industry: int = 2,
    cost: float = 0.001,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    holdings: list[str] = []
    data = df.dropna(subset=[score_col, ret_col]).copy()
    for date, day in data.groupby("trade_date"):
        day = day.sort_values(score_col, ascending=False)
        score_map = day.set_index("ts_code")[score_col].to_dict()
        industry_map = day.set_index("ts_code")["industry"].astype(str).to_dict() if "industry" in day.columns else {}
        if not holdings:
            holdings = _select_diversified(day, score_col, n, max_per_industry, exclude=set())
            turnover = 1.0
        else:
            valid_holdings = [code for code in holdings if code in score_map]
            sell = sorted(valid_holdings, key=lambda code: score_map.get(code, -float("inf")))[:k]
            remaining = [code for code in valid_holdings if code not in sell]
            buy_need = max(k, n - len(remaining))
            buy = _select_diversified(
                day,
                score_col,
                buy_need,
                max_per_industry,
                exclude=set(remaining),
                existing_industries=[industry_map.get(code, "") for code in remaining],
            )
            holdings = (remaining + buy)[:n]
            turnover = len(buy) / max(n, 1)

        held = day.loc[day["ts_code"].isin(holdings)]
        gross = held[ret_col].mean() if len(held) else 0.0
        net = gross - cost * turnover
        rows.append({"trade_date": date, "return": net, "turnover": turnover, "holding_count": len(holdings)})

    curve = pd.DataFrame(rows)
    if not curve.empty:
        curve["equity"] = (1.0 + curve["return"]).cumprod()
    return curve, performance(curve["return"] if not curve.empty else pd.Series(dtype=float))


def _select_diversified(
    day: pd.DataFrame,
    score_col: str,
    n: int,
    max_per_industry: int,
    exclude: set[str],
    existing_industries: list[str] | None = None,
) -> list[str]:
    counts: dict[str, int] = {}
    for industry in existing_industries or []:
        counts[industry] = counts.get(industry, 0) + 1
    selected: list[str] = []
    for _, row in day.sort_values(score_col, ascending=False).iterrows():
        code = row["ts_code"]
        if code in exclude:
            continue
        industry = str(row.get("industry", ""))
        if counts.get(industry, 0) >= max_per_industry:
            continue
        selected.append(code)
        counts[industry] = counts.get(industry, 0) + 1
        if len(selected) >= n:
            break
    return selected
