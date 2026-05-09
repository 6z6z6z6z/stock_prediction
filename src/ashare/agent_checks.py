from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


@dataclass
class CheckResult:
    name: str
    status: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def ok(name: str, message: str) -> CheckResult:
    return CheckResult(name=name, status="ok", message=message)


def warn(name: str, message: str) -> CheckResult:
    return CheckResult(name=name, status="warn", message=message)


def fail(name: str, message: str) -> CheckResult:
    return CheckResult(name=name, status="fail", message=message)


def check_file(path: Path, name: str) -> CheckResult:
    if path.exists() and path.stat().st_size > 0:
        return ok(name, f"{path} exists")
    return fail(name, f"{path} missing or empty")


def check_rebalance_plan(path: Path, n: int, max_per_industry: int) -> list[CheckResult]:
    checks: list[CheckResult] = [check_file(path, "rebalance_plan_file")]
    if not path.exists():
        return checks
    df = pd.read_csv(path, dtype={"ts_code": str})
    required = {"action", "ts_code", "name", "industry", "market", "score"}
    missing = sorted(required - set(df.columns))
    if missing:
        checks.append(fail("rebalance_plan_schema", f"missing columns: {missing}"))
        return checks
    checks.append(ok("rebalance_plan_schema", "required columns present"))

    buy_df = df.loc[df["action"].astype(str).str.contains("buy", case=False, na=False)]
    sell_df = df.loc[df["action"].astype(str).eq("sell")]
    if sell_df.empty:
        if len(buy_df) == n:
            checks.append(ok("target_buy_count", f"initial buy count is {n}"))
        else:
            checks.append(warn("target_buy_count", f"initial buy count {len(buy_df)} != target n {n}"))
    else:
        checks.append(ok("rebalance_actions", f"{len(sell_df)} sell rows, {len(buy_df)} buy rows"))

    if len(buy_df) > 0:
        industry_counts = buy_df["industry"].astype(str).value_counts()
        max_seen = int(industry_counts.max())
        if max_seen <= max_per_industry:
            checks.append(ok("industry_cap", f"max industry count {max_seen} <= {max_per_industry}"))
        else:
            checks.append(fail("industry_cap", f"max industry count {max_seen} > {max_per_industry}"))
        if buy_df["market"].astype(str).eq("科创板").any():
            checks.append(fail("market_filter", "buy list contains 科创板"))
        else:
            checks.append(ok("market_filter", "buy list excludes 科创板"))
    return checks


def check_news_scan(path: Path) -> list[CheckResult]:
    checks = [check_file(path, "news_scan_file")]
    if not path.exists():
        return checks
    df = pd.read_csv(path, dtype={"ts_code": str})
    if "negative_hits" not in df.columns:
        checks.append(fail("news_scan_schema", "missing negative_hits column"))
        return checks
    total_negative = int(pd.to_numeric(df["negative_hits"], errors="coerce").fillna(0).sum())
    if total_negative > 0:
        checks.append(warn("news_negative_hits", f"{total_negative} negative keyword hits require manual review"))
    else:
        checks.append(ok("news_negative_hits", "no negative keyword hits"))
    return checks


def has_failures(checks: list[CheckResult]) -> bool:
    return any(c.status == "fail" for c in checks)


def has_warnings(checks: list[CheckResult]) -> bool:
    return any(c.status == "warn" for c in checks)

