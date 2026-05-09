from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ashare.agent_checks import check_news_scan, check_rebalance_plan, has_failures, has_warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/agent_config.yaml")
    parser.add_argument("--holdings", default="")
    parser.add_argument("--skip-pipeline", action="store_true", help="Use existing daily outputs and only build news scan/report.")
    parser.add_argument("--date", default="", help="Trade date for skip-pipeline mode; defaults to latest daily data date.")
    return parser.parse_args()


def load_simple_yaml(path: Path) -> dict:
    data: dict = {}
    current_key = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  - ") and current_key:
            data.setdefault(current_key, []).append(line.strip()[2:].strip())
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            data[key] = []
            current_key = key
        else:
            data[key] = cast_scalar(value)
            current_key = key
    return data


def cast_scalar(value: str):
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def latest_daily_date(data_dir: Path) -> str:
    files = sorted((data_dir / "daily").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No daily files found under {data_dir / 'daily'}")
    return files[-1].stem


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    cols = [c for c in ["action", "ts_code", "name", "industry", "market", "close", "amount", "score", "mentions", "positive_hits", "negative_hits", "titles"] if c in view.columns]
    view = view[cols]
    view = view.fillna("")
    return view.to_markdown(index=False)


def checks_table(checks) -> str:
    rows = [c.to_dict() for c in checks]
    if not rows:
        return "_No checks._"
    return pd.DataFrame(rows).to_markdown(index=False)


def append_log(log_file: Path, row: dict) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        existing = pd.read_csv(log_file, dtype=str)
        for key in row:
            if key not in existing.columns:
                existing[key] = ""
        for key in existing.columns:
            if key not in row:
                row[key] = ""
        existing = pd.concat([existing, pd.DataFrame([row])[existing.columns]], ignore_index=True)
        existing.to_csv(log_file, index=False, encoding="utf-8-sig")
        return
    with log_file.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = load_simple_yaml(config_path)

    data_dir = resolve_path(config["data_dir"])
    output_dir = resolve_path(config.get("output_dir", "outputs/daily"))
    report_dir = resolve_path(config.get("report_dir", "outputs/agent_reports"))
    log_file = resolve_path(config.get("log_file", "outputs/trading_log.csv"))
    trade_date = args.date or latest_daily_date(data_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rebalance_plan = output_dir / f"rebalance_plan_{trade_date}.csv"
    filtered_candidates = output_dir / f"latest_candidates_filtered_{trade_date}.csv"
    news_scan = output_dir / f"candidate_news_scan_{trade_date}.csv"
    summary_file = output_dir / f"agent_summary_{trade_date}.json"
    llm_decision_file = output_dir / f"llm_decision_{trade_date}.json"
    plan_variants_file = output_dir / f"plan_variants_{trade_date}.json"

    pipeline_status = "skipped"
    if not args.skip_pipeline:
        cmd = [
            sys.executable,
            "scripts/run_daily_pipeline.py",
            "--data-dir",
            str(data_dir),
            "--start-date",
            str(config.get("start_date", 20230101)),
            "--output-dir",
            str(output_dir),
            "--n",
            str(config.get("n", 8)),
            "--k",
            str(config.get("k", 1)),
            "--max-per-industry",
            str(config.get("max_per_industry", 1)),
            "--market-mode",
            str(config.get("market_mode", "no_star")),
            "--min-amount",
            str(config.get("min_amount", 50000)),
            "--checkpoints",
            *[str(resolve_path(p)) for p in config.get("checkpoints", [])],
        ]
        holdings = args.holdings or ""
        if holdings:
            cmd.extend(["--holdings", str(resolve_path(holdings))])
        run(cmd)
        pipeline_status = "completed"

    if not rebalance_plan.exists():
        raise FileNotFoundError(f"Rebalance plan not found: {rebalance_plan}")

    run([
        sys.executable,
        "scripts/scan_candidate_news.py",
        "--data-dir",
        str(data_dir),
        "--candidates",
        str(rebalance_plan),
        "--date",
        trade_date,
        "--output",
        str(news_scan),
    ])
    news_status = "completed"

    llm_cmd = [
        sys.executable,
        "scripts/llm_decision.py",
        "--candidates",
        str(filtered_candidates),
        "--plan",
        str(rebalance_plan),
        "--news",
        str(news_scan),
        "--output",
        str(llm_decision_file),
        "--n",
        str(config.get("n", 8)),
        "--max-per-industry",
        str(config.get("max_per_industry", 1)),
        "--max-chinext",
        str(config.get("llm_max_chinext", 5)),
        "--max-candidates",
        str(config.get("llm_max_candidates", 30)),
        "--provider",
        str(config.get("llm_provider", "openai")),
        "--model",
        str(config.get("llm_model", "gpt-4.1-mini")),
        "--api-key-env",
        str(config.get("llm_api_key_env", "OPENAI_API_KEY")),
    ]
    if config.get("llm_enabled", False):
        llm_cmd.append("--enabled")
    run([
        sys.executable,
        "scripts/generate_plan_variants.py",
        "--candidates",
        str(filtered_candidates),
        "--output",
        str(plan_variants_file),
    ])
    llm_cmd.extend(["--variants", str(plan_variants_file)])
    run(llm_cmd)

    plan_df = pd.read_csv(rebalance_plan, dtype={"trade_date": str, "ts_code": str})
    news_df = pd.read_csv(news_scan, dtype={"trade_date": str, "ts_code": str})
    llm_decision = json.loads(llm_decision_file.read_text(encoding="utf-8"))
    llm_status = llm_decision.get("status", "unknown")
    llm_decision_summary = format_llm_decision(llm_decision)
    checks = []
    checks.extend(check_rebalance_plan(rebalance_plan, int(config.get("n", 8)), int(config.get("max_per_industry", 1))))
    checks.extend(check_news_scan(news_scan))
    if has_failures(checks):
        guardrail_status = "fail"
    elif has_warnings(checks):
        guardrail_status = "warn"
    else:
        guardrail_status = "ok"

    summary = {
        "trade_date": trade_date,
        "strategy_name": config.get("strategy_name", ""),
        "pipeline_status": pipeline_status,
        "news_status": news_status,
        "guardrail_status": guardrail_status,
        "checks": [c.to_dict() for c in checks],
        "files": {
            "rebalance_plan": str(rebalance_plan),
            "filtered_candidates": str(filtered_candidates),
            "news_scan": str(news_scan),
            "llm_decision": str(llm_decision_file),
            "plan_variants": str(plan_variants_file),
            "report": str(report_dir / f"agent_report_{trade_date}.md"),
            "log_file": str(log_file),
        },
        "llm_decision": llm_decision,
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    template = (ROOT / "templates" / "agent_report.md").read_text(encoding="utf-8")
    report_text = template.format(
        trade_date=trade_date,
        strategy_name=config.get("strategy_name", ""),
        data_dir=data_dir,
        output_dir=output_dir,
        human_confirm_required=config.get("human_confirm_required", True),
        pipeline_status=pipeline_status,
        news_status=news_status,
        holdings_file=args.holdings or "None",
        guardrail_status=guardrail_status,
        guardrail_table=checks_table(checks),
        market_mode=config.get("market_mode", "no_star"),
        min_amount=config.get("min_amount", 50000),
        n=config.get("n", 8),
        k=config.get("k", 1),
        max_per_industry=config.get("max_per_industry", 1),
        rebalance_table=markdown_table(plan_df),
        news_table=markdown_table(news_df),
        llm_status=llm_status,
        llm_decision_file=llm_decision_file,
        llm_decision_summary=llm_decision_summary,
        plan_variants=plan_variants_file,
        rebalance_plan=rebalance_plan,
        filtered_candidates=filtered_candidates,
        news_scan=news_scan,
        summary_file=summary_file,
        log_file=log_file,
    )
    report_path = report_dir / f"agent_report_{trade_date}.md"
    report_path.write_text(report_text, encoding="utf-8")

    append_log(
        log_file,
        {
            "trade_date": trade_date,
            "strategy_name": config.get("strategy_name", ""),
            "pipeline_status": pipeline_status,
            "news_status": news_status,
            "rebalance_plan": str(rebalance_plan),
            "news_scan": str(news_scan),
            "llm_decision": str(llm_decision_file),
            "report": str(report_path),
            "summary": str(summary_file),
            "guardrail_status": guardrail_status,
            "holdings": args.holdings or "",
            "notes": "manual confirmation required",
        },
    )
    print(f"agent report -> {report_path}")


def resolve_path(value) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def format_llm_decision(decision_payload: dict) -> str:
    decision = decision_payload.get("decision", {})
    selected = decision_payload.get("selected_details", [])
    lines = [
        f"- Source: `{decision_payload.get('source', '')}`",
        f"- Strategy: `{decision.get('selected_strategy', '')}`",
        f"- Overall risk: `{decision.get('overall_risk', '')}`",
        f"- Manual review required: `{decision.get('manual_review_required', '')}`",
        f"- Rationale: {decision.get('rationale', '')}",
    ]
    if decision_payload.get("error"):
        lines.append(f"- Error/fallback: `{decision_payload['error']}`")
    if decision_payload.get("validation_errors"):
        lines.append(f"- Validation notes: `{decision_payload['validation_errors']}`")
    if selected:
        df = pd.DataFrame(selected)
        lines.append("")
        lines.append(df.to_markdown(index=False))
    vetoed = decision.get("vetoed", [])
    if vetoed:
        lines.append("")
        lines.append("Vetoed:")
        lines.append(pd.DataFrame(vetoed).to_markdown(index=False))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
