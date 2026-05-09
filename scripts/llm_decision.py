from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--news", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-per-industry", type=int, default=1)
    parser.add_argument("--max-chinext", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--variants", default="")
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates, dtype={"trade_date": str, "ts_code": str})
    plan = pd.read_csv(args.plan, dtype={"trade_date": str, "ts_code": str})
    news = pd.read_csv(args.news, dtype={"trade_date": str, "ts_code": str})

    news_cols = ["ts_code", "mentions", "positive_hits", "negative_hits", "keywords", "titles"]
    merged = candidates.merge(news[[c for c in news_cols if c in news.columns]], on="ts_code", how="left")
    for col in ["mentions", "positive_hits", "negative_hits"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)
    for col in ["keywords", "titles"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna("")

    variants = []
    if args.variants and Path(args.variants).exists():
        variants = json.loads(Path(args.variants).read_text(encoding="utf-8"))
    packet = build_packet(args, merged, plan, variants)
    source = "offline_heuristic"
    error = ""
    if args.enabled:
        try:
            raw = call_llm(args, packet)
            decision = parse_llm_json(raw)
            source = f"{args.provider}:{args.model}"
        except Exception as exc:  # keep harness alive; invalid LLM must not block trading plan.
            decision = offline_decision(args, merged)
            error = str(exc)
            source = "offline_fallback_after_llm_error"
    else:
        decision = offline_decision(args, merged)

    valid, validation_errors = validate_decision(
        decision,
        merged,
        args.n,
        args.max_per_industry,
        args.max_chinext,
        variants,
    )
    if not valid:
        fallback = deterministic_plan_decision(plan)
        fallback_valid, fallback_errors = validate_decision(
            fallback,
            merged,
            args.n,
            args.max_per_industry,
            args.max_chinext,
            variants,
        )
        decision = fallback
        validation_errors.extend([f"fallback: {e}" for e in fallback_errors])
        source += "_invalid_fallback_to_plan"
        valid = fallback_valid

    active_constraints = decision_constraints(decision, args.n, args.max_per_industry, args.max_chinext, variants)
    output = {
        "status": "ok" if valid and not error else ("warn" if valid else "fail"),
        "source": source,
        "error": error,
        "constraints": {
            "n": args.n,
            "max_per_industry": args.max_per_industry,
            "max_chinext": args.max_chinext,
        },
        "active_constraints": active_constraints,
        "decision": decision,
        "validation_errors": validation_errors,
        "selected_details": selected_details(decision, merged),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_packet(args: argparse.Namespace, candidates: pd.DataFrame, plan: pd.DataFrame, variants: list[dict]) -> dict:
    cols = [
        "ts_code",
        "name",
        "industry",
        "market",
        "close",
        "amount",
        "ret_1",
        "score",
        "mentions",
        "positive_hits",
        "negative_hits",
        "keywords",
        "titles",
    ]
    available_cols = [c for c in cols if c in candidates.columns]
    candidate_rows = candidates.sort_values("score", ascending=False).head(args.max_candidates)[available_cols]
    default_selected = plan.loc[plan["action"].astype(str).str.contains("buy", case=False, na=False), "ts_code"].tolist()
    return {
        "task": "Choose the final A-share simulated trading buy list from the provided filtered candidates.",
        "hard_constraints": {
            "default_select_count": args.n,
            "default_max_per_industry": args.max_per_industry,
            "default_max_chinext": args.max_chinext,
            "selected_strategy_overrides_default_constraints": True,
            "candidate_pool_only": True,
            "no_star_market": True,
            "manual_confirmation_required": True,
            "do_not_use_intraday_information": True,
            "do_not_modify_model_scores": True,
        },
        "allowed_actions": [
            "Select candidates from the filtered list.",
            "Veto candidates with negative news or execution risk.",
            "Prefer diversification when scores are close.",
            "Explain risk and manual review needs.",
            "Pick one provided strategy variant when it better matches the risk/reward context.",
        ],
        "default_selected": default_selected,
        "strategy_variants": variants,
        "candidates": candidate_rows.to_dict("records"),
        "output_schema": {
            "selected_strategy": "one strategy_variants.name, or empty string to use default constraints",
            "selected": ["ts_code strings, length must match the selected strategy n or the default_select_count"],
            "vetoed": [{"ts_code": "string", "reason": "string"}],
            "manual_review_required": "boolean",
            "overall_risk": "low|medium|high",
            "rationale": "short Chinese explanation",
        },
    }


def call_llm(args: argparse.Namespace, packet: dict) -> str:
    if args.provider == "openai":
        return call_openai_responses(args, packet)
    if args.provider == "deepseek":
        return call_deepseek_chat(args, packet)
    raise ValueError(f"unsupported provider: {args.provider}")


def call_openai_responses(args: argparse.Namespace, packet: dict) -> str:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key env var: {args.api_key_env}")
    payload = {
        "model": args.model,
        "instructions": (
            "You are a constrained trading decision reviewer for a course simulation. "
            "Return only valid JSON. You may select among provided candidates only. "
            "Never violate hard constraints. Do not recommend automatic order placement."
        ),
        "input": json.dumps(packet, ensure_ascii=False),
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP {exc.code}: {body}") from exc
    if "output_text" in data:
        return data["output_text"]
    texts = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                texts.append(content.get("text", ""))
    return "\n".join(texts)


def call_deepseek_chat(args: argparse.Namespace, packet: dict) -> str:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API key env var: {args.api_key_env}")
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个受硬约束控制的A股模拟交易决策复核器。"
                    "只能从给定候选池中选择股票；必须返回严格 JSON；"
                    "不得违反数量、行业、板块、创业板数量上限等硬约束；"
                    "不得建议自动下单；不得使用盘中信息。"
                    "JSON 格式示例："
                    "{\"selected_strategy\":\"aggressive_n8_industry1\",\"selected\":[\"000001.SZ\"],\"vetoed\":[],"
                    "\"manual_review_required\":false,\"overall_risk\":\"low\","
                    "\"rationale\":\"理由\"}"
                ),
            },
            {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
        "stream": False,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {body}") from exc
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"DeepSeek API returned no choices: {data}")
    return choices[0].get("message", {}).get("content", "")


def parse_llm_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def offline_decision(args: argparse.Namespace, candidates: pd.DataFrame) -> dict:
    selected = []
    vetoed = []
    industry_counts: dict[str, int] = {}
    chinext_count = 0
    for _, row in candidates.sort_values("score", ascending=False).iterrows():
        code = str(row["ts_code"])
        industry = str(row.get("industry", ""))
        market = str(row.get("market", ""))
        negative_hits = int(row.get("negative_hits", 0) or 0)
        if negative_hits > 0:
            vetoed.append({"ts_code": code, "reason": "negative news keyword hits"})
            continue
        if industry_counts.get(industry, 0) >= args.max_per_industry:
            continue
        if market == "创业板" and chinext_count >= args.max_chinext:
            continue
        selected.append(code)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if market == "创业板":
            chinext_count += 1
        if len(selected) >= args.n:
            break
    return {
        "selected_strategy": "",
        "selected": selected,
        "vetoed": vetoed,
        "manual_review_required": bool(vetoed),
        "overall_risk": "medium" if vetoed else "low",
        "rationale": "离线启发式：按模型分数排序，并执行新闻负面剔除、行业上限和创业板数量上限。",
    }


def deterministic_plan_decision(plan: pd.DataFrame) -> dict:
    selected = plan.loc[plan["action"].astype(str).str.contains("buy", case=False, na=False), "ts_code"].tolist()
    return {
        "selected_strategy": "",
        "selected": selected,
        "vetoed": [],
        "manual_review_required": True,
        "overall_risk": "medium",
        "rationale": "LLM 决策未通过校验，回退到确定性调仓计划。",
    }


def validate_decision(
    decision: dict,
    candidates: pd.DataFrame,
    n: int,
    max_per_industry: int,
    max_chinext: int,
    variants: list[dict] | None = None,
) -> tuple[bool, list[str]]:
    errors = []
    active = decision_constraints(decision, n, max_per_industry, max_chinext, variants or [])
    target_n = int(active["n"])
    target_max_per_industry = int(active["max_per_industry"])
    target_max_chinext = int(active["max_chinext"])
    if active.get("unknown_strategy"):
        errors.append(f"unknown selected_strategy: {active.get('selected_strategy')}")
    selected = decision.get("selected", [])
    if not isinstance(selected, list):
        errors.append("selected must be a list")
        return False, errors
    if len(selected) != target_n:
        errors.append(f"selected length {len(selected)} != n {target_n}")
    if len(set(selected)) != len(selected):
        errors.append("selected contains duplicates")
    lookup = candidates.set_index("ts_code")
    missing = [code for code in selected if code not in lookup.index]
    if missing:
        errors.append(f"selected codes not in candidate pool: {missing}")
    if not missing:
        rows = lookup.loc[selected]
        industry_counts = rows["industry"].astype(str).value_counts()
        if len(industry_counts) and int(industry_counts.max()) > target_max_per_industry:
            errors.append(f"industry cap violated: {industry_counts.to_dict()}")
        if rows["market"].astype(str).eq("科创板").any():
            errors.append("selected contains 科创板")
        chinext_count = int(rows["market"].astype(str).eq("创业板").sum())
        if chinext_count > target_max_chinext:
            errors.append(f"创业板 count {chinext_count} > max_chinext {target_max_chinext}")
    return not errors, errors


def decision_constraints(
    decision: dict,
    n: int,
    max_per_industry: int,
    max_chinext: int,
    variants: list[dict] | None = None,
) -> dict:
    active = {
        "selected_strategy": decision.get("selected_strategy", ""),
        "n": n,
        "max_per_industry": max_per_industry,
        "max_chinext": max_chinext,
    }
    selected_strategy = str(decision.get("selected_strategy", "") or "")
    if not selected_strategy:
        return active
    for variant in variants or []:
        if str(variant.get("name", "")) == selected_strategy:
            active.update(
                {
                    "n": int(variant.get("n", n)),
                    "max_per_industry": int(variant.get("max_per_industry", max_per_industry)),
                    "max_chinext": int(variant.get("max_chinext", max_chinext)),
                }
            )
            return active
    active["unknown_strategy"] = True
    return active


def selected_details(decision: dict, candidates: pd.DataFrame) -> list[dict]:
    selected = decision.get("selected", [])
    if not selected:
        return []
    lookup = candidates.set_index("ts_code")
    details = []
    for code in selected:
        if code in lookup.index:
            row = lookup.loc[code]
            details.append(
                {
                    "ts_code": code,
                    "name": row.get("name", ""),
                    "industry": row.get("industry", ""),
                    "market": row.get("market", ""),
                    "score": float(row.get("score", 0)),
                    "negative_hits": int(row.get("negative_hits", 0) or 0),
                }
            )
    return details


if __name__ == "__main__":
    main()
