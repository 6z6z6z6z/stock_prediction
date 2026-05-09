# A股模拟交易 Agent 日报

## Run

- Date: {trade_date}
- Strategy: {strategy_name}
- Data directory: `{data_dir}`
- Output directory: `{output_dir}`
- Human confirmation required: {human_confirm_required}

## Checks

- Latest daily data date: {trade_date}
- Pipeline status: {pipeline_status}
- News scan status: {news_status}
- Holdings file: {holdings_file}
- Overall guardrail status: {guardrail_status}

{guardrail_table}

## Strategy Parameters

- Candidate market mode: `{market_mode}`
- Minimum amount: `{min_amount}`
- Target holdings `n`: `{n}`
- Daily rebalance `k`: `{k}`
- Max per industry: `{max_per_industry}`

## Rebalance Plan

{rebalance_table}

## News Risk Scan

{news_table}

## LLM Decision Review

- LLM status: {llm_status}
- Decision file: `{llm_decision_file}`

{llm_decision_summary}

## Files

- Rebalance plan: `{rebalance_plan}`
- Filtered candidates: `{filtered_candidates}`
- News scan: `{news_scan}`
- Plan variants: `{plan_variants}`
- LLM decision: `{llm_decision_file}`
- Run summary: `{summary_file}`
- Trading log: `{log_file}`

## Manual Execution Notes

- Do not use intraday information to modify model scores.
- If a buy order cannot be filled because of limit-up or liquidity, use the next candidate from the filtered list while preserving industry constraints.
- If a sell order cannot be filled because of limit-down, keep the position and record the reason in the trading log.
- Confirm final orders manually in 同花顺模拟盘.
