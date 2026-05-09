# A-Share Trading Agent Skill

Use this skill when running the daily simulated A-share trading workflow for the deep learning assignment.

## Hard Rules

- Only use information available up to the latest completed trading day in `A股数据`.
- Do not use same-day intraday prices or news to compute model scores.
- Do not place orders automatically. Generate a plan for manual confirmation.
- Respect the selected strategy constraints:
  - Exclude 科创板 by default.
  - Require `amount >= 50000`.
  - Hold `n = 8`.
  - Rebalance `k = 1`.
  - Keep at most one stock per industry.
- If a selected buy cannot be filled, choose the next filtered candidate that still satisfies industry constraints.
- If a selected sell cannot be filled, keep it and record the reason.

## Workflow

1. Check latest `daily/*.csv` date.
2. Generate features from `20230101` through latest date.
3. Run the 3-seed MLP ensemble.
4. Apply candidate filters.
5. Generate the industry-diversified rebalance plan.
6. Scan candidate news for risk keywords.
7. Produce a Markdown agent report.
8. Append a structured row to `outputs/trading_log.csv`.

## Outputs

- `outputs/daily/rebalance_plan_YYYYMMDD.csv`
- `outputs/daily/latest_candidates_filtered_YYYYMMDD.csv`
- `outputs/daily/candidate_news_scan_YYYYMMDD.csv`
- `outputs/agent_reports/agent_report_YYYYMMDD.md`
- `outputs/trading_log.csv`

