# A Share Stock Prediction

This project implements a reproducible baseline for the deep learning stock trend prediction assignment.

## Quick Start

From `D:\Deep Learning\stock_prediction`:

```powershell
python scripts\prepare_features.py --data-dir "..\A股数据" --start-date 20230101 --end-date 20260428 --output outputs\features_2023.pkl
python scripts\train_mlp.py --feature-file outputs\features_2023.pkl --model-out outputs\mlp_checkpoint_2023.pt --pred-out outputs\val_predictions_2023.csv --train-end 20251231 --val-start 20260101 --val-end 20260420 --epochs 8
python scripts\evaluate_strategy_grid.py --pred-file outputs\val_predictions_2023.csv --output outputs\strategy_grid_2023.csv
python scripts\predict_latest.py --feature-file outputs\features_2023.pkl --checkpoint outputs\mlp_checkpoint_2023.pt --output outputs\latest_candidates_2023.csv --top-n 80
python scripts\make_rebalance_plan.py --candidates outputs\latest_candidates_2023.csv --output outputs\rebalance_plan.csv --n 20 --k 2
```

Multi-seed ensemble:

```powershell
python scripts\train_mlp.py --feature-file outputs\features_2023.pkl --model-out outputs\mlp_checkpoint_2023_seed7.pt --pred-out outputs\val_predictions_2023_seed7.csv --metrics-out outputs\metrics_2023_seed7.json --train-end 20251231 --val-start 20260101 --val-end 20260420 --epochs 8 --batch-size 4096 --seed 7
python scripts\train_mlp.py --feature-file outputs\features_2023.pkl --model-out outputs\mlp_checkpoint_2023_seed42.pt --pred-out outputs\val_predictions_2023_seed42.csv --metrics-out outputs\metrics_2023_seed42.json --train-end 20251231 --val-start 20260101 --val-end 20260420 --epochs 8 --batch-size 4096 --seed 42
python scripts\predict_ensemble.py --feature-file outputs\features_2023.pkl --checkpoints outputs\mlp_checkpoint_2023.pt outputs\mlp_checkpoint_2023_seed7.pt outputs\mlp_checkpoint_2023_seed42.pt --output outputs\val_predictions_ensemble.csv --latest-output outputs\latest_candidates_ensemble.csv --latest-top-n 80
python scripts\evaluate_strategy_grid.py --pred-file outputs\val_predictions_ensemble.csv --output outputs\strategy_grid_ensemble.csv
python scripts\make_rebalance_plan.py --candidates outputs\latest_candidates_ensemble.csv --output outputs\rebalance_plan_ensemble_aggressive.csv --n 10 --k 1
python scripts\make_rebalance_plan.py --candidates outputs\latest_candidates_ensemble.csv --output outputs\rebalance_plan_ensemble_balanced.csv --n 20 --k 1
```

Rank-loss aggressive model:

```powershell
python scripts\train_rank_mlp.py --feature-file outputs\features_2023.pkl --model-out outputs\rank_mlp_checkpoint.pt --pred-out outputs\val_predictions_rank_mlp.csv --metrics-out outputs\metrics_rank_mlp.json --train-end 20251231 --val-start 20260101 --val-end 20260420 --epochs 10 --lr 0.0005 --rank-weight 1.0 --top-weight 0.25 --seed 77 --top-n 10 --top-k 1
python scripts\predict_latest.py --feature-file outputs\features_2023.pkl --checkpoint outputs\rank_mlp_checkpoint.pt --output outputs\latest_candidates_rank_mlp.csv --top-n 50
python scripts\make_rebalance_plan.py --candidates outputs\latest_candidates_ensemble.csv --output outputs\rebalance_plan_ensemble_top5.csv --n 5 --k 1
```

Execution-realistic open-to-open validation:

```powershell
python scripts\prepare_features.py --data-dir "..\A股数据" --start-date 20230101 --end-date 20260428 --target-mode open2open --output outputs\features_2023_open2open.pkl
python scripts\evaluate_external_scores.py --feature-file outputs\features_2023_open2open.pkl --score-file outputs\val_predictions_ensemble.csv --output outputs\open2open_ensemble_eval.csv --start-date 20260101 --end-date 20260420
python scripts\evaluate_filter_grid.py --feature-file outputs\features_2023_open2open.pkl --score-file outputs\val_predictions_ensemble.csv --output outputs\filter_grid_open2open_ensemble.csv --start-date 20260101 --end-date 20260420 --n 10 --k 1
python scripts\filter_candidates.py --candidates outputs\latest_candidates_ensemble.csv --output outputs\latest_candidates_ensemble_no_star.csv --min-amount 50000 --market-mode no_star --ret-mode all --top-n 80
python scripts\evaluate_diversified.py --feature-file outputs\features_2023_open2open.pkl --score-file outputs\val_predictions_ensemble.csv --output outputs\diversified_eval_open2open_no_star.csv --market-mode no_star --min-amount 50000 --n 10 --k 1
python scripts\evaluate_final_strategy_grid.py --feature-file outputs\features_2023_open2open.pkl --score-file outputs\val_predictions_ensemble.csv --output outputs\final_strategy_grid_open2open.csv --start-date 20260101 --end-date 20260420 --min-amount 50000
python scripts\make_diversified_plan.py --candidates outputs\latest_candidates_ensemble_no_star.csv --output outputs\rebalance_plan_final_n8_no_star_industry1.csv --n 8 --k 1 --max-per-industry 1
python scripts\run_daily_pipeline.py --data-dir "..\A股数据" --output-dir outputs\daily --n 8 --k 1 --max-per-industry 1 --market-mode no_star --min-amount 50000
python scripts\scan_candidate_news.py --data-dir "..\A股数据" --candidates outputs\rebalance_plan_final_n8_no_star_industry1.csv --output outputs\candidate_news_scan.csv
```

Use a shorter date range first for smoke tests, then expand to 2019 onward.

## Data Alignment

Rows are keyed by `signal_date`. Features on day `t` only use data available up to day `t`. The default target is next close-to-close return:

```text
future_ret_1 = close[t + 1] / close[t] - 1
```

The latest day can be predicted even when `future_ret_1` is not available.

## Main Scripts

- `scripts/prepare_features.py`: read local CSV files and build a feature panel.
- `scripts/train_mlp.py`: train an MLP deep learning baseline and evaluate IC/backtest metrics.
- `scripts/evaluate_strategy_grid.py`: compare TopN and low-turnover rebalance parameters.
- `scripts/predict_latest.py`: produce latest ranked candidates for trading.
- `scripts/predict_ensemble.py`: average scores from multiple MLP checkpoints.
- `scripts/make_rebalance_plan.py`: create initial buy or daily rebalance orders from candidates and current holdings.
- `scripts/evaluate_external_scores.py`: evaluate saved model scores on another return definition, such as `open2open`.
- `scripts/evaluate_filter_grid.py`: compare liquidity, board, and prior-return candidate filters.
- `scripts/filter_candidates.py`: apply the selected filter to the latest candidate list.
- `scripts/evaluate_diversified.py`: evaluate industry concentration caps.
- `scripts/make_diversified_plan.py`: create a diversified initial buy list.
- `scripts/run_daily_pipeline.py`: one-command daily feature, prediction, filtering, and rebalance pipeline.
- `scripts/make_market_capped_plan.py`: optional latest-day list with a maximum number of 创业板 holdings.
- `scripts/scan_candidate_news.py`: scan candidate stocks against same-day news for manual risk review.
- `scripts/train_tree_model.py`: optional sklearn tree baseline. Current experiments did not beat the MLP ensemble.
- `scripts/walk_forward_validate.py`: train and validate month by month to reduce single-window overfitting.
- `scripts/evaluate_score_blend_grid.py`: evaluate daily rank-normalized blends of multiple model score files.
- `scripts/evaluate_aux_filter_grid.py`: use an auxiliary model score as a daily percentile filter while ranking by the base score.
- `scripts/retarget_features.py`: reuse an existing feature file and rebuild `target` for `close`, `open2open`, or `intraday`.

## Current Baseline

Backtest metrics now include both `total_return` and `annual_return`. The annualized value is computed as:

```text
annual_return = (1 + total_return) ** (252 / trading_days) - 1
```

For short validation windows this can look extreme. Use `total_return`, `trading_days`, `max_drawdown`, and month-by-month checks first; use annualized return only when comparing runs over the same date range.

Using `20230101-20260428` features, `20230101-20251231` training, and `20260101-20260420` validation:

- Validation IC mean: `0.0509`
- Validation Rank IC mean: `0.0913`
- Recommended grid result so far: `rebalance n=20, k=2`
- Recommended validation annual return: about `45.9%`
- Recommended validation Sharpe: about `1.48`
- Recommended validation max drawdown: about `-11.2%`

The signal is useful but noisy. Current evidence favors moderate low turnover: hold about 20 stocks, sell the weakest 2 holdings, and buy 2 new high-score candidates each trading day.

The 3-seed ensemble improves the validation backtest:

- Validation IC mean: `0.0536`
- Validation Rank IC mean: `0.0919`
- Aggressive setting: `rebalance n=10, k=1`, annual return about `199.6%`, Sharpe about `3.81`, max drawdown about `-5.3%`
- Balanced setting: `rebalance n=20, k=1`, annual return about `84.5%`, Sharpe about `2.67`, max drawdown about `-8.4%`

For the short simulated trading contest, use `n=10,k=1` if pursuing ranking aggressively; use `n=20,k=1` if prioritizing diversification and smoother execution.

After monthly validation, the most aggressive setting is `ensemble n=5,k=1`:

- Full validation annual return: about `249.9%`
- Sharpe: about `3.69`
- Max drawdown: about `-6.8%`
- It was positive in each validation month from 2026-01 to 2026-04.

Use `outputs/rebalance_plan_ensemble_top5.csv` as the high-conviction contest plan. `rank_mlp` is kept as an auxiliary watchlist because it improves Top10 average return but has less stable low-turnover backtest behavior.

Open-to-open validation is more realistic for next-day open rebalancing. The same ensemble scores remain useful:

- Open-to-open Rank IC mean: `0.0650`
- `n=10,k=1`: annual return about `133.8%`, Sharpe about `3.35`, max drawdown about `-7.0%`
- `n=5,k=1`: annual return about `119.3%`, Sharpe about `2.47`, max drawdown about `-8.7%`

Because of this, the live contest default should be `n=10,k=1`. Use `n=5,k=1` only when deliberately taking higher concentration risk.

The latest trading-layer improvement is:

- Exclude 科创板 from the live candidate pool.
- Keep `amount >= 50000` thousand yuan.
- Hold `n=8`, rebalance `k=1`.
- Cap each industry at 1 stock in the initial list.

Open-to-open validation after these constraints:

- `no_star, n=10,k=1`: annual return about `179.3%`, Sharpe about `4.28`, max drawdown about `-5.1%`
- `no_star + industry_cap=1, n=10,k=1`: annual return about `260.2%`, Sharpe about `5.13`, max drawdown about `-6.0%`
- `no_star + industry_cap=1, n=8,k=1`: annual return about `337.8%`, Sharpe about `5.33`, max drawdown about `-5.2%`

The same default `no_star + industry_cap=1, n=8,k=1` result is `49.8%` total return over 69 validation trading days. The annualized `337.8%` is only the 252-day extrapolation of that 69-day result.

The current final contest file is:

```text
outputs/rebalance_plan_final_n8_no_star_industry1.csv
```

Use `n=5,k=1` only as the highest-risk backup: it has higher full-period annualized return but a negative February validation month.

Further model-improvement workflow:

```powershell
python scripts\walk_forward_validate.py --feature-file outputs\features_2023_open2open.pkl --output-dir outputs\walk_forward_open2open --val-start 20250101 --val-end 20260420 --epochs 6 --n 8 --k 1 --max-per-industry 1 --market-mode no_star
python scripts\evaluate_score_blend_grid.py --feature-file outputs\features_2023_open2open.pkl --score-files ensemble=outputs\val_predictions_ensemble.csv open=outputs\val_predictions_open2open_seed2026.csv rankoo=outputs\val_predictions_rank_open2open.csv tree=outputs\val_predictions_tree_close.csv --output outputs\score_blend_grid_open2open.csv --weight-step 0.5 --ns 5 8 10 --ks 1 --caps 0 1 --market-mode no_star
```

The first coarse blend check showed that the existing close-target ensemble remains the strongest live score source; current single-seed open2open/rank/tree scores did not improve the final open-to-open strategy when blended. The next useful experiment is therefore to train a multi-seed open2open ensemble with the expanded feature set, then rerun blend validation.

The expanded-feature open2open v2 ensemble improved Rank IC but did not improve the live Top8 open-to-open strategy. A small auxiliary-score filter also failed to improve the default `n=8, industry_cap=1` setup. Keep the production daily pipeline on the original 3-seed close-target ensemble until a walk-forward test shows a trading-layer improvement, not just an IC improvement.

GBDT sidecar status:

```powershell
python scripts\train_gbdt_sidecar.py --feature-file outputs\features_2023_open2open_v2.pkl --model-out outputs\gbdt_sidecar_open2open_v2_sklearn.joblib --pred-out outputs\val_predictions_gbdt_sidecar_open2open_v2.csv --latest-output outputs\latest_candidates_gbdt_sidecar_open2open_v2.csv --metrics-out outputs\metrics_gbdt_sidecar_open2open_v2.json
```

The local environment does not currently have `lightgbm` or `catboost`, so the script used sklearn `HistGradientBoostingRegressor`. It did not beat the close ensemble, and blending it into the close ensemble reduced Top5/Top8 trading returns. Keep it as a sidecar experiment only.

Daily contest command:

```powershell
python scripts\run_daily_pipeline.py --data-dir "..\A股数据" --output-dir outputs\daily --n 8 --k 1 --max-per-industry 1 --market-mode no_star --min-amount 50000
```

Latest blended contest workflow:

```powershell
python scripts\train_live_ensemble.py --feature-file outputs\features_live_20260506.pkl --output-dir outputs\live_ensemble_20260506 --prefix live_close --seeds 2026 7 42 --val-days 20 --epochs 8 --eval-ret-mode open2open --market-mode no_star --min-amount 50000 --plan-n 8 --plan-k 1 --max-per-industry 1
python scripts\run_blended_daily_pipeline.py --data-dir "..\A股数据" --output-dir outputs\daily_blend --weights old:0.9,live:0.1 --n 8 --k 1 --max-per-industry 1 --market-mode no_star --ret-mode all --min-amount 50000
```

The latest 2026-05-06 comparison favored rank-normalized blending of the old close ensemble and latest-data live ensemble:

- Old close ensemble, `n=8,k=1,industry_cap=1`, 2026-04-02 to 2026-04-30 open-to-open total return: about `7.3%`.
- Latest-data live ensemble improved Rank IC and reached about `7.7%` strategy total return on the same check.
- Blend `old=0.9, live=0.1` improved the same open-to-open low-turnover check to about `15.5%` total return, Sharpe about `6.10`, max drawdown about `-1.7%`.
- The current blended 2026-05-06 plan is `outputs/live_ensemble_20260506/rebalance_plan_blend_old90_live10_n8_industry1_20260506.csv`.

Agent harness command:

```powershell
python scripts\agent_daily.py --config config\agent_config.yaml
```

Optional LLM decision layer:

1. Set `llm_enabled: true` in `config/agent_config.yaml`.
2. Set the API key environment variable named by `llm_api_key_env`, default `DEEPSEEK_API_KEY`.
3. Run `agent_daily.py` as usual.

The LLM can only select from the filtered candidate pool, veto risky names, and explain the decision. Programmatic guardrails reject invalid outputs.

The agent also builds `outputs/daily/plan_variants_YYYYMMDD.json` before calling the LLM. Current variants are:

- `aggressive_n8_industry1`: default contest plan, 8 stocks, max 1 per industry.
- `balanced_n8_chinext4`: 8 stocks, max 1 per industry, max 4 创业板 names.
- `defensive_n8_main_bias`: 8 stocks, max 1 per industry, max 3 创业板 names.
- `concentrated_n5`: 5-stock high-conviction backup for deliberately higher concentration risk.

If the LLM returns `selected_strategy`, validation uses that strategy's `n`, industry cap, and 创业板 cap. Invalid or hallucinated strategies still fall back to the deterministic plan.

DeepSeek setup example:

```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
python scripts\agent_daily.py --config config\agent_config.yaml
```

For day 2 and later, pass current holdings:

```powershell
python scripts\run_daily_pipeline.py --data-dir "..\A股数据" --output-dir outputs\daily --holdings 当前持仓.csv --n 8 --k 1 --max-per-industry 1 --market-mode no_star --min-amount 50000
python scripts\agent_daily.py --config config\agent_config.yaml --holdings 当前持仓.csv
```

Run news risk review before placing orders:

```powershell
python scripts\scan_candidate_news.py --data-dir "..\A股数据" --candidates outputs\daily\rebalance_plan_YYYYMMDD.csv --output outputs\daily\candidate_news_scan_YYYYMMDD.csv
```

The agent writes:

- `outputs/daily/rebalance_plan_YYYYMMDD.csv`
- `outputs/daily/candidate_news_scan_YYYYMMDD.csv`
- `outputs/daily/plan_variants_YYYYMMDD.json`
- `outputs/daily/agent_summary_YYYYMMDD.json`
- `outputs/daily/llm_decision_YYYYMMDD.json`
- `outputs/agent_reports/agent_report_YYYYMMDD.md`
- `outputs/trading_log.csv`

After manual execution, record fills:

```powershell
python scripts\record_fills.py --fills 今日成交.csv --summary outputs\daily\agent_summary_YYYYMMDD.json --log outputs\fill_log.csv
```

## Notes

- `lightgbm` is optional and not required for the current baseline.
- Generated model checkpoints and feature files are working artifacts. The assignment says model weights and raw data do not need to be submitted.
