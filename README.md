# A-Share Stock Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-f7931e)
![Status](https://img.shields.io/badge/status-active-brightgreen)

面向深度学习股票趋势预测大作业的 A 股选股与调仓项目。项目从本地日线数据构建特征，训练多 seed MLP / live ensemble，按预测分数生成候选池，并通过过滤、行业约束、低换手调仓与新闻风险扫描输出模拟交易计划。

## Overview

核心目标是提高短期模拟交易比赛成绩，同时保持流程可复现、可审计。

| 模块 | 作用 |
| --- | --- |
| 特征工程 | 从本地 `A股数据/daily/*.csv` 构建按 `trade_date` 对齐的面板特征 |
| 模型训练 | MLP 多 seed 集成、rank-loss 备选模型、GBDT sidecar 实验 |
| 预测融合 | old close ensemble 与 latest-data live ensemble 做 rank-normalized blend |
| 交易约束 | 过滤科创板、成交额门槛、行业集中度约束、低换手调仓 |
| 风险检查 | 候选股票新闻扫描、agent/LLM 决策护栏、成交记录 |

## Current Status

| 项目 | 当前值 |
| --- | --- |
| 状态日期 | `2026-05-09` |
| 本地最新日线 | `20260508.csv` |
| 最新已完成 live ensemble 对比 | `20260506` |
| 推荐 live 组合 | blended `old=0.9, live=0.1` |
| 推荐交易层 | `no_star`, `amount >= 50000`, `n=8`, `k=1`, `industry_cap=1` |

说明：

- `outputs/` 下的特征、模型权重、预测文件和调仓计划是本地工作产物，默认不提交到 Git。
- README 中提到 `20260428`、`20260420`、`20260506` 的内容属于历史实验记录；当前使用前应按最新数据重新生成。
- 如果 `run_blended_daily_pipeline.py` 不传 `--live-checkpoints`，脚本会自动发现最新的 `outputs/live_ensemble_YYYYMMDD` 目录。

## Project Layout

```text
stock_prediction/
├── config/                      # agent 配置
├── docs/                        # 实验计划与数据分析记录
├── scripts/                     # 训练、预测、评估、调仓脚本
├── src/ashare/                  # 复用工具与检查逻辑
├── templates/                   # 报告模板
├── requirements.txt             # Python 依赖
└── README.md
```

## Installation

```powershell
cd "D:\Deep Learning\stock_prediction"
python -m pip install -r requirements.txt
```

数据目录默认放在项目同级：

```text
D:\Deep Learning\A股数据\daily\YYYYMMDD.csv
```

## Quick Start

下面示例使用当前本地最新数据 `20260508`。如果后续更新了新日线，把日期和输出目录同步改成新的交易日即可。

### 1. 生成最新特征

```powershell
python scripts\prepare_features.py `
  --data-dir "..\A股数据" `
  --start-date 20230101 `
  --end-date 20260508 `
  --target-mode close `
  --output outputs\features_live_20260508.pkl
```

### 2. 训练 live ensemble

```powershell
python scripts\train_live_ensemble.py `
  --feature-file outputs\features_live_20260508.pkl `
  --output-dir outputs\live_ensemble_20260508 `
  --prefix live_close `
  --seeds 2026 7 42 `
  --val-days 20 `
  --epochs 8 `
  --eval-ret-mode open2open `
  --market-mode no_star `
  --min-amount 50000 `
  --plan-n 8 `
  --plan-k 1 `
  --max-per-industry 1
```

### 3. 生成 blended 调仓计划

```powershell
python scripts\run_blended_daily_pipeline.py `
  --data-dir "..\A股数据" `
  --output-dir outputs\daily_blend `
  --live-checkpoints `
    outputs\live_ensemble_20260508\live_close_seed2026.pt `
    outputs\live_ensemble_20260508\live_close_seed7.pt `
    outputs\live_ensemble_20260508\live_close_seed42.pt `
  --weights old:0.9,live:0.1 `
  --n 8 `
  --k 1 `
  --max-per-industry 1 `
  --market-mode no_star `
  --ret-mode all `
  --min-amount 50000
```

生成结果会写入：

```text
outputs/daily_blend/rebalance_plan_YYYYMMDD.csv
outputs/daily_blend/blend_latest_YYYYMMDD.csv
outputs/daily_blend/blend_filtered_YYYYMMDD.csv
```

## Model Notes

### Data Alignment

所有样本以 `trade_date` 为键。日期 `t` 的特征只使用 `t` 日及之前可获得的数据。默认 close-target 是下一交易日 close-to-close 收益：

```text
future_ret_1 = close[t + 1] / close[t] - 1
```

最新交易日没有 `future_ret_1` 也可以参与预测。

### Validation Metric

回测同时记录 `total_return` 与 `annual_return`：

```text
annual_return = (1 + total_return) ** (252 / trading_days) - 1
```

短窗口年化收益容易被放大。实际比较时优先看 `total_return`、`trading_days`、`max_drawdown`、Sharpe 和月度稳定性。

## Key Historical Results

这些结果用于解释当前策略来源，不代表最新交易日已经重跑。

| 实验 | 验证窗口 | 结论 |
| --- | --- | --- |
| 单 MLP baseline | `20260101-20260420` | Rank IC `0.0913`，低换手 `n=20,k=2` 更稳 |
| 3-seed close ensemble | `20260101-20260420` | Rank IC `0.0919`，优于单模型 |
| open-to-open 验证 | `20260101-20260420` | 更贴近次日开盘调仓，close ensemble 仍有效 |
| no_star + industry cap | 69 个验证交易日 | `n=8,k=1,industry_cap=1` 总收益约 `49.8%` |
| 2026-05-06 blend | `20260402-20260430` | `old=0.9,live=0.1` 总收益约 `15.5%`，Sharpe 约 `6.10` |

历史 2026-04-28 计划文件：

```text
outputs/rebalance_plan_final_n8_no_star_industry1.csv
```

历史 2026-05-06 blended 计划文件：

```text
outputs/live_ensemble_20260506/rebalance_plan_blend_old90_live10_n8_industry1_20260506.csv
```

## Useful Commands

### 复现实验基线

```powershell
python scripts\prepare_features.py `
  --data-dir "..\A股数据" `
  --start-date 20230101 `
  --end-date 20260428 `
  --target-mode open2open `
  --output outputs\features_2023_open2open.pkl

python scripts\walk_forward_validate.py `
  --feature-file outputs\features_2023_open2open.pkl `
  --output-dir outputs\walk_forward_open2open `
  --val-start 20250101 `
  --val-end 20260420 `
  --epochs 6 `
  --n 8 `
  --k 1 `
  --max-per-industry 1 `
  --market-mode no_star
```

### 评估 score blend

```powershell
python scripts\evaluate_score_blend_grid.py `
  --feature-file outputs\features_2023_open2open.pkl `
  --score-files ensemble=outputs\val_predictions_ensemble.csv open=outputs\val_predictions_open2open_seed2026.csv rankoo=outputs\val_predictions_rank_open2open.csv tree=outputs\val_predictions_tree_close.csv `
  --output outputs\score_blend_grid_open2open.csv `
  --weight-step 0.5 `
  --ns 5 8 10 `
  --ks 1 `
  --caps 0 1 `
  --market-mode no_star
```

### 新闻风险扫描

```powershell
python scripts\scan_candidate_news.py `
  --data-dir "..\A股数据" `
  --candidates outputs\daily_blend\rebalance_plan_YYYYMMDD.csv `
  --output outputs\daily_blend\candidate_news_scan_YYYYMMDD.csv
```

### 记录成交

```powershell
python scripts\record_fills.py `
  --fills 今日成交.csv `
  --summary outputs\daily\agent_summary_YYYYMMDD.json `
  --log outputs\fill_log.csv
```

## Agent Workflow

```powershell
python scripts\agent_daily.py --config config\agent_config.yaml
```

注意：`agent_daily.py` 当前包装的是 legacy deterministic `run_daily_pipeline.py` 输出格式。若使用当前更推荐的 blended workflow，先运行 `run_blended_daily_pipeline.py`，再对 blended 调仓计划执行新闻扫描。

可选 LLM 决策层：

1. 在 `config/agent_config.yaml` 中设置 `llm_enabled: true`。
2. 设置 `DEEPSEEK_API_KEY`，或修改配置里的 `llm_api_key_env`。
3. 运行 `agent_daily.py`。

LLM 只能从过滤后的候选池中选择、否决风险标的并解释原因；程序会校验无效输出并回退到确定性计划。

## Main Scripts

| 脚本 | 用途 |
| --- | --- |
| `scripts/prepare_features.py` | 构建特征面板 |
| `scripts/train_mlp.py` | 训练 MLP baseline |
| `scripts/train_live_ensemble.py` | 用最新数据训练多 seed live ensemble |
| `scripts/predict_ensemble.py` | 对多个 checkpoint 做平均预测 |
| `scripts/run_blended_daily_pipeline.py` | 一键生成 old/live blended 调仓计划 |
| `scripts/run_daily_pipeline.py` | legacy deterministic 日常流水线 |
| `scripts/evaluate_score_blend_grid.py` | 搜索 score blend 权重 |
| `scripts/walk_forward_validate.py` | 月度 walk-forward 验证 |
| `scripts/filter_candidates.py` | 应用流动性、板块、涨跌幅过滤 |
| `scripts/make_diversified_plan.py` | 生成行业分散调仓计划 |
| `scripts/scan_candidate_news.py` | 新闻风险扫描 |
| `scripts/agent_daily.py` | agent 汇总、风险检查和可选 LLM 决策 |

## Notes

- `lightgbm` 和 `catboost` 不是当前必需依赖；GBDT sidecar 默认可使用 sklearn 实现。
- 模型权重、特征文件、原始数据和 `outputs/` 下的产物不需要提交。
- 大作业报告建议基于 `docs/` 中的实验记录和最终运行结果整理，不要直接把 `outputs/` 全量纳入仓库。
