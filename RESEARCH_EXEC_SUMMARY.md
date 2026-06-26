# SPX 0DTE Research Executive Summary

## Objective

Transition this repository from replication into a robust quantitative research platform for discovering statistically significant, out-of-sample, live-tradable SPX 0DTE edges.

## What Is Already In Place

- Rich option-level and strategy-level datasets:
  - `data/data_opt.parquet`
  - `data/data_structures.parquet`
  - `data/vix.parquet`
  - `data/slopes.parquet`
  - `data/future_moments_SPX.parquet`
  - `data/future_moments_VIX.parquet`
  - `data/ALL_eod.csv`
- Existing analysis stack covers:
  - unconditional performance,
  - implementability and tail risk,
  - clustered inference,
  - regime conditioning,
  - strict OOS protocol,
  - model-zoo comparisons,
  - basket-level investment time series.

## Primary Risks Identified

- Multiple testing and model-mining risk across large model/feature grids.
- Some global selection steps can leak information if used for strict live deployment.
- Execution modeling is simplified (spread + fixed bps), with no explicit market impact/queue model.

## Research Direction

Prioritize conditional strategies where edge is economically explainable:

- volatility and skew regimes,
- overnight gap and trend/reversion states,
- realized volatility clustering,
- weekday/month-end/OPEX seasonality,
- macro event day conditioning (FOMC/CPI),
- dealer-gamma and liquidity/flow proxies.

## Automated Framework Added

A separate research layer has been added under `research/` (non-invasive to replication):

- point-in-time feature panel construction,
- hypothesis generation,
- walk-forward selection and OOS execution,
- cost and slippage inclusion,
- full risk metric computation,
- ranked strategy outputs and experiment artifacts.

Run with:

```bash
c:/Projects/0dte-strategies/.venv/Scripts/python.exe research/run_research.py
```

## Ranking Standard

Strategies are ranked by:

1. OOS Sharpe,
2. robustness,
3. stability across years,
4. statistical significance,
5. implementation simplicity.

## Live-Readiness Rules

- No optimization on final test set.
- No curve fitting to isolated episodes.
- Require economic rationale in addition to statistical fit.
- Enforce costs, slippage, turnover awareness, and tail controls.

## Documents

- Full specification: `RESEARCH.md`
- This short brief: `RESEARCH_EXEC_SUMMARY.md`
