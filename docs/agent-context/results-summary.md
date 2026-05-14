# Key Results Summary

## Unconditional Results (§3)

### Variance Risk Premium
- **Exists but small**: Median VRP from 10:00 ET to expiration ≈ 0.0011% of underlying
- Too small to trade after realistic friction

### Individual Options
- Returns highly volatile, especially OTM
- Mean returns often statistically indistinguishable from zero
- Selling slightly OTM calls/puts positive in up to 75% of observations

### Strategy Performance (Table 1)
| Strategy | Mean PNL (% und) | Median | SR (ann.) | Observation |
|----------|-----------------|--------|-----------|-------------|
| Strangle/Straddle | Mixed | Mostly negative | Low | Compact distributions |
| Iron Butterfly/Condor | Slightly positive | Near zero | Moderate | Short vol benefits |
| Risk Reversal | ~+0.01% | Positive | Low | Only consistently positive structure |
| Bull Call Spread | Near zero | Negative | Low | Directional, hurt by costs |
| Bear Put Spread | Near zero | Negative | Low | Directional, hurt by costs |
| Call Ratio Spread | Mixed | Variable | Low | Unstable across regimes |
| Put Ratio Spread | Slightly positive | Variable | Moderate | Best unconditional candidate |

### Regime Instability
- Significant PNL sign changes between pre-2022 and post-2022
- Statistically weak after date-clustered SE correction

### Tail Risk (Table 3)
- ES₁% ranges from 0.58% to 1.58% of underlying
- Worst-day outcomes severe for all strategies
- Cumulative drawdowns substantial

## What Drives PNL (§4)

- **Realized skewness >> realized variance** for asymmetric strategies
- RS explains 20–40% of PNL variation for directional spreads/risk reversals
- IV and IS add 2–7 pp of R² beyond RV alone
- All results survive Benjamini-Hochberg multiple-testing correction (Table 4)

## Conditional Results (§5)

### VIX Regime Filter (Table 6)
- Clear regime heterogeneity: downside structures earn more in high-VIX; upside in low-VIX
- Simple but not sufficient for implementation

### Strategy-Level OOS (Table 7)
| Strategy | Protocol | Hit Rate | SR gross | SR net | Mean net (bps) |
|----------|----------|----------|----------|--------|----------------|
| Put Ratio Spread | Expanding | 67% | **1.18** | **0.93** | 1.91 |
| Iron Butterfly/Condor | Expanding | 64% | 0.77 | -0.20 | -0.11 |
| Strangle/Straddle | Expanding | 72% | 0.56 | 0.39 | 1.20 |
| Risk Reversal | Expanding | 75% | 0.01 | -0.11 | -0.45 |
| Bull Call Spread | Expanding | 58% | -0.14 | -0.54 | -0.70 |
| Bear Put Spread | Expanding | 64% | -0.00 | -0.39 | -0.52 |
| Call Ratio Spread | Expanding | 59% | -0.26 | -0.48 | -1.16 |

### Portfolio Implementation (Table 9)
| Basket | SR gross | SR net | Mean net (bps) |
|--------|----------|--------|----------------|
| Top-3 (by Mean or SR) | **1.12** | **0.82** | 1.36 |
| All-strategies | 0.81 | 0.25 | 0.22 |

### Model Zoo Key Finding
- Binary target (direction prediction) >> return target (magnitude prediction)
- Hard mapping ≥ soft mapping for most model families
- Ridge-logit and elastic-net-logit achieve SR slightly above 1.0
- Tree models (LightGBM, XGBoost) are competitive but less stable

## Bottom Line
0DTE is a **tightly risk-budgeted tactical overlay**, not a standing carry strategy. Selected strategies under disciplined OOS rules can deliver meaningful SR, but diversification is essential and tail risk never vanishes.
