# Out-of-Sample Portfolio Evaluation

**Evaluation period:** 2024-01-02 to 2026-11-25

**Initial capital:** INR 1,000,000.00

**Transaction cost rate:** 0.1000%

## Performance Comparison

| Strategy         | Cumulative Return   | CAGR   |   Sharpe Ratio | Annualized Volatility   | Maximum Drawdown   |   Calmar Ratio |   Total Turnover | Transaction Costs   |
|:-----------------|:--------------------|:-------|---------------:|:------------------------|:-------------------|---------------:|-----------------:|:--------------------|
| Equal Weight     | 109.36%             | 29.14% |          1.077 | 24.72%                  | 26.48%             |          1.101 |            8.887 | INR 14,638.91       |
| Buy & Hold       | 100.53%             | 27.23% |          0.933 | 27.78%                  | 28.69%             |          0.949 |            1     | INR 1,000.00        |
| Markowitz MVO    | 151.07%             | 37.53% |          1.3   | 25.45%                  | 29.86%             |          1.257 |           23.233 | INR 42,925.58       |
| PPO (No Regime)  | 97.94%              | 26.66% |          1.067 | 22.68%                  | 26.88%             |          0.992 |           12.332 | INR 18,362.50       |
| Regime-Aware PPO | 105.01%             | 28.21% |          1.07  | 24.05%                  | 26.43%             |          1.067 |           49.101 | INR 82,963.42       |

## Data-Driven Findings

- Highest Sharpe Ratio: Markowitz MVO (1.300).
- Highest CAGR: Markowitz MVO (37.53%).
- Lowest maximum drawdown: Regime-Aware PPO (26.43%).
- Lowest annualized volatility: PPO (No Regime) (22.68%).
- Regime-aware PPO vs PPO without regime: Sharpe difference = +0.003; maximum drawdown difference = -0.44%.
- These are historical out-of-sample observations and do not establish statistical significance or guarantee future performance.


## Methodological Limitations

- Historical performance does not guarantee future results.
- No statistical significance testing was conducted.
- No sensitivity analysis was conducted in this script.
- Results depend on the supplied data and trained models.
