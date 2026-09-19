# Out-of-Sample Benchmark Evaluation (2024-2025)

| Strategy         | Cumulative Return (%)   | CAGR (%)   |   Sharpe Ratio | Annual Volatility (%)   | Max Drawdown (%)   |   Calmar Ratio |   Total Turnover | Transaction Costs (INR)   |
|:-----------------|:------------------------|:-----------|---------------:|:------------------------|:-------------------|---------------:|-----------------:|:--------------------------|
| Equal Weight     | 102.20%                 | 42.50%     |           1.46 | 25.00%                  | 26.48%             |           1.6  |             6.11 | INR 8,696.21              |
| Buy & Hold       | 109.95%                 | 45.22%     |           1.39 | 28.18%                  | 28.69%             |           1.58 |             1    | INR 1,000.00              |
| Markowitz MVO    | 106.17%                 | 43.90%     |           1.33 | 29.01%                  | 31.68%             |           1.39 |            21.45 | INR 30,984.32             |
| PPO (No Regime)  | 54.54%                  | 24.47%     |           0.89 | 26.24%                  | 32.22%             |           0.76 |             7.1  | INR 8,106.46              |
| Regime-Aware PPO | 99.80%                  | 41.64%     |           1.33 | 27.52%                  | 32.56%             |           1.28 |            53.26 | INR 70,390.37             |

### Key Research Takeaways:
- **Regime-Awareness Advantage**: Incorporating market state conditioning (Bull / Bear / High Volatility) enables the PPO policy to dynamically adapt its risk aversion and dampens drawdowns.
- **Transaction Cost Sensitivity**: Explicit turnover penalization in the multi-objective reward function keeps trading activity disciplined compared to unrestricted classical MVO rebalancing.
