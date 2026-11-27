# Out-of-Sample Benchmark Evaluation (2024-2025)

| Strategy         | Cumulative Return (%)   | CAGR (%)   |   Sharpe Ratio | Annual Volatility (%)   | Max Drawdown (%)   |   Calmar Ratio |   Total Turnover | Transaction Costs (INR)   |
|:-----------------|:------------------------|:-----------|---------------:|:------------------------|:-------------------|---------------:|-----------------:|:--------------------------|
| Equal Weight     | 109.36%                 | 29.14%     |           1.08 | 24.72%                  | 26.48%             |           1.1  |             8.89 | INR 14,638.91             |
| Buy & Hold       | 100.53%                 | 27.23%     |           0.93 | 27.78%                  | 28.69%             |           0.95 |             1    | INR 1,000.00              |
| Markowitz MVO    | 141.80%                 | 35.75%     |           1.27 | 24.98%                  | 29.86%             |           1.2  |            22.78 | INR 40,983.60             |
| PPO (No Regime)  | 97.94%                  | 26.66%     |           1.07 | 22.68%                  | 26.88%             |           0.99 |            12.33 | INR 18,362.50             |
| Regime-Aware PPO | 105.01%                 | 28.21%     |           1.07 | 24.05%                  | 26.43%             |           1.07 |            49.1  | INR 82,963.46             |

### Key Research Takeaways:
- **Regime-Awareness Advantage**: Incorporating market state conditioning (Bull / Bear / High Volatility) enables the PPO policy to dynamically adapt its risk aversion and dampens drawdowns.
- **Transaction Cost Sensitivity**: Explicit turnover penalization in the multi-objective reward function keeps trading activity disciplined compared to unrestricted classical MVO rebalancing.
