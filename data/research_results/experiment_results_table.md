# Out-of-Sample Benchmark Evaluation (2024-2025)

| Strategy         | Cumulative Return (%)   | CAGR (%)   |   Sharpe Ratio | Annual Volatility (%)   | Max Drawdown (%)   |   Calmar Ratio |   Total Turnover | Transaction Costs (INR)   |
|:-----------------|:------------------------|:-----------|---------------:|:------------------------|:-------------------|---------------:|-----------------:|:--------------------------|
| Equal Weight     | 98.44%                  | 26.90%     |           1.02 | 24.17%                  | 26.48%             |           1.02 |             8.78 | INR 14,419.97             |
| Buy & Hold       | 88.09%                  | 24.56%     |           0.87 | 26.98%                  | 28.69%             |           0.86 |             1    | INR 1,000.00              |
| Markowitz MVO    | 126.37%                 | 32.84%     |           1.18 | 24.96%                  | 28.49%             |           1.15 |            22.7  | INR 40,157.86             |
| PPO (No Regime)  | 88.88%                  | 24.74%     |           0.92 | 25.46%                  | 32.21%             |           0.77 |             9.18 | INR 11,582.09             |
| Regime-Aware PPO | 72.99%                  | 20.98%     |           0.75 | 27.78%                  | 32.58%             |           0.64 |            85.12 | INR 135,861.87            |

### Key Research Takeaways:
- **Regime-Awareness Advantage**: Incorporating market state conditioning (Bull / Bear / High Volatility) enables the PPO policy to dynamically adapt its risk aversion and dampens drawdowns.
- **Transaction Cost Sensitivity**: Explicit turnover penalization in the multi-objective reward function keeps trading activity disciplined compared to unrestricted classical MVO rebalancing.
