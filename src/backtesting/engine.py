"""
Python Portfolio Backtesting Engine
Simulates chronological portfolio evolution with realistic trading frictions:
- Starting capital (e.g. INR 1,000,000)
- Transaction costs (proportional turnover fee, e.g. 10 bps)
- Weights drift tracking between rebalancing events
- Comprehensive metrics generation (CAGR, Sharpe, Volatility, Max Drawdown, Turnover)
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional, Callable

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from src.evaluation.metrics import evaluate_portfolio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Chronological portfolio backtesting engine.
    """
    def __init__(
        self,
        prices_df: pd.DataFrame,
        initial_capital: float = 1_000_000.0,
        transaction_cost_rate: float = 0.0010,  # 10 bps (0.1%)
        risk_free_rate: float = 0.02
    ):
        self.prices_df = prices_df.copy()
        self.tickers = list(prices_df.columns)
        self.n_assets = len(self.tickers)
        self.initial_capital = float(initial_capital)
        self.transaction_cost_rate = float(transaction_cost_rate)
        self.risk_free_rate = float(risk_free_rate)

        # Compute asset daily returns
        self.returns_df = self.prices_df.pct_change().fillna(0.0)

    def run_strategy_weights(
        self,
        weights_df: pd.DataFrame,
        strategy_name: str = "Strategy"
    ) -> Dict[str, Any]:
        """
        Backtests a pre-computed or generated weights DataFrame indexed by dates.
        weights_df: DataFrame with columns matching self.tickers.
        """
        # Align dates
        common_dates = self.returns_df.index.intersection(weights_df.index)
        if len(common_dates) < 2:
            raise ValueError("Insufficient date overlap between prices and weights.")

        returns_sub = self.returns_df.loc[common_dates].values
        weights_matrix = weights_df.loc[common_dates].values

        n_days = len(common_dates)
        portfolio_values = np.zeros(n_days)
        daily_returns = np.zeros(n_days)
        turnovers = np.zeros(n_days)
        fees = np.zeros(n_days)

        current_val = self.initial_capital
        current_weights = weights_matrix[0]
        # Initial rebalance cost to establish portfolio
        initial_turnover = np.sum(np.abs(current_weights))
        initial_fee = initial_turnover * current_val * self.transaction_cost_rate
        current_val -= initial_fee

        portfolio_values[0] = current_val
        turnovers[0] = initial_turnover
        fees[0] = initial_fee

        # Iterate through remaining trading days
        for t in range(1, n_days):
            target_weights = weights_matrix[t]
            day_returns = returns_sub[t]

            # 1. Asset returns applied to existing positions
            step_asset_growth = 1.0 + day_returns
            drifted_values = current_weights * current_val * step_asset_growth
            drifted_total_val = np.sum(drifted_values)

            # Drifted weights before rebalancing
            drifted_weights = drifted_values / drifted_total_val if drifted_total_val > 0 else current_weights

            # 2. Rebalancing & Transaction Cost
            delta_w = np.sum(np.abs(target_weights - drifted_weights))
            cost = delta_w * drifted_total_val * self.transaction_cost_rate

            current_val = drifted_total_val - cost
            turnovers[t] = delta_w
            fees[t] = cost
            portfolio_values[t] = current_val
            daily_returns[t] = (current_val - portfolio_values[t - 1]) / portfolio_values[t - 1]

            current_weights = target_weights

        # Construct results dataframe
        results_df = pd.DataFrame({
            "portfolio_value": portfolio_values,
            "daily_return": daily_returns,
            "turnover": turnovers,
            "transaction_cost": fees
        }, index=common_dates)

        # Merge weights
        for idx, ticker in enumerate(self.tickers):
            results_df[f"weight_{ticker}"] = weights_matrix[:, idx]

        metrics = evaluate_portfolio(
            portfolio_values=portfolio_values,
            turnovers=turnovers,
            transaction_costs=fees,
            risk_free_rate=self.risk_free_rate
        )
        metrics["strategy"] = strategy_name
        metrics["initial_capital"] = self.initial_capital
        metrics["final_value"] = float(portfolio_values[-1])

        return {
            "strategy_name": strategy_name,
            "metrics": metrics,
            "history": results_df
        }

    def backtest_equal_weight(self, rebalance_interval: int = 1) -> Dict[str, Any]:
        """Backtests 1/N Equal Weight strategy."""
        dates = self.returns_df.index
        target = np.ones(self.n_assets) / self.n_assets
        weights_data = np.tile(target, (len(dates), 1))
        weights_df = pd.DataFrame(weights_data, index=dates, columns=self.tickers)
        return self.run_strategy_weights(weights_df, strategy_name="Equal Weight")

    def backtest_buy_and_hold(self) -> Dict[str, Any]:
        """Backtests Buy & Hold (only rebalanced on Day 0)."""
        dates = self.returns_df.index
        n_days = len(dates)
        weights_matrix = np.zeros((n_days, self.n_assets))
        w = np.ones(self.n_assets) / self.n_assets
        weights_matrix[0] = w

        returns = self.returns_df.values
        for t in range(1, n_days):
            grown = w * (1.0 + returns[t])
            s = np.sum(grown)
            w = grown / s if s > 0 else w
            weights_matrix[t] = w

        weights_df = pd.DataFrame(weights_matrix, index=dates, columns=self.tickers)
        return self.run_strategy_weights(weights_df, strategy_name="Buy & Hold")

    def backtest_markowitz(
        self,
        lookback_window: int = 60,
        rebalance_interval: int = 21  # Rebalance monthly to keep realistic turnover
    ) -> Dict[str, Any]:
        """Backtests rolling Markowitz Mean-Variance Optimization."""
        from src.portfolio.markowitz import MarkowitzOptimizer
        opt = MarkowitzOptimizer(n_assets=self.n_assets, lookback_window=lookback_window)
        dates = self.returns_df.index
        n_days = len(dates)

        weights_matrix = np.zeros((n_days, self.n_assets))
        # Initial burn-in period equal weights
        current_w = np.ones(self.n_assets) / self.n_assets

        for t in range(n_days):
            if t >= lookback_window and (t % rebalance_interval == 0):
                hist = self.returns_df.iloc[t - lookback_window:t].values
                current_w = opt.optimize(hist)
            weights_matrix[t] = current_w

        weights_df = pd.DataFrame(weights_matrix, index=dates, columns=self.tickers)
        return self.run_strategy_weights(weights_df, strategy_name="Markowitz MVO")


if __name__ == "__main__":
    prices = pd.read_csv("data/processed/clean_prices.csv", index_col=0, parse_dates=True)
    engine = BacktestEngine(prices)

    ew_res = engine.backtest_equal_weight()
    bh_res = engine.backtest_buy_and_hold()
    mvo_res = engine.backtest_markowitz()

    print("\n--- Baseline Results Comparison ---")
    for res in [ew_res, bh_res, mvo_res]:
        m = res["metrics"]
        print(f"Strategy: {m['strategy']:<18} | CAGR: {m['cagr']*100:6.2f}% | Sharpe: {m['sharpe_ratio']:5.2f} | Vol: {m['annualized_volatility']*100:5.2f}% | MDD: {m['max_drawdown']*100:5.2f}%")
