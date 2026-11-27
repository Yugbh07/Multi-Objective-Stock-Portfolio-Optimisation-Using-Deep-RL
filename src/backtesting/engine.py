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

        # Iterate through remaining trading days. weights_matrix[t] is the target
        # portfolio available before return_sub[t] is realized. This convention
        # matches PPO inference: the action observed at t-1 is stored on date t
        # and earns the return from t-1 to t (no one-day execution lag).
        for t in range(1, n_days):
            target_weights = weights_matrix[t]
            day_returns = returns_sub[t]

            # 1. Rebalance at the start of day t, before its return is earned.
            turnover = float(np.sum(np.abs(target_weights - current_weights)))
            cost = turnover * current_val * self.transaction_cost_rate
            value_after_cost = current_val - cost

            # 2. Earn day-t returns on the target allocation.
            asset_growth = 1.0 + day_returns
            portfolio_gross_return = float(np.dot(target_weights, day_returns))
            current_val = value_after_cost * (1.0 + portfolio_gross_return)


            # 3. Track end-of-day drifted weights for next day's turnover.
            grown_positions = target_weights * asset_growth
            gross_factor = float(np.sum(grown_positions))
            current_weights = (
                grown_positions / gross_factor
                if gross_factor > 0.0
                else target_weights.copy()
            )

            turnovers[t] = turnover
            fees[t] = cost
            portfolio_values[t] = current_val
            previous_value = portfolio_values[t - 1]
            daily_returns[t] = (
                (current_val - previous_value) / previous_value
                if previous_value > 0.0
                else 0.0
            )

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
        """Backtests Buy & Hold (only rebalanced on Day 0).


        Each row contains the allocation available before that date's return is
        realized. We therefore store the prior day's drifted weights for date t,
        then update the allocation using return[t] for the next date.
        """
        dates = self.returns_df.index
        n_days = len(dates)
        weights_matrix = np.zeros((n_days, self.n_assets))
        w = np.ones(self.n_assets) / self.n_assets
        weights_matrix[0] = w


        returns = self.returns_df.values
        for t in range(1, n_days):
            # Today's target is yesterday's end-of-day drifted allocation.
            weights_matrix[t] = w
            grown = w * (1.0 + returns[t])
            total = float(np.sum(grown))
            w = grown / total if total > 0.0 else w

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
