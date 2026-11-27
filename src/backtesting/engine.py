"""
Python Portfolio Backtesting Engine

Chronological portfolio simulation with:
- Transaction costs
- Portfolio weight drift
- Equal Weight, Buy & Hold and Markowitz strategies
- Financial performance metrics
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.metrics import evaluate_portfolio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class BacktestEngine:
    """Chronological portfolio backtesting engine."""

    def __init__(
        self,
        prices_df: pd.DataFrame,
        initial_capital: float = 1_000_000.0,
        transaction_cost_rate: float = 0.0010,
        risk_free_rate: float = 0.02,
    ):
        if not isinstance(prices_df, pd.DataFrame) or prices_df.empty:
            raise ValueError("Price data must be a non-empty DataFrame.")

        if prices_df.shape[1] == 0:
            raise ValueError("At least one asset is required.")

        if prices_df.columns.duplicated().any():
            raise ValueError("Duplicate ticker columns detected.")

        if prices_df.index.duplicated().any():
            raise ValueError("Duplicate price dates detected.")

        if not prices_df.index.is_monotonic_increasing:
            raise ValueError("Price dates must be sorted chronologically.")

        prices = prices_df.astype(float)

        if not np.isfinite(prices.to_numpy()).all():
            raise ValueError("Prices contain NaN or infinite values.")

        if (prices <= 0).any().any():
            raise ValueError("Prices must be strictly positive.")

        if not np.isfinite(initial_capital) or initial_capital <= 0:
            raise ValueError("Initial capital must be positive and finite.")

        if not np.isfinite(transaction_cost_rate) or not 0 <= transaction_cost_rate < 1:
            raise ValueError("Transaction cost rate must be in [0, 1).")

        if not np.isfinite(risk_free_rate):
            raise ValueError("Risk-free rate must be finite.")

        self.prices_df = prices.copy()
        self.tickers = list(prices.columns)
        self.n_assets = len(self.tickers)
        self.initial_capital = float(initial_capital)
        self.transaction_cost_rate = float(transaction_cost_rate)
        self.risk_free_rate = float(risk_free_rate)

        self.returns_df = self.prices_df.pct_change().fillna(0.0)

    def _validate_weights(self, weights_df: pd.DataFrame) -> pd.DataFrame:
        """Validate portfolio weights and align ticker columns."""

        if not isinstance(weights_df, pd.DataFrame) or weights_df.empty:
            raise ValueError("Weights must be a non-empty DataFrame.")

        missing_tickers = set(self.tickers) - set(weights_df.columns)
        extra_tickers = set(weights_df.columns) - set(self.tickers)

        if missing_tickers:
            raise ValueError(f"Missing weight columns: {missing_tickers}")

        if extra_tickers:
            logger.warning("Ignoring extra weight columns: %s", extra_tickers)

        weights = weights_df[self.tickers].copy()

        if weights.index.duplicated().any():
            raise ValueError("Duplicate dates in weights.")

        values = weights.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError("Weights contain NaN or infinite values.")

        if (values < -1e-8).any():
            raise ValueError("Negative weights are not supported.")

        values = np.clip(values, 0.0, None)
        row_sums = values.sum(axis=1)

        if not np.allclose(row_sums, 1.0, atol=1e-5):
            raise ValueError("Every weight row must sum to 1.")

        weights.loc[:, :] = values
        return weights

    def run_strategy_weights(
        self,
        weights_df: pd.DataFrame,
        strategy_name: str = "Strategy",
        rebalance: bool = True,
    ) -> Dict[str, Any]:
        """Backtest target weights using same-day forward return convention."""

        weights_df = self._validate_weights(weights_df)

        common_dates = self.returns_df.index.intersection(weights_df.index)

        if len(common_dates) < 2:
            raise ValueError("Insufficient date overlap between prices and weights.")

        returns_sub = self.returns_df.loc[common_dates].to_numpy()
        weights_matrix = weights_df.loc[common_dates].to_numpy()

        n_days = len(common_dates)

        portfolio_values = np.zeros(n_days)
        daily_returns = np.zeros(n_days)
        turnovers = np.zeros(n_days)
        fees = np.zeros(n_days)

        current_val = self.initial_capital
        current_weights = weights_matrix[0].copy()

        # Initial portfolio establishment cost
        initial_turnover = float(np.abs(current_weights).sum())
        initial_fee = current_val * initial_turnover * self.transaction_cost_rate
        current_val -= initial_fee

        portfolio_values[0] = current_val
        turnovers[0] = initial_turnover
        fees[0] = initial_fee

        for t in range(1, n_days):
            target_weights = weights_matrix[t]
            day_returns = returns_sub[t]

            if rebalance:
                turnover = float(np.abs(target_weights - current_weights).sum())
            else:
                turnover = 0.0
                target_weights = current_weights.copy()

            cost = current_val * turnover * self.transaction_cost_rate
            value_after_cost = current_val - cost

            # Earn forward returns on the allocation held during this period
            portfolio_gross_return = float(np.dot(target_weights, day_returns))
            current_val = value_after_cost * (1.0 + portfolio_gross_return)

            # Update drifted holdings
            grown_positions = target_weights * (1.0 + day_returns)
            gross_factor = float(grown_positions.sum())

            if gross_factor > 0:
                current_weights = grown_positions / gross_factor
            else:
                current_weights = target_weights.copy()

            turnovers[t] = turnover
            fees[t] = cost
            portfolio_values[t] = current_val

            previous_value = portfolio_values[t - 1]
            daily_returns[t] = (
                current_val / previous_value - 1.0
                if previous_value > 0
                else 0.0
            )

        history = pd.DataFrame(
            {
                "portfolio_value": portfolio_values,
                "daily_return": daily_returns,
                "turnover": turnovers,
                "transaction_cost": fees,
            },
            index=common_dates,
        )

        for i, ticker in enumerate(self.tickers):
            history[f"weight_{ticker}"] = weights_matrix[:, i]

        metrics = evaluate_portfolio(
            portfolio_values=portfolio_values,
            turnovers=turnovers,
            transaction_costs=fees,
            risk_free_rate=self.risk_free_rate,
        )

        metrics["strategy"] = strategy_name
        metrics["initial_capital"] = self.initial_capital
        metrics["final_value"] = float(portfolio_values[-1])

        return {
            "strategy_name": strategy_name,
            "metrics": metrics,
            "history": history,
        }

    def backtest_equal_weight(self, rebalance_interval: int = 1) -> Dict[str, Any]:
        """Backtest equal weight with periodic rebalancing."""

        if not isinstance(rebalance_interval, int) or rebalance_interval < 1:
            raise ValueError("rebalance_interval must be a positive integer.")

        dates = self.returns_df.index
        target = np.ones(self.n_assets) / self.n_assets

        weights_matrix = np.zeros((len(dates), self.n_assets))
        current_weights = target.copy()

        for t in range(len(dates)):
            if t % rebalance_interval == 0:
                current_weights = target.copy()

            weights_matrix[t] = current_weights

            if t < len(dates) - 1:
                next_returns = self.returns_df.iloc[t + 1].to_numpy()
                grown = current_weights * (1.0 + next_returns)
                total = grown.sum()

                if total > 0:
                    current_weights = grown / total

        weights_df = pd.DataFrame(
            weights_matrix,
            index=dates,
            columns=self.tickers,
        )

        return self.run_strategy_weights(
            weights_df,
            strategy_name="Equal Weight",
            rebalance=True,
        )

    def backtest_buy_and_hold(self) -> Dict[str, Any]:
        """Initial equal allocation with no subsequent rebalancing."""

        dates = self.returns_df.index
        n_days = len(dates)

        weights_matrix = np.zeros((n_days, self.n_assets))
        current_weights = np.ones(self.n_assets) / self.n_assets

        for t in range(n_days):
            weights_matrix[t] = current_weights

            if t < n_days - 1:
                next_returns = self.returns_df.iloc[t + 1].to_numpy()
                grown_positions = current_weights * (1.0 + next_returns)
                total = grown_positions.sum()

                if total > 0:
                    current_weights = grown_positions / total

        weights_df = pd.DataFrame(
            weights_matrix,
            index=dates,
            columns=self.tickers,
        )

        return self.run_strategy_weights(
            weights_df,
            strategy_name="Buy & Hold",
            rebalance=False,
        )

    def backtest_markowitz(
        self,
        lookback_window: int = 60,
        rebalance_interval: int = 21,
        warmup_prices: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Backtest rolling Markowitz MVO using historical warm-up data."""

        if not isinstance(lookback_window, int) or lookback_window < 2:
            raise ValueError("lookback_window must be >= 2.")

        if not isinstance(rebalance_interval, int) or rebalance_interval < 1:
            raise ValueError("rebalance_interval must be >= 1.")

        from src.portfolio.markowitz import MarkowitzOptimizer

        optimizer = MarkowitzOptimizer(
            n_assets=self.n_assets,
            lookback_window=lookback_window,
        )

        test_prices = self.prices_df.copy()
        test_dates = test_prices.index

        if warmup_prices is not None:
            warmup_prices = warmup_prices.copy()

            if list(warmup_prices.columns) != self.tickers:
                raise ValueError(
                    "Warm-up price columns must match test price columns."
                )

            warmup_prices = warmup_prices.loc[
                warmup_prices.index < test_dates[0]
            ].tail(lookback_window + 1)

            combined_prices = pd.concat([warmup_prices, test_prices])

            combined_prices = combined_prices[
                ~combined_prices.index.duplicated(keep="last")
            ].sort_index()
        else:
            combined_prices = test_prices

        combined_returns = combined_prices.pct_change()

        weights_matrix = np.zeros((len(test_dates), self.n_assets))
        current_weights = np.ones(self.n_assets) / self.n_assets

        for i, date in enumerate(test_dates):
            historical_returns = combined_returns.loc[
                combined_returns.index < date
            ].dropna().tail(lookback_window)

            if (
                len(historical_returns) >= lookback_window
                and i % rebalance_interval == 0
            ):
                current_weights = optimizer.optimize(
                    historical_returns.to_numpy()
                )

            weights_matrix[i] = current_weights

        weights_df = pd.DataFrame(
            weights_matrix,
            index=test_dates,
            columns=self.tickers,
        )

        return self.run_strategy_weights(
            weights_df,
            strategy_name="Markowitz MVO",
            rebalance=True,
        )


if __name__ == "__main__":
    prices = pd.read_csv(
        "data/processed/clean_prices.csv",
        index_col=0,
        parse_dates=True,
    )

    engine = BacktestEngine(prices)

    ew_result = engine.backtest_equal_weight()
    bh_result = engine.backtest_buy_and_hold()
    mvo_result = engine.backtest_markowitz()

    print("\n--- Baseline Results Comparison ---")

    for result in [ew_result, bh_result, mvo_result]:
        metrics = result["metrics"]
        print(
            f"Strategy: {metrics['strategy']:<18} | "
            f"CAGR: {metrics['cagr']*100:6.2f}% | "
            f"Sharpe: {metrics['sharpe_ratio']:5.2f} | "
            f"Vol: {metrics['annualized_volatility']*100:5.2f}% | "
            f"MDD: {metrics['max_drawdown']*100:5.2f}%"
        )