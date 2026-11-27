import unittest
import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestEngine


class BacktestingTimingTests(unittest.TestCase):
    def test_day_one_target_weight_earns_day_one_forward_return(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        prices = pd.DataFrame(
            {"A": [100.0, 110.0], "B": [100.0, 100.0]}, index=dates
        )
        weights = pd.DataFrame(
            {"A": [0.5, 1.0], "B": [0.5, 0.0]}, index=dates
        )
        engine = BacktestEngine(
            prices, initial_capital=100.0, transaction_cost_rate=0.0
        )

        result = engine.run_strategy_weights(weights, "Timing Test")
        history = result["history"]

        # The date-1 allocation (100% A) is in place for the date-1 return.
        # Therefore the portfolio should gain 10%, not the old 5% lagged return.
        self.assertAlmostEqual(history["portfolio_value"].iloc[1], 110.0, places=8)
        self.assertAlmostEqual(history["daily_return"].iloc[1], 0.10, places=8)
        np.testing.assert_allclose(
            history[["weight_A", "weight_B"]].iloc[1].values,
            np.array([1.0, 0.0]),
        )

    def test_buy_and_hold_does_not_rebalance_after_initial_allocation(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        prices = pd.DataFrame(
            {"A": [100.0, 110.0, 121.0], "B": [100.0, 100.0, 100.0]},
            index=dates,
        )
        engine = BacktestEngine(
            prices, initial_capital=100.0, transaction_cost_rate=0.001
        )

        result = engine.backtest_buy_and_hold()
        history = result["history"]


        # Only the initial establishment fee should be charged; subsequent
        # weights are drifted holdings, not same-day-return-informed targets.
        self.assertAlmostEqual(history["transaction_cost"].iloc[1], 0.0, places=8)
        self.assertAlmostEqual(history["transaction_cost"].iloc[2], 0.0, places=8)
        self.assertAlmostEqual(history["portfolio_value"].iloc[2], 110.3895, places=5)



    def test_turnover_cost_is_charged_before_forward_return(self):
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        prices = pd.DataFrame(
            {"A": [100.0, 110.0], "B": [100.0, 100.0]}, index=dates
        )
        weights = pd.DataFrame(
            {"A": [0.5, 1.0], "B": [0.5, 0.0]}, index=dates
        )
        engine = BacktestEngine(
            prices, initial_capital=100.0, transaction_cost_rate=0.001
        )

        result = engine.run_strategy_weights(weights, "Cost Timing Test")
        history = result["history"]


        initial_value = 100.0 - 100.0 * 1.0 * 0.001
        rebalance_fee = initial_value * 1.0 * 0.001
        expected = (initial_value - rebalance_fee) * 1.10
        self.assertAlmostEqual(history["transaction_cost"].iloc[1], rebalance_fee, places=8)
        self.assertAlmostEqual(history["portfolio_value"].iloc[1], expected, places=8)



if __name__ == "__main__":
    unittest.main()
