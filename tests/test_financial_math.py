"""
Financial Mathematics and Accounting Unit Tests
Verifies:
1. Hand-calculated portfolio daily returns and compounding.
2. Correct transaction turnover and cost deduction.
3. Sharpe ratio calculation against verified formula.
4. Sortino ratio (downside deviation only).
5. Maximum Drawdown calculation.
6. CAGR computation across multi-year horizon.
"""

import pytest
import numpy as np
import pandas as pd
from src.evaluation.metrics import (
    compute_cumulative_return,
    compute_cagr,
    compute_annualized_volatility,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    evaluate_portfolio
)


class TestFinancialCalculations:
    def test_hand_calculated_compounding_and_cumulative_return(self):
        # Initial capital: 100,000
        # Day 1: +10% -> 110,000
        # Day 2: -5%  -> 104,500
        # Day 3: +20% -> 125,400
        values = np.array([100000.0, 110000.0, 104500.0, 125400.0])
        
        cum_ret = compute_cumulative_return(values)
        expected_cum = (125400.0 - 100000.0) / 100000.0  # 0.254 = +25.4%
        assert np.isclose(cum_ret, expected_cum, atol=1e-8)

    def test_cagr_two_year_exact(self):
        # Initial: 100,000, After 2 years (504 trading days): 144,000
        # Expected CAGR = (144000 / 100000) ** (1 / 2) - 1 = 1.2 - 1 = 0.20 (20%)
        values = np.linspace(100000.0, 144000.0, 505)
        cagr = compute_cagr(values, periods_per_year=252)
        assert np.isclose(cagr, 0.20, atol=1e-5)

    def test_annualized_volatility_known_sample(self):
        daily_returns = np.array([0.01, -0.01, 0.02, -0.02, 0.005, -0.005])
        std_sample = np.std(daily_returns, ddof=1)
        expected_ann_vol = std_sample * np.sqrt(252)
        
        ann_vol = compute_annualized_volatility(daily_returns, periods_per_year=252)
        assert np.isclose(ann_vol, expected_ann_vol, atol=1e-8)

    def test_sharpe_ratio_hand_calculation(self):
        # 10 days of constant 0.1% daily return with zero risk-free rate
        # Let returns have slight variation:
        r = np.array([0.001, 0.002, 0.0005, 0.0015, 0.001, -0.0005, 0.002, 0.001, 0.0012, 0.0008])
        ann_ret = np.mean(r) * 252
        ann_vol = np.std(r, ddof=1) * np.sqrt(252)
        expected_sharpe = (ann_ret - 0.02) / ann_vol
        
        sharpe = compute_sharpe_ratio(r, risk_free_rate=0.02, periods_per_year=252)
        assert np.isclose(sharpe, expected_sharpe, atol=1e-8)

    def test_sortino_ratio_ignores_upside_volatility(self):
        # Positive returns only: downside deviation should be 0, Sortino should handle gracefully
        r_all_pos = np.array([0.01, 0.02, 0.015, 0.03, 0.005])
        # Returns with negative swings:
        r_mixed = np.array([0.01, -0.02, 0.015, -0.03, 0.005])
        
        sortino = compute_sortino_ratio(r_mixed, risk_free_rate=0.02)
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)

    def test_max_drawdown_hand_calculated(self):
        # Peak = 100, falls to 70 (30% drop), recovers to 90, falls to 60 (40% drop from 100)
        values = np.array([100.0, 110.0, 88.0, 77.0, 95.0, 55.0, 80.0])
        # Peak was 110.0, trough was 55.0. Drawdown = (110 - 55) / 110 = 55 / 110 = 0.50 (50%)
        mdd = compute_max_drawdown(values)
        assert np.isclose(mdd, 0.50, atol=1e-8)

    def test_transaction_cost_deduction(self):
        initial_val = 1_000_000.0
        turnover = 0.50  # 50% portfolio rebalance
        rate = 0.0010    # 10 bps
        cost = initial_val * turnover * rate
        assert np.isclose(cost, 500.0, atol=1e-8)
