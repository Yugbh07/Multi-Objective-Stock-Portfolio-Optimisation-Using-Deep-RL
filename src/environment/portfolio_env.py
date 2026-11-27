"""
Gymnasium Portfolio Management Environment
Custom reinforcement learning environment for dynamic asset allocation:
- Non-Anticipative MDP: Information at day t determines action w_t; return is earned over day t+1.
- Observation Space: Trailing returns, rolling volatilities, causal regime one-hot,
  current drifted portfolio weights, and cash reserve fraction.
- Action Space: Continuous action vector mapped to probability simplex via Softmax.
- Multi-Objective Reward: Scalarized balance of net forward return, rolling covariance variance penalty,
  and rebalancing turnover penalty.
"""

import os
import sys
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.portfolio.risk_engine import project_to_bounded_simplex


class PortfolioEnv(gym.Env):
    """
    Gymnasium-compliant multi-asset portfolio rebalancing environment.
    Strictly causal and non-anticipative.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices_df: pd.DataFrame,
        features_df: pd.DataFrame,
        regimes_df: pd.DataFrame,
        initial_capital: float = 1_000_000.0,
        transaction_cost_rate: float = 0.0010,
        lambda_risk: float = 1.0,
        lambda_cost: float = 0.5,
        lookback_window: int = 20,
        include_regime: bool = True
    ):
        super().__init__()

        # Synchronize indices across all datasets
        common_index = (
            prices_df.index
            .intersection(features_df.index)
            .intersection(regimes_df.index)
        )

        self.prices_df = prices_df.loc[common_index].copy()
        self.features_df = features_df.loc[common_index].copy()
        self.regimes_df = regimes_df.loc[common_index].copy()

        self.tickers = list(self.prices_df.columns)
        self.n_assets = len(self.tickers)

        self.initial_capital = float(initial_capital)
        self.transaction_cost_rate = float(transaction_cost_rate)

        self.lambda_risk = float(lambda_risk)
        self.lambda_cost = float(lambda_cost)

        self.lookback_window = int(lookback_window)
        self.include_regime = bool(include_regime)

        # Returns: row t = return from day t-1 to day t
        self.returns_matrix = (
            self.prices_df.pct_change()
            .fillna(0.0)
            .values
        )

        self.dates = list(self.prices_df.index)
        self.n_steps = len(self.dates)

        # Feature columns
        self.return_cols = [
            f"{ticker}_return"
            for ticker in self.tickers
            if f"{ticker}_return" in self.features_df.columns
        ]

        self.vol_cols = [
            f"{ticker}_vol_20"
            for ticker in self.tickers
            if f"{ticker}_vol_20" in self.features_df.columns
        ]

        # Observation dimensions
        # N returns + N volatilities + 3 regime indicators + N weights + 1 cash ratio
        self.regime_dim = 3 if self.include_regime else 0

        self.obs_dim = (
            self.n_assets
            + self.n_assets
            + self.regime_dim
            + self.n_assets
            + 1
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # PPO action logits
        self.action_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        # Initial state
        self.current_step = 0
        self.portfolio_value = self.initial_capital

        self.current_weights = (
            np.ones(self.n_assets, dtype=np.float32)
            / self.n_assets
        )

        self.cash_ratio = 0.0

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """
        Convert PPO action logits into a bounded portfolio allocation.

        Constraints:
        - Minimum allocation: 2%
        - Maximum allocation: 35%
        - Sum of allocations: 100%
        """
        action = np.asarray(action, dtype=np.float64)

        action = np.nan_to_num(
            action,
            nan=0.0,
            posinf=5.0,
            neginf=-5.0
        )

        shifted = action - np.max(action)
        exp_action = np.exp(shifted)

        raw_weights = exp_action / np.sum(exp_action)

        bounded_weights = project_to_bounded_simplex(
            raw_weights=raw_weights,
            lower_bound=0.02,
            upper_bound=0.35
        )

        return bounded_weights.astype(np.float64)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Backward-compatible alias."""
        return self._action_to_weights(x)

    def _get_regime_one_hot(self, step: int) -> np.ndarray:
        """
        Three-element regime encoding:
        [Bull, Bear, High Volatility]
        """
        regime_id = int(
            self.regimes_df["regime_id"].iloc[step]
        )

        one_hot = np.zeros(3, dtype=np.float32)

        if 0 <= regime_id < 3:
            one_hot[regime_id] = 1.0

        return one_hot

    def _get_observation(self) -> np.ndarray:
        step = self.current_step

        # Historical return features
        returns = (
            self.features_df[self.return_cols]
            .iloc[step]
            .values.astype(np.float32)
        )

        # Rolling volatility features
        vols = (
            self.features_df[self.vol_cols]
            .iloc[step]
            .values.astype(np.float32)
        )

        observation_components = [returns, vols]

        # Regime information
        if self.include_regime:
            regime_vector = self._get_regime_one_hot(step)
            observation_components.append(regime_vector)

        # Current portfolio state
        observation_components.append(
            self.current_weights.astype(np.float32)
        )

        observation_components.append(
            np.array([self.cash_ratio], dtype=np.float32)
        )

        observation = np.concatenate(
            observation_components
        ).astype(np.float32)

        observation = np.nan_to_num(
            observation,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )

        return observation

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment without charging transaction costs."""

        super().reset(seed=seed)

        self.current_step = 0
        self.portfolio_value = self.initial_capital

        self.current_weights = (
            np.ones(self.n_assets, dtype=np.float32)
            / self.n_assets
        )

        self.cash_ratio = 0.0

        observation = self._get_observation()

        info = {
            "step": self.current_step,
            "date": str(self.dates[self.current_step].date()),
            "portfolio_value": self.portfolio_value
        }

        return observation, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        At day t:
        1. Observe information available at t.
        2. Rebalance to target weights at t.
        3. Pay transaction costs.
        4. Earn returns from t to t+1.
        5. Update portfolio value and drifted weights.
        """

        t = self.current_step
        next_t = t + 1

        # Prevent advancing beyond available data
        if next_t >= self.n_steps:
            return (
                self._get_observation(),
                0.0,
                True,
                False,
                {
                    "step": t,
                    "portfolio_value": self.portfolio_value
                }
            )

        # 1. Convert action into constrained portfolio weights
        target_weights = self._action_to_weights(action)

        # 2. Calculate turnover and transaction costs
        turnover = float(
            np.sum(np.abs(target_weights - self.current_weights))
        )

        cost_in_capital = (
            self.portfolio_value
            * turnover
            * self.transaction_cost_rate
        )

        fee_factor = max(
            0.0,
            1.0 - turnover * self.transaction_cost_rate
        )

        # 3. Forward returns from t to t+1
        forward_returns = self.returns_matrix[next_t]

        # 4. Portfolio performance
        portfolio_gross_return = float(
            np.dot(target_weights, forward_returns)
        )

        value_after_fee = (
            self.portfolio_value * fee_factor
        )

        next_portfolio_value = (
            value_after_fee
            * (1.0 + portfolio_gross_return)
        )

        portfolio_net_return = (
            (next_portfolio_value - self.portfolio_value)
            / self.portfolio_value
            if self.portfolio_value > 0
            else 0.0
        )

        self.portfolio_value = float(
            next_portfolio_value
        )

        # 5. Historical covariance risk (information available at t)
        lookback_start = max(
            0,
            t - self.lookback_window
        )

        recent_returns = self.returns_matrix[
            lookback_start:t + 1
        ]

        if len(recent_returns) > 2:
            covariance = np.cov(
                recent_returns,
                rowvar=False
            )

            portfolio_variance = float(
                target_weights.T
                @ covariance
                @ target_weights
            )

            portfolio_variance = max(
                portfolio_variance,
                0.0
            )

        else:
            portfolio_variance = 0.0001

        # 6. Multi-objective reward
        reward = float(
            (portfolio_net_return * 100.0)
            - (
                self.lambda_risk
                * portfolio_variance
                * 1000.0
            )
            - (
                self.lambda_cost
                * turnover
            )
        )

        # 7. Drift weights according to realized forward returns
        grown_positions = (
            target_weights
            * (1.0 + forward_returns)
        )

        drifted_total = float(
            np.sum(grown_positions)
        )

        self.current_weights = (
            grown_positions / drifted_total
            if drifted_total > 0
            else target_weights.astype(np.float32)
        )

        # Advance environment
        self.current_step = next_t

        terminated = bool(
            self.current_step >= self.n_steps - 1
        )

        truncated = False

        observation = (
            self._get_observation()
            if not terminated
            else np.zeros(
                self.obs_dim,
                dtype=np.float32
            )
        )

        info = {
            "step": self.current_step,
            "date": str(self.dates[self.current_step].date()),
            "portfolio_value": self.portfolio_value,
            "net_return": portfolio_net_return,
            "turnover": turnover,
            "cost": cost_in_capital,
            "weights": target_weights
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )
    