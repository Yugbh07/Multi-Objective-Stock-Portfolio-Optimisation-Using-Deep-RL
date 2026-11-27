"""
Gymnasium Portfolio Management Environment
Custom reinforcement learning environment for dynamic asset allocation:
- Non-Anticipative MDP: Information at day t determines action w_t; return is earned over day t+1.
  (Strictly eliminates lookahead bias).
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
    Strictly causal and non-anticipative: action chosen at step t earns return at step t+1.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices_df: pd.DataFrame,
        features_df: pd.DataFrame,
        regimes_df: pd.DataFrame,
        initial_capital: float = 1_000_000.0,
        transaction_cost_rate: float = 0.0010,  # 10 bps (0.1%)
        lambda_risk: float = 1.0,
        lambda_cost: float = 0.5,
        lookback_window: int = 20,
        include_regime: bool = True
    ):
        super().__init__()

        # Synchronize indices across dataframes
        common_index = prices_df.index.intersection(features_df.index).intersection(regimes_df.index)
        self.prices_df = prices_df.loc[common_index].copy()
        self.features_df = features_df.loc[common_index].copy()
        self.regimes_df = regimes_df.loc[common_index].copy()

        self.tickers = list(prices_df.columns)
        self.n_assets = len(self.tickers)
        self.initial_capital = float(initial_capital)
        self.transaction_cost_rate = float(transaction_cost_rate)
        self.lambda_risk = float(lambda_risk)
        self.lambda_cost = float(lambda_cost)
        self.lookback_window = lookback_window
        self.include_regime = include_regime

        # Returns matrix for all assets: row t is return from day t-1 to day t
        self.returns_matrix = self.prices_df.pct_change().fillna(0.0).values
        self.dates = list(self.prices_df.index)
        self.n_steps = len(self.dates)

        # Feature column subsets
        self.return_cols = [f"{t}_return" for t in self.tickers if f"{t}_return" in self.features_df.columns]
        self.vol_cols = [f"{t}_vol_20" for t in self.tickers if f"{t}_vol_20" in self.features_df.columns]

        # Calculate observation dimension
        # N returns + N volatilities + (3 regime one-hot if enabled) + N current weights + 1 cash ratio
        self.regime_dim = 3 if self.include_regime else 0
        self.obs_dim = self.n_assets + self.n_assets + self.regime_dim + self.n_assets + 1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # Action space: unconstrained continuous vector converted to simplex weights via Softmax
        self.action_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        # State tracking
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.current_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.cash_ratio = 0.0

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """
        Maps continuous PPO action logits to bounded probability simplex:
        1. Numerical stable softmax for valid initial probability simplex.
        2. Exact convex water-filling projection guaranteeing:
           - lower_bound = 0.02 (2% diversification floor)
           - upper_bound = 0.35 (35% institutional concentration cap)
           - sum(weights) == 1.0 (exact full investment simplex constraint).
        """
        a = np.asarray(action, dtype=np.float64)
        a = np.nan_to_num(a, nan=0.0, posinf=5.0, neginf=-5.0)
        shifted = a - np.max(a)
        exp_a = np.exp(shifted)
        raw_w = exp_a / np.sum(exp_a)

        bounded_w = project_to_bounded_simplex(
            raw_weights=raw_w,
            lower_bound=0.02,
            upper_bound=0.35
        )
        return bounded_w.astype(np.float32)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for _action_to_weights."""
        return self._action_to_weights(x)

    def _get_regime_one_hot(self, step: int) -> np.ndarray:
        """Returns 3-element one-hot vector: [Bull, Bear, High Volatility]."""
        regime_id = int(self.regimes_df["regime_id"].iloc[step])
        one_hot = np.zeros(3, dtype=np.float32)
        if 0 <= regime_id < 3:
            one_hot[regime_id] = 1.0
        return one_hot

    def _get_observation(self) -> np.ndarray:
        """Constructs state observation vector for step self.current_step."""
        step = self.current_step

        # 1. Historical asset returns up to step
        returns = self.features_df[self.return_cols].iloc[step].values.astype(np.float32)

        # 2. Asset rolling volatilities up to step
        vols = self.features_df[self.vol_cols].iloc[step].values.astype(np.float32)

        obs_components = [returns, vols]

        # 3. Market regime one-hot up to step (if enabled)
        if self.include_regime:
            regime_vec = self._get_regime_one_hot(step)
            obs_components.append(regime_vec)

        # 4. Current drifted portfolio weights and cash
        obs_components.append(self.current_weights.astype(np.float32))
        obs_components.append(np.array([self.cash_ratio], dtype=np.float32))

        obs = np.concatenate(obs_components).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to the beginning of the episode."""
        super().reset(seed=seed)

        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.current_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.cash_ratio = 0.0

        observation = self._get_observation()
        info = {
            "step": self.current_step,
            "date": str(self.dates[self.current_step].date()),
            "portfolio_value": self.portfolio_value
        }
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes strictly non-anticipative trading step:
        1. Agent observes state at time t, outputs target weights w_t via Softmax.
        2. Rebalancing turnover and transaction costs computed at close of day t.
        3. Market advances to day t+1; portfolio earns forward return r_{t+1}.
        4. Scalarized multi-objective reward evaluated on r_{t+1}, trailing covariance risk, and turnover.
        5. Drifted weights computed at end of day t+1; next state S_{t+1} returned.
        """
        t = self.current_step
        next_t = t + 1

        # Check if episode can advance to next trading day
        if next_t >= self.n_steps:
            terminated = True
            truncated = False
            return self._get_observation(), 0.0, terminated, truncated, {"step": t, "portfolio_value": self.portfolio_value}

        # 1. Action mapped to target weights on the bounded simplex (35% cap, 2% floor)
        target_weights = self._action_to_weights(action)

        # 2. Rebalancing turnover and transaction fee at day t
        turnover = float(np.sum(np.abs(target_weights - self.current_weights)))
        cost_rate = self.transaction_cost_rate
        fee_factor = max(0.0, 1.0 - turnover * cost_rate)
        cost_in_capital = self.portfolio_value * turnover * cost_rate

        # 3. Next-day forward asset returns realized over day t -> t+1 (STRICTLY CAUSAL)
        forward_returns = self.returns_matrix[next_t]

        # 4. Portfolio value and return compounding (strictly aligned with BacktestEngine)
        portfolio_gross_return = float(np.dot(target_weights, forward_returns))
        value_after_fee = self.portfolio_value * fee_factor
        next_portfolio_value = value_after_fee * (1.0 + portfolio_gross_return)
        portfolio_net_return = float((next_portfolio_value - self.portfolio_value) / self.portfolio_value) if self.portfolio_value > 0 else 0.0
        self.portfolio_value = next_portfolio_value

        # 5. Risk estimate: Portfolio variance estimated using trailing historical covariance up to day t
        lookback_start = max(0, t - self.lookback_window)
        recent_slice = self.returns_matrix[lookback_start:t + 1]
        if len(recent_slice) > 2:
            cov = np.cov(recent_slice, rowvar=False)
            port_variance = float(np.dot(target_weights.T, np.dot(cov, target_weights)))
        else:
            port_variance = 0.0001

        # 6. Scalarized Multi-Objective Reward
        reward = float(
            (portfolio_net_return * 100.0)
            - (self.lambda_risk * port_variance * 1000.0)
            - (self.lambda_cost * turnover)
        )

        # 7. Post-return asset price drift: update weights for end of day t+1
        grown_positions = target_weights * (1.0 + forward_returns)
        drifted_total = np.sum(grown_positions)
        self.current_weights = grown_positions / drifted_total if drifted_total > 0 else target_weights

        # Advance time to t+1
        self.current_step = next_t
        terminated = bool(self.current_step >= self.n_steps - 1)
        truncated = False

        obs = self._get_observation() if not terminated else np.zeros(self.obs_dim, dtype=np.float32)

        info = {
            "step": self.current_step,
            "date": str(self.dates[self.current_step].date()),
            "portfolio_value": self.portfolio_value,
            "net_return": portfolio_net_return,
            "turnover": turnover,
            "cost": cost_in_capital,
            "weights": target_weights
        }

        return obs, reward, terminated, truncated, info
