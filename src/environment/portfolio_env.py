"""
Gymnasium Portfolio Management Environment
Custom reinforcement learning environment for dynamic asset allocation:
- Observation Space: Recent asset returns, rolling volatilities, market regime one-hot,
  current portfolio weights, and cash reserve fraction.
- Action Space: Continuous action vector mapped to valid portfolio simplex via Softmax.
- Multi-Objective Reward: Return - Risk Penalty (Portfolio Variance) - Turnover Cost.
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


class PortfolioEnv(gym.Env):
    """
    Gymnasium-compliant multi-asset portfolio rebalancing environment.
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

        # Returns matrix for all assets
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

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax to produce valid probability simplex weights."""
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

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

        # 1. Asset returns
        returns = self.features_df[self.return_cols].iloc[step].values.astype(np.float32)

        # 2. Asset rolling volatilities
        vols = self.features_df[self.vol_cols].iloc[step].values.astype(np.float32)

        obs_components = [returns, vols]

        # 3. Market regime one-hot (if enabled)
        if self.include_regime:
            regime_vec = self._get_regime_one_hot(step)
            obs_components.append(regime_vec)

        # 4. Current portfolio weights and cash
        obs_components.append(self.current_weights.astype(np.float32))
        obs_components.append(np.array([self.cash_ratio], dtype=np.float32))

        obs = np.concatenate(obs_components).astype(np.float32)
        # Ensure finite observation
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
        Executes one trading step:
        1. Action mapped to target weights via Softmax.
        2. Transaction costs and turnover computed.
        3. Portfolio returns and variance computed.
        4. Multi-objective reward formulated.
        """
        # 1. Normalize action to valid weights
        target_weights = self._softmax(action)

        # 2. Current day asset returns
        step_returns = self.returns_matrix[self.current_step]

        # Drifted weights before rebalancing
        grown_positions = self.current_weights * (1.0 + step_returns)
        drifted_total = np.sum(grown_positions)
        drifted_weights = grown_positions / drifted_total if drifted_total > 0 else self.current_weights

        # 3. Turnover and transaction cost
        turnover = float(np.sum(np.abs(target_weights - drifted_weights)))
        cost = turnover * self.transaction_cost_rate

        # 4. Portfolio return net of transaction cost
        portfolio_gross_return = float(np.dot(target_weights, step_returns))
        portfolio_net_return = portfolio_gross_return - cost

        # Update portfolio value
        self.portfolio_value = self.portfolio_value * (1.0 + portfolio_net_return)

        # 5. Risk (Portfolio Variance estimate)
        # Use recent 20-day returns covariance
        lookback_start = max(0, self.current_step - self.lookback_window)
        recent_slice = self.returns_matrix[lookback_start:self.current_step + 1]
        if len(recent_slice) > 2:
            cov = np.cov(recent_slice, rowvar=False)
            port_variance = float(np.dot(target_weights.T, np.dot(cov, target_weights)))
        else:
            port_variance = 0.0001

        # 6. Multi-Objective Reward: Return - lambda_risk * Variance - lambda_cost * Turnover
        reward = float(
            (portfolio_net_return * 100.0)
            - (self.lambda_risk * port_variance * 1000.0)
            - (self.lambda_cost * turnover)
        )

        # Update current weights for next step
        self.current_weights = target_weights

        # Advance step
        self.current_step += 1
        terminated = bool(self.current_step >= self.n_steps - 1)
        truncated = False

        obs = self._get_observation() if not terminated else np.zeros(self.obs_dim, dtype=np.float32)

        info = {
            "step": self.current_step,
            "date": str(self.dates[min(self.current_step, self.n_steps - 1)].date()),
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_net_return,
            "turnover": turnover,
            "weights": target_weights.tolist()
        }

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    prices = pd.read_csv("data/processed/clean_prices.csv", index_col=0, parse_dates=True)
    features = pd.read_csv("data/features/portfolio_features.csv", index_col=0, parse_dates=True)
    regimes = pd.read_csv("data/features/market_regimes.csv", index_col=0, parse_dates=True)

    env = PortfolioEnv(prices, features, regimes)
    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    print("Sample observation:", np.round(obs[:6], 4))

    # Take a random step
    action = env.action_space.sample()
    next_obs, reward, term, trunc, info = env.step(action)
    print(f"Step Reward: {reward:.4f} | Net Return: {info['portfolio_return']*100:.2f}% | Turnover: {info['turnover']:.4f}")
    print("Assigned Weights:", np.round(info['weights'], 4))
