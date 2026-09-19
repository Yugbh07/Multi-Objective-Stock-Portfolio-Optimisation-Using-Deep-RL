"""
Performance Evaluation Metrics Engine
Computes standard quantitative investment metrics:
- Cumulative Return
- Annualized Return (CAGR)
- Annualized Volatility
- Sharpe Ratio (annualized, default Rf = 2.0%)
- Maximum Drawdown (MDD)
- Calmar Ratio
- Average & Total Turnover
- Total Transaction Cost
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional


def compute_cumulative_return(portfolio_values: Union[pd.Series, np.ndarray]) -> float:
    """Cumulative percentage return from start to finish."""
    v = np.asarray(portfolio_values)
    if len(v) < 2 or v[0] == 0:
        return 0.0
    return float((v[-1] - v[0]) / v[0])


def compute_cagr(portfolio_values: Union[pd.Series, np.ndarray], periods_per_year: int = 252) -> float:
    """Compound Annual Growth Rate (CAGR)."""
    v = np.asarray(portfolio_values)
    n = len(v)
    if n < 2 or v[0] <= 0 or v[-1] <= 0:
        return 0.0
    years = (n - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float((v[-1] / v[0]) ** (1.0 / years) - 1.0)


def compute_annualized_volatility(daily_returns: Union[pd.Series, np.ndarray], periods_per_year: int = 252) -> float:
    """Annualized standard deviation of daily returns."""
    r = np.asarray(daily_returns)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def compute_sharpe_ratio(
    daily_returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Annualized Sharpe ratio.
    Sharpe = (Annualized Return - Rf) / Annualized Volatility
    """
    r = np.asarray(daily_returns)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 0.0
    annual_vol = compute_annualized_volatility(r, periods_per_year)
    if annual_vol <= 1e-8:
        return 0.0
    mean_daily = np.mean(r)
    annual_return = mean_daily * periods_per_year
    return float((annual_return - risk_free_rate) / annual_vol)


def compute_max_drawdown(portfolio_values: Union[pd.Series, np.ndarray]) -> float:
    """
    Maximum Drawdown (MDD) as a non-negative fractional loss (e.g. 0.25 = -25% decline).
    """
    v = np.asarray(portfolio_values)
    if len(v) < 2:
        return 0.0
    peak = np.maximum.accumulate(v)
    drawdowns = (peak - v) / peak
    return float(np.max(drawdowns))


def compute_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """Calmar ratio: CAGR / Max Drawdown."""
    if max_drawdown <= 1e-8:
        return 0.0
    return float(cagr / max_drawdown)


def evaluate_portfolio(
    portfolio_values: Union[pd.Series, np.ndarray],
    turnovers: Optional[Union[pd.Series, np.ndarray]] = None,
    transaction_costs: Optional[Union[pd.Series, np.ndarray]] = None,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Computes full comprehensive evaluation report dictionary.
    """
    v = np.asarray(portfolio_values)
    daily_returns = np.diff(v) / v[:-1] if len(v) > 1 else np.array([0.0])

    cum_ret = compute_cumulative_return(v)
    cagr = compute_cagr(v)
    ann_vol = compute_annualized_volatility(daily_returns)
    sharpe = compute_sharpe_ratio(daily_returns, risk_free_rate=risk_free_rate)
    mdd = compute_max_drawdown(v)
    calmar = compute_calmar_ratio(cagr, mdd)

    total_turnover = float(np.sum(turnovers)) if turnovers is not None else 0.0
    avg_turnover = float(np.mean(turnovers)) if turnovers is not None and len(turnovers) > 0 else 0.0
    total_fees = float(np.sum(transaction_costs)) if transaction_costs is not None else 0.0

    return {
        "cumulative_return": cum_ret,
        "cagr": cagr,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "calmar_ratio": calmar,
        "total_turnover": total_turnover,
        "avg_turnover": avg_turnover,
        "total_transaction_costs": total_fees
    }
