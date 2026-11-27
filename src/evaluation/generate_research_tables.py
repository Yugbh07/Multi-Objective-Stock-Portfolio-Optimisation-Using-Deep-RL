"""
Research Experiments & Performance Evaluation Generator

Evaluates portfolio strategies on a shared out-of-sample period:
1. Equal Weight
2. Buy & Hold
3. Markowitz MVO
4. PPO Without Regime
5. Regime-Aware PPO

Outputs:
- Numeric CSV results
- Markdown research report
- Portfolio growth chart
- Drawdown chart

All strategies are evaluated over the same out-of-sample dates.
"""

import os
import sys
import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backtesting.engine import BacktestEngine
from src.models.agent import PPOAgent


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


INITIAL_CAPITAL = 1_000_000.0
TRANSACTION_COST_RATE = 0.0010
DEFAULT_TEST_START = "2024-01-01"


# ============================================================
# DATA LOADING & VALIDATION
# ============================================================

def load_dataset(path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} file not found: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    if df.empty:
        raise ValueError(f"{name} dataset is empty.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must have a DatetimeIndex.")

    if df.index.isna().any():
        raise ValueError(f"{name} contains invalid dates.")

    if df.index.has_duplicates:
        raise ValueError(f"{name} contains duplicate dates.")

    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{name} dates are not chronological.")

    if df.columns.has_duplicates:
        raise ValueError(f"{name} contains duplicate columns.")

    if df.isna().any().any():
        raise ValueError(f"{name} contains NaN values.")

    numeric = df.select_dtypes(include=[np.number])

    if numeric.empty:
        raise ValueError(f"{name} has no numeric columns.")

    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{name} contains infinite values.")

    logger.info(
        "Loaded %s: %d rows, %d columns, %s to %s",
        name,
        len(df),
        len(df.columns),
        df.index.min().date(),
        df.index.max().date()
    )

    return df


def load_research_data():
    prices = load_dataset(
        os.path.join(
            PROJECT_ROOT,
            "data",
            "processed",
            "clean_prices.csv"
        ),
        "Prices"
    )

    features = load_dataset(
        os.path.join(
            PROJECT_ROOT,
            "data",
            "features",
            "portfolio_features.csv"
        ),
        "Features"
    )

    regimes = load_dataset(
        os.path.join(
            PROJECT_ROOT,
            "data",
            "features",
            "market_regimes.csv"
        ),
        "Regimes"
    )

    if not prices.select_dtypes(include=[np.number]).gt(0).all().all():
        raise ValueError("Price dataset contains non-positive prices.")

    if "regime_id" not in regimes.columns:
        raise ValueError("Regime dataset missing regime_id column.")

    tickers = list(prices.columns)

    required_feature_columns = []

    for ticker in tickers:
        required_feature_columns.extend([
            f"{ticker}_return",
            f"{ticker}_vol_20"
        ])

    missing_features = [
        col for col in required_feature_columns
        if col not in features.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required feature columns: {missing_features}"
        )

    # Align datasets on shared dates and shared assets.
    common_dates = (
        prices.index
        .intersection(features.index)
        .intersection(regimes.index)
    )

    if len(common_dates) == 0:
        raise ValueError("No common dates across research datasets.")

    prices = prices.loc[common_dates].copy()
    features = features.loc[common_dates].copy()
    regimes = regimes.loc[common_dates].copy()

    if not prices.index.equals(features.index):
        raise RuntimeError("Price-feature date alignment failed.")

    if not prices.index.equals(regimes.index):
        raise RuntimeError("Price-regime date alignment failed.")

    return prices, features, regimes


# ============================================================
# STRATEGY EVALUATION
# ============================================================

def run_all_strategies(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    regimes: pd.DataFrame,
    test_start_date: str
):
    test_start = pd.Timestamp(test_start_date)

    test_mask = prices.index >= test_start

    test_dates = prices.index[test_mask]

    if len(test_dates) < 3:
        raise ValueError(
            f"Insufficient out-of-sample data from {test_start_date}."
        )

    test_prices = prices.loc[test_dates].copy()
    test_features = features.loc[test_dates].copy()
    test_regimes = regimes.loc[test_dates].copy()

    logger.info(
        "Out-of-sample period: %s to %s (%d observations)",
        test_dates.min().date(),
        test_dates.max().date(),
        len(test_dates)
    )

    engine = BacktestEngine(
        test_prices,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_rate=TRANSACTION_COST_RATE
    )

    results = {}

    # Equal Weight
    logger.info("Running Equal Weight...")
    results["Equal Weight"] = engine.backtest_equal_weight()

    # Buy & Hold
    logger.info("Running Buy & Hold...")
    results["Buy & Hold"] = engine.backtest_buy_and_hold()

        # Markowitz
    logger.info("Running Markowitz MVO...")
    results["Markowitz MVO"] = engine.backtest_markowitz(
        lookback_window=60,
        rebalance_interval=21,
        warmup_prices=prices.loc[
            prices.index < test_dates[0]
        ]
    )

    # PPO without regime
    logger.info("Running PPO Without Regime...")
    agent_no_regime = PPOAgent(include_regime=False)

    weights_no_regime = agent_no_regime.generate_weights_for_dataset(
        test_prices,
        test_features,
        test_regimes
    )

    results["PPO (No Regime)"] = engine.run_strategy_weights(
        weights_no_regime,
        strategy_name="PPO (No Regime)"
    )

    # Regime-aware PPO
    logger.info("Running Regime-Aware PPO...")
    agent_regime = PPOAgent(include_regime=True)

    weights_regime = agent_regime.generate_weights_for_dataset(
        test_prices,
        test_features,
        test_regimes
    )

    results["Regime-Aware PPO"] = engine.run_strategy_weights(
        weights_regime,
        strategy_name="Regime-Aware PPO"
    )

    # Validate evaluation history date coverage.
    for name, result in results.items():
        history = result.get("history")

        if history is None or history.empty:
            raise ValueError(f"{name} returned empty backtest history.")

        if "portfolio_value" not in history.columns:
            raise ValueError(
                f"{name} history missing portfolio_value."
            )

        if not np.isfinite(
            history["portfolio_value"].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{name} produced non-finite portfolio values."
            )

        logger.info(
            "%s evaluated over %s to %s (%d rows)",
            name,
            history.index.min().date(),
            history.index.max().date(),
            len(history)
        )

    return results, test_prices


# ============================================================
# RESULTS TABLE
# ============================================================

def build_results_table(results: dict) -> pd.DataFrame:
    rows = []

    for name, result in results.items():
        metrics = result["metrics"]

        rows.append({
            "Strategy": name,
            "Cumulative Return": metrics["cumulative_return"],
            "CAGR": metrics["cagr"],
            "Sharpe Ratio": metrics["sharpe_ratio"],
            "Annualized Volatility": metrics["annualized_volatility"],
            "Maximum Drawdown": metrics["max_drawdown"],
            "Calmar Ratio": metrics["calmar_ratio"],
            "Total Turnover": metrics["total_turnover"],
            "Transaction Costs": metrics["total_transaction_costs"]
        })

    summary = pd.DataFrame(rows)

    numeric_columns = [
        col for col in summary.columns
        if col != "Strategy"
    ]

    summary[numeric_columns] = summary[numeric_columns].astype(float)

    if not np.isfinite(
        summary[numeric_columns].to_numpy()
    ).all():
        raise ValueError("Summary table contains non-finite metrics.")

    return summary


# ============================================================
# RESEARCH TAKEAWAYS
# ============================================================

def generate_takeaways(summary: pd.DataFrame) -> list:
    takeaways = []

    best_sharpe = summary.loc[
        summary["Sharpe Ratio"].idxmax()
    ]

    best_cagr = summary.loc[
        summary["CAGR"].idxmax()
    ]

    lowest_drawdown = summary.loc[
        summary["Maximum Drawdown"].idxmin()
    ]

    lowest_volatility = summary.loc[
        summary["Annualized Volatility"].idxmin()
    ]

    takeaways.append(
        f"Highest Sharpe Ratio: {best_sharpe['Strategy']} "
        f"({best_sharpe['Sharpe Ratio']:.3f})."
    )

    takeaways.append(
        f"Highest CAGR: {best_cagr['Strategy']} "
        f"({best_cagr['CAGR']:.2%})."
    )

    takeaways.append(
        f"Lowest maximum drawdown: {lowest_drawdown['Strategy']} "
        f"({lowest_drawdown['Maximum Drawdown']:.2%})."
    )

    takeaways.append(
        f"Lowest annualized volatility: {lowest_volatility['Strategy']} "
        f"({lowest_volatility['Annualized Volatility']:.2%})."
    )

    ppo_row = summary[
        summary["Strategy"] == "Regime-Aware PPO"
    ]

    no_regime_row = summary[
        summary["Strategy"] == "PPO (No Regime)"
    ]

    if not ppo_row.empty and not no_regime_row.empty:
        ppo = ppo_row.iloc[0]
        no_regime = no_regime_row.iloc[0]

        sharpe_difference = (
            ppo["Sharpe Ratio"]
            - no_regime["Sharpe Ratio"]
        )

        drawdown_difference = (
            ppo["Maximum Drawdown"]
            - no_regime["Maximum Drawdown"]
        )

        takeaways.append(
            "Regime-aware PPO vs PPO without regime: "
            f"Sharpe difference = {sharpe_difference:+.3f}; "
            f"maximum drawdown difference = {drawdown_difference:+.2%}."
        )

    takeaways.append(
        "These are historical out-of-sample observations and do not "
        "establish statistical significance or guarantee future performance."
    )

    return takeaways


# ============================================================
# CHARTS
# ============================================================

def plot_portfolio_growth(results: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, result in results.items():
        history = result["history"]

        ax.plot(
            history.index,
            history["portfolio_value"],
            label=name,
            linewidth=1.8
        )

    ax.set_title("Out-of-Sample Portfolio Growth")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (INR)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()

    path = os.path.join(output_dir, "cumulative_returns.png")

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved growth chart: %s", path)


def plot_drawdowns(results: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(12, 5))

    for name, result in results.items():
        history = result["history"]

        values = history["portfolio_value"].astype(float)

        peaks = values.cummax()

        drawdowns = (values / peaks - 1.0) * 100.0

        ax.plot(
            history.index,
            drawdowns,
            label=name,
            linewidth=1.7
        )

    ax.set_title("Out-of-Sample Drawdown Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()

    path = os.path.join(output_dir, "drawdowns.png")

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved drawdown chart: %s", path)


# ============================================================
# REPORT GENERATION
# ============================================================

def save_research_report(
    summary: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    output_dir: str
):
    csv_path = os.path.join(
        output_dir,
        "experiment_results.csv"
    )

    md_path = os.path.join(
        output_dir,
        "experiment_results_table.md"
    )

    summary.to_csv(csv_path, index=False)

    display_table = summary.copy()

    percentage_columns = [
        "Cumulative Return",
        "CAGR",
        "Annualized Volatility",
        "Maximum Drawdown"
    ]

    for col in percentage_columns:
        display_table[col] = display_table[col].map(
            lambda x: f"{x:.2%}"
        )

    for col in [
        "Sharpe Ratio",
        "Calmar Ratio",
        "Total Turnover"
    ]:
        display_table[col] = display_table[col].map(
            lambda x: f"{x:.3f}"
        )

    display_table["Transaction Costs"] = (
        display_table["Transaction Costs"]
        .map(lambda x: f"INR {x:,.2f}")
    )

    takeaways = generate_takeaways(summary)

    with open(md_path, "w", encoding="utf-8") as file:
        file.write("# Out-of-Sample Portfolio Evaluation\n\n")

        file.write(
            f"**Evaluation period:** {test_start.date()} to "
            f"{test_end.date()}\n\n"
        )

        file.write(
            f"**Initial capital:** INR {INITIAL_CAPITAL:,.2f}\n\n"
        )

        file.write(
            f"**Transaction cost rate:** "
            f"{TRANSACTION_COST_RATE:.4%}\n\n"
        )

        file.write("## Performance Comparison\n\n")

        file.write(display_table.to_markdown(index=False))

        file.write("\n\n## Data-Driven Findings\n\n")

        for takeaway in takeaways:
            file.write(f"- {takeaway}\n")

        file.write(
            "\n\n## Methodological Limitations\n\n"
            "- Historical performance does not guarantee future results.\n"
            "- No statistical significance testing was conducted.\n"
            "- No sensitivity analysis was conducted in this script.\n"
            "- Results depend on the supplied data and trained models.\n"
        )

    logger.info("Saved CSV report: %s", csv_path)
    logger.info("Saved Markdown report: %s", md_path)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_research_experiments(
    test_start_date: str = DEFAULT_TEST_START,
    output_dir: Optional[str] = None
) -> pd.DataFrame:

    if output_dir is None:
        output_dir = os.path.join(
            PROJECT_ROOT,
            "data",
            "research_results"
        )

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading research datasets...")

    prices, features, regimes = load_research_data()

    logger.info("Evaluating strategies...")

    results, test_prices = run_all_strategies(
        prices,
        features,
        regimes,
        test_start_date
    )

    summary = build_results_table(results)

    actual_start = test_prices.index.min()
    actual_end = test_prices.index.max()

    save_research_report(
        summary,
        actual_start,
        actual_end,
        output_dir
    )

    plot_portfolio_growth(
        results,
        output_dir
    )

    plot_drawdowns(
        results,
        output_dir
    )

    logger.info(
        "Research pipeline completed successfully. "
        "Evaluation period: %s to %s",
        actual_start.date(),
        actual_end.date()
    )

    return summary


if __name__ == "__main__":
    results_df = run_research_experiments()

    print("\n" + "=" * 80)
    print("RESEARCH EVALUATION RESULTS")
    print("=" * 80)

    print(results_df.to_string(index=False))

    print("\nResearch evaluation completed.")