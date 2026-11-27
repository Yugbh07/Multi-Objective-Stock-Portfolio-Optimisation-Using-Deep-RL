
"""
Market Regime Detector

Classifies the portfolio universe into:
- BULL
- BEAR
- HIGH_VOLATILITY

Uses historical-only volatility thresholds to avoid lookahead leakage.
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

REGIME_MAP = {
    "BULL": 0,
    "BEAR": 1,
    "HIGH_VOLATILITY": 2
}

INV_REGIME_MAP = {v: k for k, v in REGIME_MAP.items()}


def detect_market_regimes(
    features_path: Optional[str] = None,
    output_path: Optional[str] = None,
    vol_quantile_threshold: float = 0.75,
    min_warmup_periods: int = 60
) -> pd.DataFrame:

    """
    Classifies market regimes using causal momentum, SMA and volatility features.

    Volatility threshold at date t is calculated exclusively using observations
    available before t. No future observations are used.

    Invalid, conflicting, or insufficient-history observations are assigned
    regime=None and regime_id=-1.
    """

    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    if features_path is None:
        features_path = os.path.join(
            root_dir,
            "data",
            "features",
            "portfolio_features.csv"
        )

    if output_path is None:
        output_path = os.path.join(
            root_dir,
            "data",
            "features",
            "market_regimes.csv"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    features = pd.read_csv(
        features_path,
        index_col=0,
        parse_dates=True
    )

    # Input validation
    if features.empty:
        raise ValueError("Feature dataset is empty.")

    if not 0 < vol_quantile_threshold < 1:
        raise ValueError("vol_quantile_threshold must be between 0 and 1.")

    if min_warmup_periods < 1:
        raise ValueError("min_warmup_periods must be at least 1.")

    if features.index.has_duplicates:
        raise ValueError("Duplicate dates found in feature dataset.")

    features = features.sort_index()

    # Required feature columns
    mom_cols = [
        c for c in features.columns
        if c.endswith("_mom_20")
    ]

    sma50_cols = [
        c for c in features.columns
        if c.endswith("_price_to_sma50")
    ]

    vol_cols = [
        c for c in features.columns
        if c.endswith("_vol_20")
    ]

    if not mom_cols or not sma50_cols or not vol_cols:
        raise ValueError(
            "Required regime features missing: "
            "mom_20, price_to_sma50, or vol_20."
        )

    # 1. Aggregate portfolio-universe momentum
    avg_momentum = features[mom_cols].mean(axis=1)

    # 2. Aggregate portfolio-universe price-to-SMA50 ratio
    avg_sma50_ratio = features[sma50_cols].mean(axis=1)

    # 3. Aggregate portfolio-universe volatility
    avg_volatility = features[vol_cols].mean(axis=1)

    # 4. Causal expanding volatility threshold
    # Today's threshold uses only prior observations.
    expanding_vol_threshold = (
        avg_volatility.shift(1)
        .expanding(min_periods=min_warmup_periods)
        .quantile(vol_quantile_threshold)
    )

    logger.info(
        "Classifying regimes using causal expanding volatility threshold."
    )

    regimes = []
    regime_ids = []

    # 5. Regime classification
    for date, vol in avg_volatility.items():

        mom = avg_momentum.loc[date]
        sma_ratio = avg_sma50_ratio.loc[date]
        vol_cutoff = expanding_vol_threshold.loc[date]

        # Reject invalid observations and warm-up periods
        if (
            pd.isna(vol)
            or pd.isna(mom)
            or pd.isna(sma_ratio)
            or pd.isna(vol_cutoff)
        ):
            regimes.append(None)
            regime_ids.append(-1)
            continue

        # High-volatility regime
        if vol > vol_cutoff:
            regime = "HIGH_VOLATILITY"

        # Bull trend
        elif mom > 0 and sma_ratio > 1.0:
            regime = "BULL"

        # Bear trend
        elif mom < 0 and sma_ratio < 1.0:
            regime = "BEAR"

        # Conflicting signals: remain unclassified
        else:
            regime = None

        regimes.append(regime)

        regime_ids.append(
            REGIME_MAP[regime] if regime is not None else -1
        )

    # 6. Construct output dataframe
    regime_df = pd.DataFrame(
        {
            "avg_momentum": avg_momentum,
            "avg_sma50_ratio": avg_sma50_ratio,
            "avg_volatility": avg_volatility,
            "vol_threshold": expanding_vol_threshold,
            "regime": regimes,
            "regime_id": regime_ids
        },
        index=features.index
    )

    # 7. Persist classifications
    regime_df.to_csv(output_path)

    logger.info(
        "Saved market regime classifications %s to %s",
        regime_df.shape,
        output_path
    )

    # 8. Report regime distribution
    counts = regime_df["regime"].value_counts()
    percentages = regime_df["regime"].value_counts(normalize=True) * 100

    logger.info("Market Regime Distribution:")

    for regime_name in REGIME_MAP:
        count = counts.get(regime_name, 0)
        percentage = percentages.get(regime_name, 0.0)

        logger.info(
            "  - %s: %d days (%.2f%%)",
            regime_name,
            count,
            percentage
        )

    invalid_count = (regime_df["regime_id"] == -1).sum()

    logger.info(
        "Unclassified observations: %d (%.2f%%)",
        invalid_count,
        invalid_count / len(regime_df) * 100
    )

    return regime_df


if __name__ == "__main__":

    df_regimes = detect_market_regimes()

    print("\n--- Recent Market Regimes ---")

    print(
        df_regimes.tail(10)[
            [
                "avg_volatility",
                "vol_threshold",
                "regime",
                "regime_id"
            ]
        ]
    )