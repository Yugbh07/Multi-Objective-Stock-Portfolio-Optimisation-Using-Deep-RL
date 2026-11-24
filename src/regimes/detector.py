"""
Market Regime Detector
Detects prevailing market regimes based on strictly causal, lookahead-free rules:
- Causal Expanding Volatility Threshold: 75th percentile of historical volatility up to date t.
  (Strictly eliminates future information leakage across all periods).
- BULL: Medium-term momentum > 0 AND Price > 50-day SMA.
- BEAR: Medium-term momentum < 0 AND Price < 50-day SMA.
- HIGH_VOLATILITY: 20-day rolling volatility exceeds expanding 75th percentile threshold.
"""

import os
import logging
from typing import Optional
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
    Classifies each trading day into BULL, BEAR, or HIGH_VOLATILITY using
    strictly causal, historical-only indicators (zero lookahead leakage).
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if features_path is None:
        features_path = os.path.join(root_dir, "data", "features", "portfolio_features.csv")
    if output_path is None:
        output_path = os.path.join(root_dir, "data", "features", "market_regimes.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    features = pd.read_csv(features_path, index_col=0, parse_dates=True)

    # 1. Average market momentum (mean across assets' 20-day momentum)
    mom_cols = [c for c in features.columns if c.endswith("_mom_20")]
    avg_momentum = features[mom_cols].mean(axis=1)

    # 2. Average market price-to-SMA50 ratio
    sma50_cols = [c for c in features.columns if c.endswith("_price_to_sma50")]
    avg_sma50_ratio = features[sma50_cols].mean(axis=1)

    # 3. Average 20-day rolling market volatility
    vol_cols = [c for c in features.columns if c.endswith("_vol_20")]
    avg_volatility = features[vol_cols].mean(axis=1)

    # 4. Strictly causal expanding 75th percentile threshold (NO future information leakage)
    expanding_vol_threshold = avg_volatility.expanding(min_periods=min_warmup_periods).quantile(vol_quantile_threshold)
    warmup_val = float(avg_volatility.iloc[:min_warmup_periods].quantile(vol_quantile_threshold))
    expanding_vol_threshold = expanding_vol_threshold.fillna(warmup_val)

    logger.info("Classifying market regimes with strictly causal expanding volatility threshold...")

    regimes = []
    regime_ids = []

    for date, vol in avg_volatility.items():
        mom = avg_momentum.loc[date]
        sma_ratio = avg_sma50_ratio.loc[date]
        vol_cutoff = expanding_vol_threshold.loc[date]

        # Priority 1: High Volatility regime
        if vol >= vol_cutoff:
            regime = "HIGH_VOLATILITY"
        # Priority 2: Bull market
        elif mom > 0 and sma_ratio > 1.0:
            regime = "BULL"
        # Priority 3: Bear market
        elif mom < 0 and sma_ratio < 1.0:
            regime = "BEAR"
        else:
            # Fallback based on sign of momentum
            regime = "BULL" if mom >= 0 else "BEAR"

        regimes.append(regime)
        regime_ids.append(REGIME_MAP[regime])

    regime_df = pd.DataFrame({
        "avg_momentum": avg_momentum,
        "avg_sma50_ratio": avg_sma50_ratio,
        "avg_volatility": avg_volatility,
        "vol_threshold": expanding_vol_threshold,
        "regime": regimes,
        "regime_id": regime_ids
    }, index=features.index)

    regime_df.to_csv(output_path)
    logger.info(f"Saved market regime classifications ({regime_df.shape}) to {output_path}")

    counts = regime_df["regime"].value_counts()
    percentages = regime_df["regime"].value_counts(normalize=True) * 100
    logger.info("Market Regime Distribution:")
    for r in REGIME_MAP.keys():
        c = counts.get(r, 0)
        p = percentages.get(r, 0.0)
        logger.info(f"  - {r}: {c} days ({p:.1f}%)")

    return regime_df


if __name__ == "__main__":
    df_regimes = detect_market_regimes()
    print("\n--- Recent Market Regimes (Leak-Free) ---")
    print(df_regimes.tail(10)[["avg_volatility", "vol_threshold", "regime", "regime_id"]])
