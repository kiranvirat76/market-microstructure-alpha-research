"""
validation.py
Predictive validation: prove the signal has edge BEFORE trading it.

  IC             — Spearman correlation(score[t], forward_return[t+h])
  Quintile       — mean forward return per score quintile
  Signal Decay   — IC vs horizon (tells you optimal holding period)
"""

import numpy as np
import pandas as pd
from scipy import stats


def information_coefficient(
    signal_df: pd.DataFrame,
    horizons=None,
) -> pd.DataFrame:
    """
    Spearman rank IC between score and forward mid-price return.

    Rule of thumb:
      IC > 0.02  →  weak but real
      IC > 0.05  →  practically useful
      IC > 0.10  →  strong for microstructure signals
    """
    if horizons is None:
        horizons = [1, 5, 10, 20, 50]

    rows = []
    for (product, day), grp in signal_df.groupby(["product", "file_day"]):
        grp = grp.sort_values("timestamp").copy()
        for h in horizons:
            fwd_ret = grp["mid_price"].pct_change(h).shift(-h)
            clean   = pd.DataFrame({"score": grp["score"], "fwd": fwd_ret}).dropna()
            if len(clean) < 20:
                continue
            result  = stats.spearmanr(clean["score"], clean["fwd"])
            rows.append({
                "product":  product,
                "file_day": day,
                "horizon":  h,
                "ic":       float(result.statistic),
                "p_value":  float(result.pvalue),
                "n":        len(clean),
            })
    return pd.DataFrame(rows)


def quintile_analysis(
    signal_df: pd.DataFrame,
    horizon: int = 20,
) -> pd.DataFrame:
    """
    Bin score into 5 quintiles; compute mean forward return per bin.
    Monotonic Q1→Q5 pattern = directional predictive power confirmed.
    Q1 = lowest scores (bearish), Q5 = highest (bullish).
    """
    rows = []
    for product, grp in signal_df.groupby("product"):
        grp = grp.copy()
        grp = grp[grp["mid_price"] > 0].copy()

        grp["fwd_ret"] = (
                grp["mid_price"].shift(-horizon)
                / grp["mid_price"]
                - 1
        )

        grp = grp.replace([np.inf, -np.inf], np.nan)
        grp = grp.dropna(subset=["fwd_ret"])

        try:
            grp["quintile"] = pd.qcut(grp["score"], q=5, labels=False, duplicates="drop")
        except ValueError:
            continue

        agg = grp.groupby("quintile")["fwd_ret"].agg(["mean", "std", "count"])
        for q, row in agg.iterrows():
            rows.append({
                "product":      product,
                "quintile":     int(q) + 1,   # 1-indexed for readability
                "mean_fwd_ret": float(row["mean"]),
                "std_fwd_ret":  float(row["std"]),
                "n":            int(row["count"]),
                "horizon":      horizon,
            })
    return pd.DataFrame(rows)


def signal_decay(
    signal_df: pd.DataFrame,
    max_horizon: int = 100,
    step: int = 10,
) -> pd.DataFrame:
    """
    Mean IC at each horizon from 1 to max_horizon.
    Steeper decay → shorter optimal holding period.
    Flat decay    → signal works at multiple timescales.
    """
    horizons = [1] + list(range(step, max_horizon + 1, step))
    ic_df    = information_coefficient(signal_df, horizons=horizons)

    summary = (
        ic_df.groupby(["product", "horizon"])
        .agg(mean_ic=("ic", "mean"), mean_pval=("p_value", "mean"), n=("n", "sum"))
        .reset_index()
    )
    return summary