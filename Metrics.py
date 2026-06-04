"""
metrics.py
Computes all market microstructure metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────
# 1. Lee-Ready Trade Direction Classification
# ─────────────────────────────────────────────

def lee_ready_direction(trade_price: pd.Series, mid_price: pd.Series) -> pd.Series:
    """
    Lee-Ready (1991) algorithm.
    If trade price > mid  → buyer-initiated (+1)
    If trade price < mid  → seller-initiated (-1)
    If trade price == mid → tick test (use price change direction)
    Returns series of +1 / -1.
    """
    direction = np.where(
        trade_price > mid_price,  1,
        np.where(trade_price < mid_price, -1, 0)
    )
    # For zero cases: tick test (look at price change)
    price_change = trade_price.diff()
    tick = np.where(price_change > 0, 1, np.where(price_change < 0, -1, np.nan))
    # forward-fill last known tick for tie-breaking
    tick_series = pd.Series(tick).ffill().fillna(1).values

    direction = np.where(direction == 0, tick_series, direction)
    return pd.Series(direction.astype(int), index=trade_price.index)


# ─────────────────────────────────────────────
# 2. Effective Spread
# ─────────────────────────────────────────────

def effective_spread(trade_price: pd.Series, mid_price: pd.Series,
                     direction: pd.Series) -> pd.Series:
    """
    Effective spread = 2 * direction * (trade_price - mid_price)
    Measures actual transaction cost paid by liquidity takers.
    """
    return 2 * direction * (trade_price - mid_price)


def effective_spread_bps(eff_spread: pd.Series, mid_price: pd.Series) -> pd.Series:
    """Effective spread in basis points."""
    return (eff_spread / mid_price) * 10_000


# ─────────────────────────────────────────────
# 3. Kyle's Lambda (Price Impact)
# ─────────────────────────────────────────────

def kyles_lambda(merged_trades: pd.DataFrame) -> dict:
    """
    Regress mid-price change on signed order flow:
        ΔP = λ × Q_signed + ε
    λ = Kyle's lambda; larger → more illiquid.
    Returns dict with lambda, r2, pvalue per product.
    """
    results = {}

    for product, grp in merged_trades.groupby("symbol"):
        grp = grp.sort_values(["file_day", "timestamp"]).copy()
        grp["delta_mid"] = grp["mid_price"].diff()
        grp["signed_qty"] = grp["quantity"] * grp["direction"]
        clean = grp.dropna(subset=["delta_mid", "signed_qty"])

        if len(clean) < 10:
            continue

        x = clean["signed_qty"].values.reshape(-1, 1)
        y = clean["delta_mid"].values

        slope, intercept, r, p, se = stats.linregress(x.ravel(), y)

        results[product] = {
            "lambda":    slope,
            "intercept": intercept,
            "r_squared": r ** 2,
            "p_value":   p,
            "n_trades":  len(clean),
        }

    return results


# ─────────────────────────────────────────────
# 4. Roll's Spread Estimator
# ─────────────────────────────────────────────

def rolls_estimator(trade_prices: pd.Series) -> float:
    """
    Roll (1984): c = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))
    Estimates effective half-spread from serial covariance of price changes.
    If covariance is positive (no mean reversion), returns NaN.
    """
    dp = trade_prices.diff().dropna()
    cov = dp.cov(dp.shift(1).dropna().reindex(dp.index))

    if cov >= 0:
        return np.nan   # Roll estimator requires negative serial covariance
    return 2 * np.sqrt(-cov)


def rolls_per_product(trades: pd.DataFrame) -> pd.DataFrame:
    """Compute Roll's estimator per product per day."""
    rows = []
    for (product, day), grp in trades.groupby(["symbol", "file_day"]):
        grp_sorted = grp.sort_values("timestamp")
        roll = rolls_estimator(grp_sorted["price"])
        rows.append({
            "product": product,
            "file_day": day,
            "rolls_spread": roll,
            "n_trades": len(grp_sorted),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 5. Amihud Illiquidity
# ─────────────────────────────────────────────

def amihud_illiquidity(ob: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """
    Amihud (2002) illiquidity ratio:
        ILLIQ = |R| / Volume
    Computed as rolling ratio using mid_price returns and total depth as proxy for volume.
    Higher ILLIQ → less liquid → price moves more per unit of volume.
    """
    frames = []
    for (product, day), grp in ob.groupby(["product", "file_day"]):
        grp = grp.sort_values("timestamp").copy()
        grp["mid_return"] = grp["mid_price"].pct_change().abs()
        total_vol = grp["total_bid_depth"] + grp["total_ask_depth"]
        grp["amihud"] = grp["mid_return"] / total_vol.replace(0, np.nan)
        grp["amihud_roll"] = grp["amihud"].rolling(window, min_periods=10).mean()
        frames.append(grp)
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────
# 6. VWAP
# ─────────────────────────────────────────────

def compute_vwap(trades: pd.DataFrame) -> pd.DataFrame:
    """Compute daily VWAP per product."""
    rows = []
    for (product, day), grp in trades.groupby(["symbol", "file_day"]):
        pv = (grp["price"] * grp["quantity"]).sum()
        vol = grp["quantity"].sum()
        rows.append({
            "product":   product,
            "file_day":  day,
            "vwap":      pv / vol if vol > 0 else np.nan,
            "total_vol": vol,
            "n_trades":  len(grp),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 7. Intraday Spread Pattern
# ─────────────────────────────────────────────

def intraday_spread_pattern(ob: pd.DataFrame, buckets: int = 20) -> pd.DataFrame:
    """
    Split trading day into N time buckets; compute mean spread per bucket.
    Classic U-shape: wide at open/close, narrow at midday.
    """
    ob = ob.copy()
    ts_max = ob["timestamp"].max()
    ob["time_bucket"] = pd.cut(ob["timestamp"], bins=buckets, labels=False)

    result = (ob.groupby(["product", "time_bucket"])
                .agg(mean_spread=("spread", "mean"),
                     mean_rel_spread=("relative_spread", "mean"),
                     mean_obi=("obi", "mean"))
                .reset_index())
    return result