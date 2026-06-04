"""
signals.py
Three alpha modules + combined score.
All signals normalised to approximately [-1, +1].
"""

import numpy as np
import pandas as pd


# ── 1. OBI Signal ─────────────────────────────────────────────────────────────
def obi_signal(ob: pd.DataFrame, smooth: int = 50) -> pd.DataFrame:
    """
    Smoothed Order Book Imbalance.
    OBI already ∈ (-1, 1) from loader; rolling mean reduces noise.
    """
    frames = []
    for (product, day), grp in ob.groupby(["product", "file_day"]):
        grp = grp.sort_values("timestamp").copy()
        grp["obi_signal"] = (
            grp["obi"].rolling(smooth, min_periods=5).mean().fillna(0)
        )
        frames.append(grp[["product", "file_day", "timestamp", "obi_signal"]])
    return pd.concat(frames, ignore_index=True)


# ── 2. Liquidity Shock Signal ─────────────────────────────────────────────────
def liquidity_shock_signal(
    ob: pd.DataFrame,
    short_win: int = 20,
    long_win: int = 200,
) -> pd.DataFrame:
    """
    Detects sudden spread widening vs baseline.
    signal = -(short_ma / long_ma - 1), clipped to [-1, +1].

    Spread widens  → signal < 0  (high execution cost, penalise entry).
    Spread narrows → signal > 0  (cheap to trade, amplify entry).
    """
    frames = []
    for (product, day), grp in ob.groupby(["product", "file_day"]):
        grp  = grp.sort_values("timestamp").copy()
        s_ma = grp["spread"].rolling(short_win, min_periods=5).mean()
        l_ma = grp["spread"].rolling(long_win,  min_periods=20).mean()
        ratio = (s_ma / l_ma.replace(0, np.nan) - 1).fillna(0)
        grp["liq_shock_signal"] = (-ratio).clip(-1, 1)
        frames.append(grp[["product", "file_day", "timestamp", "liq_shock_signal"]])
    return pd.concat(frames, ignore_index=True)


# ── 3. Trade Flow Signal ──────────────────────────────────────────────────────
def trade_flow_signal(
    merged_trades: pd.DataFrame,
    ob: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Rolling net signed-volume fraction, forward-filled onto the ob snapshot grid.
    Uses Lee-Ready direction column already on merged_trades.

    tf_raw = roll_net_vol / roll_total_vol  ∈ [-1, +1]
    """
    frames = []
    for product in ob["product"].unique():
        for day in ob["file_day"].unique():
            o_grp = (
                ob[(ob["product"] == product) & (ob["file_day"] == day)]
                [["timestamp"]]
                .sort_values("timestamp")
                .copy()
            )
            if o_grp.empty:
                continue

            t_grp = (
                merged_trades[
                    (merged_trades["symbol"] == product)
                    & (merged_trades["file_day"] == day)
                ]
                .sort_values("timestamp")
                .copy()
            )

            if t_grp.empty:
                o_grp["tf_signal"] = 0.0
                o_grp["product"]   = product
                o_grp["file_day"]  = day
                frames.append(o_grp[["product", "file_day", "timestamp", "tf_signal"]])
                continue

            t_grp["signed_vol"] = t_grp["quantity"] * t_grp["direction"]
            roll_net = t_grp["signed_vol"].rolling(window, min_periods=3).sum()
            roll_tot = t_grp["quantity"].rolling(window, min_periods=3).sum()
            t_grp["tf_raw"] = (roll_net / roll_tot.replace(0, np.nan)).fillna(0)

            merged = pd.merge_asof(
                o_grp,
                t_grp[["timestamp", "tf_raw"]],
                on="timestamp",
                direction="backward",
            ).fillna(0)

            merged["product"]   = product
            merged["file_day"]  = day
            merged["tf_signal"] = merged["tf_raw"].clip(-1, 1)
            frames.append(merged[["product", "file_day", "timestamp", "tf_signal"]])

    return pd.concat(frames, ignore_index=True)


# ── 5. Combined Score ─────────────────────────────────────────────────────────
def build_combined_signal(
    ob: pd.DataFrame,
    merged_trades: pd.DataFrame,
    w_obi: float = 0.4,
    w_liq: float = 0.3,
    w_tf:  float = 0.3,
) -> pd.DataFrame:
    """
    Weighted combination of the three signals.
    Returns ob-aligned DataFrame with individual signals + 'score' ∈ [-1, +1].

        score = w_obi * obi_signal
              + w_liq * liq_shock_signal
              + w_tf  * tf_signal
    """
    obi_df = obi_signal(ob)
    liq_df = liquidity_shock_signal(ob)
    tf_df  = trade_flow_signal(merged_trades, ob)

    base = ob[[
        "product", "file_day", "timestamp",
        "mid_price", "spread", "obi",
        "total_bid_depth", "total_ask_depth",
    ]].copy()

    base = base.merge(obi_df, on=["product", "file_day", "timestamp"], how="left")
    base = base.merge(liq_df, on=["product", "file_day", "timestamp"], how="left")
    base = base.merge(tf_df,  on=["product", "file_day", "timestamp"], how="left")

    for col in ["obi_signal", "liq_shock_signal", "tf_signal"]:
        base[col] = base[col].fillna(0)

    base["score"] = (
        w_obi * base["obi_signal"]
        + w_liq * base["liq_shock_signal"]
        + w_tf  * base["tf_signal"]
    )
    base["score"] = (
        base.groupby("product")["score"]
        .transform(lambda x: x.ewm(span=20).mean())
    )

    return base