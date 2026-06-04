"""
backtest.py
Signal → position → execution → PnL.

Two cost components:
  1. Spread cost   : cost_multiplier * spread * |Δpos|         (always paid)
  2. Market impact : impact_coeff    * sqrt(|Δpos|)            (sqrt model)

sqrt model: impact grows sub-linearly with size — standard in institutional practice.
Set impact_coeff=0 to disable (original behaviour preserved).

No look-ahead: position[t] set from signal[t],
PnL earned from holding position[t-1] into period t.
"""

import numpy as np
import pandas as pd


def run_backtest(
    signal_df: pd.DataFrame,
    position_limit:  int   = 50,
    entry_threshold: float = 0.20,
    exit_threshold:  float = 0.10,
    cost_multiplier: float = 0.50,
    impact_coeff:    float = 0.00,   # sqrt market impact coefficient
) -> pd.DataFrame:
    """
    Per-product, per-day simulation.

    Position sizing
    ---------------
    target   = score * position_limit    (proportional to signal strength)
    position = clip(target, -limit, +limit)
    Entry only if |score| >= entry_threshold.
    Hold  if exit_threshold <= |score| < entry_threshold.
    Exit  if |score| <  exit_threshold.

    PnL
    ---
    gross[t]        = position[t-1] * (mid[t] - mid[t-1])
    spread_cost[t]  = cost_multiplier * spread[t] * |Δpos[t]|
    impact_cost[t]  = impact_coeff    * sqrt(|Δpos[t]|)
    trans_cost[t]   = spread_cost[t] + impact_cost[t]
    net[t]          = gross[t] - trans_cost[t]

    Added columns: position, delta_pos, gross_pnl, spread_cost,
                   impact_cost, trans_cost, net_pnl, cum_gross, cum_net
    """
    results = []

    for (product, day), grp in signal_df.groupby(["product", "file_day"]):
        grp = (
            grp.sort_values("timestamp")
            .dropna(subset=["score", "mid_price", "spread"])
            .copy()
        )
        n = len(grp)
        if n < 5:
            continue

        score  = grp["score"].values
        mid    = grp["mid_price"].values
        spread = grp["spread"].values

        # ── Position ──────────────────────────────────────────────
        position = np.zeros(n)
        minimum_hold = 20
        hold_counter = 0
        for t in range(1, n):
            # Force hold if already in a position
            if position[t - 1] != 0 and hold_counter < minimum_hold:
                position[t] = position[t - 1]
                hold_counter += 1
                continue

            abs_s = abs(score[t])
            if abs_s >= entry_threshold:
                new_pos = np.clip(
                    score[t] * position_limit,
                    -position_limit,
                    position_limit,
                )
                # New trade entered
                if new_pos != position[t - 1]:
                    hold_counter = 0
                position[t] = new_pos
            elif abs_s < exit_threshold:
                position[t] = 0.0
                hold_counter = 0
            else:
                position[t] = position[t - 1]

        # ── PnL (no look-ahead) ───────────────────────────────────
        delta_mid   = np.diff(mid)
        gross_pnl   = np.concatenate([[0.0], position[:-1] * delta_mid])

        delta_pos   = np.concatenate([[0.0], np.diff(position)])
        abs_dp      = np.abs(delta_pos)

        spread_cost = cost_multiplier * spread * abs_dp
        impact_cost = impact_coeff * np.sqrt(abs_dp)      # sqrt model
        trans_cost  = spread_cost + impact_cost
        net_pnl     = gross_pnl - trans_cost

        grp["position"]    = position
        grp["delta_pos"]   = delta_pos
        grp["gross_pnl"]   = gross_pnl
        grp["spread_cost"] = spread_cost
        grp["impact_cost"] = impact_cost
        grp["trans_cost"]  = trans_cost
        grp["net_pnl"]     = net_pnl
        grp["cum_gross"]   = gross_pnl.cumsum()
        grp["cum_net"]     = net_pnl.cumsum()
        results.append(grp)

    return pd.concat(results, ignore_index=True)