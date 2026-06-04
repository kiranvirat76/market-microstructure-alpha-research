"""
capacity.py
Capacity analysis: net Sharpe and PnL per unit as position limit scales.

Answers: at what size does market impact destroy the alpha?
Very few students do this — immediately signals institutional awareness.

Two sweeps:
  1. capacity_sweep  : vary position limit, fixed impact_coeff
  2. impact_sweep    : vary impact_coeff at fixed position limit
"""

import numpy as np
import pandas as pd
from backtest import run_backtest
from risk     import compute_risk_report


def capacity_sweep(
    signal_df: pd.DataFrame,
    limits=None,
    impact_coeff: float = 0.05,
) -> pd.DataFrame:
    """
    Run backtest at each position limit.
    pnl_per_lot measures capital efficiency (degrades at scale).
    """
    if limits is None:
        limits = [10, 25, 50, 100, 250, 500, 1000]

    rows = []
    for lim in limits:
        pnl  = run_backtest(signal_df, position_limit=lim, impact_coeff=impact_coeff)
        risk = compute_risk_report(pnl)
        net  = risk["net_pnl"].sum()
        rows.append({
            "position_limit":  lim,
            "gross_sharpe":    round(float(risk["gross_sharpe"].mean()), 5),
            "net_sharpe":      round(float(risk["net_sharpe"].mean()),   5),
            "net_pnl":         round(float(net),                         4),
            "total_cost":      round(float(risk["total_cost"].sum()),     4),
            "cost_drag_pct":   round(float(risk["cost_drag_pct"].mean()), 2),
            "pnl_per_lot":     round(float(net / lim),                   5),
        })
    df = pd.DataFrame(rows)
    # Find the capacity limit: first size where net Sharpe drops below half of min-size Sharpe
    base_sr = df["net_sharpe"].iloc[0]
    threshold = base_sr * 0.5
    breached = df[df["net_sharpe"] < threshold]
    if not breached.empty:
        cap_limit = int(breached.iloc[0]["position_limit"])
    else:
        cap_limit = int(df["position_limit"].iloc[-1])

    print(f"  [capacity_sweep] Base net Sharpe @ limit={df['position_limit'].iloc[0]}: {base_sr:.4f}")
    print(f"  [capacity_sweep] Alpha half-life: limit={cap_limit} "
          f"(Sharpe drops below {threshold:.4f})")
    return df


def impact_sweep(
    signal_df: pd.DataFrame,
    position_limit: int = 50,
    coeffs=None,
) -> pd.DataFrame:
    """
    Vary the market impact coefficient at fixed position size.
    Shows how sensitive the strategy is to price impact assumptions.
    """
    if coeffs is None:
        coeffs = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]

    rows = []
    for coeff in coeffs:
        pnl  = run_backtest(signal_df, position_limit=position_limit, impact_coeff=coeff)
        risk = compute_risk_report(pnl)
        rows.append({
            "impact_coeff":   coeff,
            "net_sharpe":     round(float(risk["net_sharpe"].mean()),   5),
            "net_pnl":        round(float(risk["net_pnl"].sum()),       4),
            "impact_cost":    round(float(pnl["impact_cost"].sum()),    4),
            "spread_cost":    round(float(pnl["spread_cost"].sum()),    4),
        })
    return pd.DataFrame(rows)