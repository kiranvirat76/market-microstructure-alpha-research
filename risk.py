"""
risk.py
Risk and performance metrics computed from the backtest PnL DataFrame.
No cross-module imports — pure functions on numpy arrays / DataFrames.
"""

import numpy as np
import pandas as pd


# ── Primitives ────────────────────────────────────────────────────────────────
def _sharpe(pnl: np.ndarray) -> float:
    std = pnl.std()
    return float(pnl.mean() / std) if std > 0 else 0.0


def _max_drawdown(cum_pnl: np.ndarray) -> float:
    """Peak-to-trough drawdown on cumulative PnL."""
    peak = np.maximum.accumulate(cum_pnl)
    return float((cum_pnl - peak).min())


def _hit_rate(pnl: np.ndarray) -> float:
    """Fraction of non-zero periods that were profitable."""
    nz = pnl[pnl != 0]
    return float((nz > 0).mean()) if len(nz) else np.nan


def _profit_factor(pnl: np.ndarray) -> float:
    """Sum of wins / |sum of losses|.  > 1 means profitable in aggregate."""
    wins   = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    return float(wins / losses) if losses > 0 else np.nan


# ── Main report ───────────────────────────────────────────────────────────────
def compute_risk_report(pnl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-product summary.

    Columns
    -------
    product         symbol
    gross_sharpe    Sharpe on raw mid-price PnL
    net_sharpe      Sharpe after transaction costs  ← key interview metric
    gross_pnl       cumulative gross PnL
    net_pnl         cumulative net PnL
    total_cost      sum of all transaction costs
    cost_drag_pct   cost as % of gross PnL
    max_dd          maximum drawdown on cum_net
    hit_rate        % of active periods that were net-profitable
    profit_factor   sum_wins / |sum_losses|
    turnover        mean |Δposition| per snapshot
    n_trades        number of position changes
    """
    rows = []
    for product, grp in pnl_df.groupby("product"):
        g   = grp["gross_pnl"].values
        n   = grp["net_pnl"].values
        gsum = g.sum()

        rows.append({
            "product":       product,
            "gross_sharpe":  _sharpe(g),
            "net_sharpe":    _sharpe(n),
            "gross_pnl":     float(gsum),
            "net_pnl":       float(n.sum()),
            "total_cost":    float(grp["trans_cost"].sum()),
            "cost_drag_pct": float(grp["trans_cost"].sum() / gsum * 100)
                             if gsum != 0 else np.nan,
            "max_dd":        _max_drawdown(grp["cum_net"].values),
            "hit_rate":      _hit_rate(n),
            "profit_factor": _profit_factor(n),
            "turnover":      float(grp["delta_pos"].abs().mean()),
            "n_trades":      int((grp["delta_pos"] != 0).sum()),
        })

    return pd.DataFrame(rows)