"""
walk_forward.py
Out-of-sample validation via expanding-window walk-forward.

With 3 days of data:
  Fold 1 : train=[0],   test=[1]
  Fold 2 : train=[0,1], test=[2]

Key output: IS Sharpe vs OOS Sharpe.
Large gap → overfitted.  Small gap → robust.

Note: signals are computed per (product, file_day) so no day-level
look-ahead leaks across fold boundaries.
"""

import pandas as pd
from backtest import run_backtest
from risk     import compute_risk_report


def walk_forward(
    signal_df: pd.DataFrame,
    test_window: int   = 1,
    position_limit: int = 50,
) -> dict:
    """
    Expanding window walk-forward backtest.

    Returns
    -------
    folds_df    : per-fold OOS Sharpe / PnL
    oos_pnl_df  : concatenated OOS PnL rows (for drawdown, etc.)
    is_risk_df  : in-sample risk  (full data, for comparison)
    oos_risk_df : out-of-sample risk summary
    """
    days     = sorted(signal_df["file_day"].unique())
    n_days   = len(days)

    fold_rows  = []
    oos_frames = []

    for test_start in range(1, n_days - test_window + 1):
        test_days  = days[test_start : test_start + test_window]
        train_days = days[:test_start]

        test_sig = signal_df[signal_df["file_day"].isin(test_days)].copy()
        if test_sig.empty:
            continue

        oos_pnl  = run_backtest(test_sig, position_limit=position_limit)
        oos_risk = compute_risk_report(oos_pnl)

        oos_pnl["fold"] = test_start
        oos_frames.append(oos_pnl)

        fold_rows.append({
            "fold":           test_start,
            "train_days":     str(train_days),
            "test_days":      str(test_days),
            "oos_gross_sr":   round(float(oos_risk["gross_sharpe"].mean()), 5),
            "oos_net_sr":     round(float(oos_risk["net_sharpe"].mean()),   5),
            "oos_net_pnl":    round(float(oos_risk["net_pnl"].sum()),       4),
            "oos_hit_rate":   round(float(oos_risk["hit_rate"].mean()),     4),
        })

    # In-sample baseline (all days)
    is_pnl  = run_backtest(signal_df, position_limit=position_limit)
    is_risk = compute_risk_report(is_pnl)

    oos_all      = pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
    oos_risk_all = compute_risk_report(oos_all) if not oos_all.empty else pd.DataFrame()

    return {
        "folds_df":    pd.DataFrame(fold_rows),
        "oos_pnl_df":  oos_all,
        "is_risk_df":  is_risk,
        "oos_risk_df": oos_risk_all,
    }