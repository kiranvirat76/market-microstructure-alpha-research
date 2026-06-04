"""
attribution.py
Alpha attribution: how much does each signal module contribute?

Runs 7 configs (solo, pairs, full) and computes:
  - Net Sharpe per config
  - Marginal Sharpe contribution of each module
  - Diversification benefit from combining signals

Sharpe progression A → B → C is the number recruiters love to see.
"""

import pandas as pd
from signals  import build_combined_signal
from backtest import run_backtest
from risk     import compute_risk_report


_CONFIGS = {
    "1  OBI only":        dict(w_obi=1.0, w_liq=0.0, w_tf=0.0),
    "2  Liq only":        dict(w_obi=0.0, w_liq=1.0, w_tf=0.0),
    "3  TF only":         dict(w_obi=0.0, w_liq=0.0, w_tf=1.0),
    "4  OBI + Liq":       dict(w_obi=0.6, w_liq=0.4, w_tf=0.0),
    "5  OBI + TF":        dict(w_obi=0.6, w_liq=0.0, w_tf=0.4),
    "6  Liq + TF":        dict(w_obi=0.0, w_liq=0.5, w_tf=0.5),
    "7  OBI + Liq + TF":  dict(w_obi=0.4, w_liq=0.3, w_tf=0.3),
}


def full_attribution(
    ob: pd.DataFrame,
    merged_trades: pd.DataFrame,
    position_limit: int = 50,
) -> pd.DataFrame:
    """
    Run all 7 signal configs through the full backtest + risk pipeline.
    Returns one row per config with gross/net Sharpe, PnL, hit rate.
    """
    rows = []
    for label, weights in _CONFIGS.items():
        sig  = build_combined_signal(ob, merged_trades, **weights)
        pnl  = run_backtest(sig, position_limit=position_limit)
        risk = compute_risk_report(pnl)
        rows.append({
            "config":        label,
            "w_obi":         weights["w_obi"],
            "w_liq":         weights["w_liq"],
            "w_tf":          weights["w_tf"],
            "gross_sharpe":  round(float(risk["gross_sharpe"].mean()), 5),
            "net_sharpe":    round(float(risk["net_sharpe"].mean()),   5),
            "net_pnl":       round(float(risk["net_pnl"].sum()),       4),
            "hit_rate":      round(float(risk["hit_rate"].mean()),     4),
        })
    return pd.DataFrame(rows)


def marginal_contribution(attribution_df: pd.DataFrame) -> pd.DataFrame:
    """
    Marginal Sharpe contribution of each module (Shapley-style):

      MC_OBI = Sharpe(OBI+Liq+TF) − Sharpe(Liq+TF)
      MC_Liq = Sharpe(OBI+Liq+TF) − Sharpe(OBI+TF)
      MC_TF  = Sharpe(OBI+Liq+TF) − Sharpe(OBI+Liq)

    share_pct = marginal / sum(marginals) × 100
    """
    def _sr(label):
        return float(
            attribution_df.loc[attribution_df["config"] == label, "net_sharpe"].iloc[0]
        )

    full    = _sr("7  OBI + Liq + TF")
    liq_tf  = _sr("6  Liq + TF")
    obi_tf  = _sr("5  OBI + TF")
    obi_liq = _sr("4  OBI + Liq")

    mc = pd.DataFrame([
        {"module": "OBI",        "marginal_sharpe": full - liq_tf},
        {"module": "Liq Shock",  "marginal_sharpe": full - obi_tf},
        {"module": "Trade Flow", "marginal_sharpe": full - obi_liq},
    ])

    total = mc["marginal_sharpe"].sum()
    mc["share_pct"] = (mc["marginal_sharpe"] / total * 100).round(1) if total != 0 else 0.0
    return mc


def sharpe_progression(attribution_df: pd.DataFrame) -> pd.DataFrame:
    """
    The A → B → C table.
    Shows cumulative Sharpe gain as signals are added one-by-one.
    """
    steps = [
        ("A  OBI only",       "1  OBI only"),
        ("B  OBI + Liq",      "4  OBI + Liq"),
        ("C  OBI + Liq + TF", "7  OBI + Liq + TF"),
    ]
    rows = []
    prev_sr = 0.0
    for label, config_key in steps:
        sr = float(
            attribution_df.loc[attribution_df["config"] == config_key, "net_sharpe"].iloc[0]
        )
        rows.append({
            "step":        label,
            "net_sharpe":  round(sr, 5),
            "delta":       round(sr - prev_sr, 5),
        })
        prev_sr = sr
    return pd.DataFrame(rows)
