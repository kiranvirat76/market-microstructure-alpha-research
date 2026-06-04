"""
main.py
Full pipeline: Load → Directions → Signal → Validate → Walk-Forward
               → Backtest (market impact) → Risk → Capacity → Attribution

Steps
-----
1. Load data
2. Lee-Ready trade directions
3. Build combined signal
4. Predictive validation  (IC, quintile, decay)
5. Walk-forward           (IS vs OOS Sharpe)
6. Backtest + risk        (sqrt market impact model)
7. Capacity analysis      (position sweep + impact sweep)
8. Alpha attribution      (7 configs + marginal contribution)
"""

import pandas as pd

from Data_Loader  import load_orderbook, load_trades, merge_trades_with_ob
from Metrics      import lee_ready_direction
from Signals      import build_combined_signal
from backtest     import run_backtest
from risk         import compute_risk_report
from Validation   import information_coefficient, quintile_analysis, signal_decay
from Walk_forward import walk_forward
from capacity     import capacity_sweep, impact_sweep
from attribution import full_attribution, marginal_contribution, sharpe_progression


def _banner(title: str):
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print(f"{'─' * 65}")


def _fmt(df, f="{:.5f}"):
    pd.set_option("display.float_format", f.format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 130)
    return df.to_string(index=False)


def main():
    print("\n=== Market Microstructure Alpha Engine  (Full Pipeline) ===")

    # ── 1. Load ────────────────────────────────────────────────────────────────
    _banner("1 / 8  Loading data")
    ob     = load_orderbook()
    trades = load_trades()
    print(f"  OB rows    : {len(ob):,}")
    print(f"  Trade rows : {len(trades):,}")
    print(f"  Products   : {sorted(ob['product'].unique())}")
    print(f"  Days       : {sorted(ob['file_day'].unique())}")

    # ── 2. Trade directions ────────────────────────────────────────────────────
    _banner("2 / 8  Lee-Ready trade directions")
    merged = merge_trades_with_ob(trades, ob)
    merged["direction"] = lee_ready_direction(merged["price"], merged["mid_price"])
    buy_pct = (merged["direction"] == 1).mean()
    print(f"  Trades enriched : {len(merged):,}")
    print(f"  Buy / Sell      : {buy_pct:.1%} / {1 - buy_pct:.1%}")

    # ── 3. Signal ──────────────────────────────────────────────────────────────
    _banner("3 / 8  Combined signal  (w_obi=0.6, w_liq=0.4, w_tf=0.0)")
    signal_df = build_combined_signal(ob, merged)
    sc = signal_df["score"]
    print(f"  mean={sc.mean():.5f}  std={sc.std():.5f}  "
          f"range=[{sc.min():.4f}, {sc.max():.4f}]")
    print(f"  long  (>+0.10) : {(sc >  0.10).sum():,}")
    print(f"  short (<-0.10) : {(sc < -0.10).sum():,}")

    # ── 4. Predictive validation ───────────────────────────────────────────────
    _banner("4 / 8  Predictive validation")

    ic_df = information_coefficient(signal_df)
    ic_sum = (
        ic_df.groupby(["product", "horizon"])[["ic", "p_value"]]
        .mean()
        .reset_index()
    )
    print("\n  Information Coefficient (Spearman rank):")
    print(_fmt(ic_sum))

    q_df = quintile_analysis(signal_df, horizon=20)
    print("\n  Quintile analysis  (horizon=20 steps):")
    print(_fmt(q_df[["product", "quintile", "mean_fwd_ret", "std_fwd_ret", "n"]]))

    decay_df = signal_decay(signal_df, max_horizon=100, step=10)
    print("\n  Signal decay  (mean IC per horizon):")
    print(_fmt(decay_df))

    # ── 5. Walk-forward ────────────────────────────────────────────────────────
    _banner("5 / 8  Walk-forward  (expanding window, IS vs OOS)")
    wf = walk_forward(signal_df, test_window=1, position_limit=50)

    print("\n  Per-fold OOS results:")
    print(_fmt(wf["folds_df"]))

    if not wf["oos_risk_df"].empty:
        cols = ["product", "gross_sharpe", "net_sharpe", "net_pnl", "hit_rate"]
        print("\n  In-Sample  (all days):")
        print(_fmt(wf["is_risk_df"][cols]))
        print("\n  Out-of-Sample  (OOS folds concatenated):")
        print(_fmt(wf["oos_risk_df"][cols]))

    # ── 6. Backtest + risk ────────────────────────────────────────────────────
    _banner("6 / 8  Backtest + risk  (impact_coeff=0.05)")
    pnl_df  = run_backtest(signal_df, position_limit=50, impact_coeff=0.05)
    risk_df = compute_risk_report(pnl_df)
    # ── Structural Benchmark ─────────────────────────────
    from combined_structural_strategy import build_structural_strategy

    structural_signal = build_structural_strategy(ob)

    structural_pnl = run_backtest(
        structural_signal,
        position_limit=50,
        impact_coeff=0.05
    )

    structural_risk = compute_risk_report(
        structural_pnl
    )

    print("\n  Performance:")
    print(_fmt(risk_df[[
        "product", "gross_sharpe", "net_sharpe",
        "gross_pnl", "net_pnl", "total_cost", "cost_drag_pct",
    ]]))

    print("\n  Execution stats:")
    print(_fmt(risk_df[["product", "max_dd", "hit_rate", "profit_factor", "turnover", "n_trades"]]))

    g = risk_df["gross_pnl"].sum()
    n = risk_df["net_pnl"].sum()
    c = risk_df["total_cost"].sum()
    ic_total = pnl_df["impact_cost"].sum()
    sc_total = pnl_df["spread_cost"].sum()
    print(f"\n  Gross {g:.2f}  |  Net {n:.2f}  |  "
          f"Cost {c:.2f}  (spread {sc_total:.2f}  +  impact {ic_total:.2f})")

    print("\nSTRUCTURAL STRATEGY")
    print(
        _fmt(
            structural_risk[
                [
                    "product",
                    "gross_sharpe",
                    "net_sharpe",
                    "gross_pnl",
                    "net_pnl",
                    "total_cost",
                    "cost_drag_pct",
                ]
            ]
        )
    )



    # ── 7. Capacity ────────────────────────────────────────────────────────────
    _banner("7 / 8  Capacity analysis")
    print("\n  Position sweep  (impact_coeff=0.05):")
    cap_df = capacity_sweep(signal_df, impact_coeff=0.05)
    print(_fmt(cap_df))

    print("\n  Impact coefficient sweep  (position_limit=50):")
    imp_df = impact_sweep(signal_df, position_limit=50)
    print(_fmt(imp_df))

    # ── 8. Attribution ─────────────────────────────────────────────────────────
    _banner("8 / 8  Alpha attribution")
    attr_df = full_attribution(ob, merged, position_limit=50)

    print("\n  All configs:")
    print(_fmt(attr_df[["config", "gross_sharpe", "net_sharpe", "net_pnl", "hit_rate"]]))

    mc_df = marginal_contribution(attr_df)
    print("\n  Marginal contribution per module:")
    print(_fmt(mc_df))

    prog_df = sharpe_progression(attr_df)
    print("\n  Sharpe progression  A → B → C:")
    print(_fmt(prog_df))

    print("\n" + "=" * 65 + "\n")
    return {
        "signal_df": signal_df, "pnl_df": pnl_df,  "risk_df":  risk_df,
        "ic_df":     ic_df,     "decay_df": decay_df, "wf":     wf,
        "cap_df":    cap_df,    "imp_df":  imp_df,  "attr_df":  attr_df,
        "mc_df":     mc_df,     "prog_df": prog_df,
    }


if __name__ == "__main__":
    results = main()
