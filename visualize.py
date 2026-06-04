"""
visualize.py
All plots for the microstructure analysis.
Saves figures to reports/plots/.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "reports" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────
COLORS = {
    "ASH_COATED_OSMIUM":    "#2196F3",   # blue
    "INTARIAN_PEPPER_ROOT": "#FF6B35",   # orange
}
LIGHT_GRAY = "#F4F4F4"
DARK       = "#1A1A2E"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    LIGHT_GRAY,
    "axes.edgecolor":    "#CCCCCC",
    "axes.labelcolor":   DARK,
    "xtick.color":       DARK,
    "ytick.color":       DARK,
    "font.family":       "monospace",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "legend.fontsize":   8,
})


def _save(name: str):
    path = PLOT_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────
# 1. Mid-price + spread over time
# ─────────────────────────────────────────────────────────────────────

def plot_midprice_and_spread(ob: pd.DataFrame):
    products = ob["product"].unique()
    fig, axes = plt.subplots(len(products), 2, figsize=(14, 5 * len(products)))
    fig.suptitle("Mid-Price & Bid-Ask Spread  (3 Days)", fontsize=13, fontweight="bold")

    for row, product in enumerate(products):
        color = COLORS.get(product, "#333")
        sub = ob[ob["product"] == product].sort_values(["file_day", "timestamp"])

        ax_price  = axes[row, 0] if len(products) > 1 else axes[0]
        ax_spread = axes[row, 1] if len(products) > 1 else axes[1]

        # Build a continuous x-axis across days
        sub = sub.copy()
        sub["x"] = sub["file_day"] * 1_000_000 + sub["timestamp"]

        ax_price.plot(sub["x"], sub["mid_price"], color=color, lw=0.6, alpha=0.85)
        ax_price.set_title(f"{product}  — Mid Price")
        ax_price.set_ylabel("Price (XIRECS)")
        ax_price.set_xlabel("Time →")
        ax_price.xaxis.set_major_formatter(mticker.NullFormatter())

        # Add day separators
        for d in sub["file_day"].unique()[1:]:
            sep = d * 1_000_000
            ax_price.axvline(sep, color="#888", lw=0.8, ls="--", alpha=0.5)

        ax_spread.plot(sub["x"], sub["spread"], color=color, lw=0.5, alpha=0.7)
        ax_spread.axhline(sub["spread"].mean(), color="red", lw=1, ls="--", label=f"Mean = {sub['spread'].mean():.1f}")
        ax_spread.set_title(f"{product}  — Bid-Ask Spread")
        ax_spread.set_ylabel("Spread (ticks)")
        ax_spread.set_xlabel("Time →")
        ax_spread.xaxis.set_major_formatter(mticker.NullFormatter())
        ax_spread.legend()

        for d in sub["file_day"].unique()[1:]:
            ax_spread.axvline(d * 1_000_000, color="#888", lw=0.8, ls="--", alpha=0.5)

    plt.tight_layout()
    _save("01_midprice_spread")


# ─────────────────────────────────────────────────────────────────────
# 2. Spread Distribution
# ─────────────────────────────────────────────────────────────────────

def plot_spread_distribution(ob: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Spread Distribution by Product", fontsize=13, fontweight="bold")

    for ax, product in zip(axes, ob["product"].unique()):
        color = COLORS.get(product, "#333")
        data = ob[ob["product"] == product]["spread"].dropna()

        ax.hist(data, bins=30, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(data.mean(),   color="red",    lw=1.5, ls="--", label=f"Mean={data.mean():.1f}")
        ax.axvline(data.median(), color="purple", lw=1.5, ls=":",  label=f"Median={data.median():.1f}")
        ax.set_title(product)
        ax.set_xlabel("Spread (ticks)")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout()
    _save("02_spread_distribution")


# ─────────────────────────────────────────────────────────────────────
# 3. Intraday Spread Pattern (U-shape test)
# ─────────────────────────────────────────────────────────────────────

def plot_intraday_spread(pattern_df: pd.DataFrame):
    products = pattern_df["product"].unique()
    fig, axes = plt.subplots(1, len(products), figsize=(13, 4), sharey=False)
    fig.suptitle("Intraday Spread Pattern — Testing for U-Shape", fontsize=13, fontweight="bold")

    for ax, product in zip(np.atleast_1d(axes), products):
        color = COLORS.get(product, "#333")
        sub = pattern_df[pattern_df["product"] == product]
        ax.plot(sub["time_bucket"], sub["mean_spread"], marker="o", ms=4,
                color=color, lw=2, label="Mean Spread")
        ax.fill_between(sub["time_bucket"], sub["mean_spread"],
                        alpha=0.15, color=color)
        ax.set_title(product)
        ax.set_xlabel("Intraday Time Bucket (early → late)")
        ax.set_ylabel("Mean Spread (ticks)")
        ax.legend()

    plt.tight_layout()
    _save("03_intraday_spread_pattern")


# ─────────────────────────────────────────────────────────────────────
# 4. Order Book Imbalance
# ─────────────────────────────────────────────────────────────────────

def plot_obi(ob: pd.DataFrame):
    products = ob["product"].unique()
    fig, axes = plt.subplots(len(products), 1, figsize=(13, 5 * len(products)))
    fig.suptitle("Order Book Imbalance (OBI)\n"
                 "OBI > 0 → buy pressure | OBI < 0 → sell pressure",
                 fontsize=13, fontweight="bold")

    for ax, product in zip(np.atleast_1d(axes), products):
        color = COLORS.get(product, "#333")
        sub = ob[ob["product"] == product].sort_values(["file_day", "timestamp"]).copy()
        sub["x"] = sub["file_day"] * 1_000_000 + sub["timestamp"]

        # Rolling OBI for readability
        sub["obi_roll"] = sub["obi"].rolling(200, min_periods=20).mean()

        ax.fill_between(sub["x"], sub["obi_roll"],
                        where=sub["obi_roll"] >= 0, color="#4CAF50", alpha=0.6, label="Buy pressure")
        ax.fill_between(sub["x"], sub["obi_roll"],
                        where=sub["obi_roll"] < 0,  color="#F44336", alpha=0.6, label="Sell pressure")
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_title(product)
        ax.set_ylabel("OBI (rolling 200)")
        ax.set_xlabel("Time →")
        ax.xaxis.set_major_formatter(mticker.NullFormatter())
        ax.legend(loc="upper right")

    plt.tight_layout()
    _save("04_order_book_imbalance")


# ─────────────────────────────────────────────────────────────────────
# 5. OBI as Price Predictor
# ─────────────────────────────────────────────────────────────────────

def plot_obi_predictive(ob: pd.DataFrame, lookahead: int = 50):
    """
    Bin OBI into 5 quintiles; show forward mid-price return per quintile.
    Tests whether OBI predicts short-term price direction.
    """
    results = []
    for product, grp in ob.groupby("product"):
        grp = grp.sort_values(["file_day", "timestamp"]).copy()
        grp["fwd_return"] = grp["mid_price"].pct_change(lookahead).shift(-lookahead)
        grp["obi_bin"] = pd.qcut(grp["obi"], q=5, labels=False, duplicates="drop")
        avg = grp.groupby("obi_bin")["fwd_return"].mean()
        for bin_id, ret in avg.items():
            results.append({"product": product, "obi_bin": bin_id, "fwd_return": ret})

    df = pd.DataFrame(results)

    products = df["product"].unique()
    fig, axes = plt.subplots(1, len(products), figsize=(12, 4))
    fig.suptitle(f"OBI → Forward Return (lookahead={lookahead} ticks)\n"
                 "Monotonic ↑ = OBI has predictive power",
                 fontsize=13, fontweight="bold")

    for ax, product in zip(np.atleast_1d(axes), products):
        color = COLORS.get(product, "#333")
        sub = df[df["product"] == product].sort_values("obi_bin")
        bars = ax.bar(sub["obi_bin"], sub["fwd_return"] * 10_000,
                      color=[("#4CAF50" if v >= 0 else "#F44336") for v in sub["fwd_return"]])
        ax.axhline(0, color="#333", lw=1)
        ax.set_title(product)
        ax.set_xlabel("OBI Quintile (low → high)")
        ax.set_ylabel("Mean Fwd Return (bps)")
        ax.set_xticks(sub["obi_bin"])
        ax.set_xticklabels(["Q1\n(sell)", "Q2", "Q3\n(neutral)", "Q4", "Q5\n(buy)"])

    plt.tight_layout()
    _save("05_obi_predictive")


# ─────────────────────────────────────────────────────────────────────
# 6. Kyle's Lambda — Price Impact
# ─────────────────────────────────────────────────────────────────────

def plot_kyles_lambda(merged_trades: pd.DataFrame, lambda_results: dict):
    """Scatter: signed volume vs mid-price change + regression line."""
    products = list(lambda_results.keys())
    fig, axes = plt.subplots(1, len(products), figsize=(13, 5))
    fig.suptitle("Kyle's Lambda — Price Impact of Order Flow\n"
                 "Slope (λ) = price change per unit signed volume",
                 fontsize=13, fontweight="bold")

    for ax, product in zip(np.atleast_1d(axes), products):
        color  = COLORS.get(product, "#333")
        res    = lambda_results[product]
        sub    = merged_trades[merged_trades["symbol"] == product].copy()
        sub["delta_mid"]  = sub["mid_price"].diff()
        sub["signed_qty"] = sub["quantity"] * sub["direction"]
        sub = sub.dropna(subset=["delta_mid", "signed_qty"])

        ax.scatter(sub["signed_qty"], sub["delta_mid"],
                   alpha=0.35, s=15, color=color, label="Trades")

        xline = np.linspace(sub["signed_qty"].min(), sub["signed_qty"].max(), 100)
        yline = res["lambda"] * xline + res["intercept"]
        ax.plot(xline, yline, color="red", lw=2,
                label=f"λ = {res['lambda']:.4f}\nR² = {res['r_squared']:.3f}\np = {res['p_value']:.4f}")

        ax.axhline(0, color="#888", lw=0.7)
        ax.axvline(0, color="#888", lw=0.7)
        ax.set_title(product)
        ax.set_xlabel("Signed Volume (+ = buy, − = sell)")
        ax.set_ylabel("Mid-Price Change (ΔP)")
        ax.legend()

    plt.tight_layout()
    _save("06_kyles_lambda")


# ─────────────────────────────────────────────────────────────────────
# 7. Trade Arrival Rate
# ─────────────────────────────────────────────────────────────────────

def plot_trade_arrival(trades: pd.DataFrame):
    """Distribution of inter-trade durations per product."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Inter-Trade Duration Distribution\n"
                 "Thin right tail → clustered activity (Hawkes-like)",
                 fontsize=13, fontweight="bold")

    for ax, product in zip(axes, trades["symbol"].unique()):
        color = COLORS.get(product, "#333")
        sub = trades[trades["symbol"] == product].sort_values(["file_day", "timestamp"])
        sub["idt"] = sub.groupby("file_day")["timestamp"].diff()
        idt = sub["idt"].dropna()

        ax.hist(np.log1p(idt), bins=40, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(product)
        ax.set_xlabel("log(1 + Inter-Trade Duration)")
        ax.set_ylabel("Count")
        mean_idt = idt.mean()
        ax.axvline(np.log1p(mean_idt), color="red", lw=1.5, ls="--",
                   label=f"Mean = {mean_idt:.0f}")
        ax.legend()

    plt.tight_layout()
    _save("07_trade_arrival")


# ─────────────────────────────────────────────────────────────────────
# 8. Amihud Illiquidity
# ─────────────────────────────────────────────────────────────────────

def plot_amihud(ob_amihud: pd.DataFrame):
    products = ob_amihud["product"].unique()
    fig, axes = plt.subplots(len(products), 1, figsize=(13, 4 * len(products)))
    fig.suptitle("Amihud Illiquidity Ratio  (Rolling)\n"
                 "Higher = price moves more per unit of depth",
                 fontsize=13, fontweight="bold")

    for ax, product in zip(np.atleast_1d(axes), products):
        color = COLORS.get(product, "#333")
        sub = ob_amihud[ob_amihud["product"] == product].sort_values(["file_day", "timestamp"]).copy()
        sub["x"] = sub["file_day"] * 1_000_000 + sub["timestamp"]

        ax.plot(sub["x"], sub["amihud_roll"], color=color, lw=1.2, alpha=0.85)
        ax.set_title(product)
        ax.set_ylabel("Amihud ILLIQ (rolling)")
        ax.set_xlabel("Time →")
        ax.xaxis.set_major_formatter(mticker.NullFormatter())

    plt.tight_layout()
    _save("08_amihud_illiquidity")


# ─────────────────────────────────────────────────────────────────────
# 9. Summary Dashboard
# ─────────────────────────────────────────────────────────────────────

def plot_summary_dashboard(ob: pd.DataFrame, merged_trades: pd.DataFrame,
                           lambda_results: dict, rolls_df: pd.DataFrame,
                           vwap_df: pd.DataFrame):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Market Microstructure Dashboard  —  Round 1 (3 Days)",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

    products = ob["product"].unique()
    color_map = COLORS

    # ── Row 0: Mid-price per product ──────────────────────────────────
    for col, product in enumerate(products):
        ax = fig.add_subplot(gs[0, col])
        color = color_map.get(product, "#333")
        sub = ob[ob["product"] == product].sort_values(["file_day", "timestamp"]).copy()
        sub["x"] = sub["file_day"] * 1_000_000 + sub["timestamp"]
        ax.plot(sub["x"], sub["mid_price"], color=color, lw=0.7)
        ax.set_title(f"{product[:10]}…\nMid Price", fontsize=8)
        ax.xaxis.set_major_formatter(mticker.NullFormatter())
        ax.set_ylabel("Price", fontsize=7)

    # ── Spread comparison (right of row 0) ───────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    spread_stats = ob.groupby("product")["spread"].agg(["mean", "std", "median"])
    x_pos = np.arange(len(products))
    ax3.bar(x_pos, spread_stats["mean"], color=[color_map.get(p, "#333") for p in products],
            alpha=0.8, width=0.4)
    ax3.errorbar(x_pos, spread_stats["mean"], yerr=spread_stats["std"],
                 fmt="none", color="black", capsize=5, lw=1.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([p[:10] for p in products], fontsize=7)
    ax3.set_title("Mean Spread ± Std", fontsize=8)
    ax3.set_ylabel("Ticks", fontsize=7)

    # ── Row 1: OBI rolling per product ───────────────────────────────
    for col, product in enumerate(products):
        ax = fig.add_subplot(gs[1, col])
        color = color_map.get(product, "#333")
        sub = ob[ob["product"] == product].sort_values(["file_day", "timestamp"]).copy()
        sub["x"] = sub["file_day"] * 1_000_000 + sub["timestamp"]
        sub["obi_roll"] = sub["obi"].rolling(500, min_periods=20).mean()
        ax.fill_between(sub["x"], sub["obi_roll"],
                        where=sub["obi_roll"] >= 0, color="#4CAF50", alpha=0.6)
        ax.fill_between(sub["x"], sub["obi_roll"],
                        where=sub["obi_roll"] < 0,  color="#F44336", alpha=0.6)
        ax.axhline(0, color="#333", lw=0.7)
        ax.set_title(f"{product[:10]}…\nOBI (rolling)", fontsize=8)
        ax.xaxis.set_major_formatter(mticker.NullFormatter())

    # ── Kyle's Lambda comparison ──────────────────────────────────────
    ax_lambda = fig.add_subplot(gs[1, 2])
    lam_products = list(lambda_results.keys())
    lam_values   = [lambda_results[p]["lambda"]    for p in lam_products]
    lam_r2       = [lambda_results[p]["r_squared"] for p in lam_products]
    x_pos = np.arange(len(lam_products))
    bars = ax_lambda.bar(x_pos, lam_values,
                         color=[color_map.get(p, "#333") for p in lam_products],
                         alpha=0.8, width=0.4)
    ax_lambda.set_xticks(x_pos)
    ax_lambda.set_xticklabels([p[:10] for p in lam_products], fontsize=7)
    ax_lambda.set_title("Kyle's Lambda (λ)\nPrice Impact per Unit Volume", fontsize=8)
    ax_lambda.set_ylabel("λ", fontsize=7)
    for bar, r2 in zip(bars, lam_r2):
        ax_lambda.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() * 1.02,
                       f"R²={r2:.3f}", ha="center", fontsize=6)

    # ── Row 2: VWAP table and Roll's estimator ────────────────────────
    ax_vwap = fig.add_subplot(gs[2, :2])
    ax_vwap.axis("off")
    vwap_display = vwap_df[["product", "file_day", "vwap", "total_vol", "n_trades"]].copy()
    vwap_display.columns = ["Product", "Day", "VWAP", "Volume", "#Trades"]
    vwap_display["VWAP"] = vwap_display["VWAP"].round(2)
    tbl = ax_vwap.table(cellText=vwap_display.values,
                        colLabels=vwap_display.columns,
                        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    ax_vwap.set_title("VWAP & Volume Summary", fontsize=9, pad=10)

    ax_roll = fig.add_subplot(gs[2, 2])
    rolls_df_clean = rolls_df.dropna(subset=["rolls_spread"])
    for product, grp in rolls_df_clean.groupby("product"):
        color = color_map.get(product, "#333")
        ax_roll.bar(grp["file_day"] + (0.2 if product != list(color_map.keys())[0] else 0),
                    grp["rolls_spread"],
                    width=0.35, color=color, alpha=0.85, label=product[:12])
    ax_roll.set_title("Roll's Spread Estimator\nper Day", fontsize=8)
    ax_roll.set_xlabel("Day", fontsize=7)
    ax_roll.set_ylabel("Estimated Spread", fontsize=7)
    ax_roll.set_xticks([0, 1, 2])
    ax_roll.legend(fontsize=6)

    _save("09_summary_dashboard")