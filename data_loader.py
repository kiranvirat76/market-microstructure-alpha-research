#Loads and cleans order book + trade data for microstructure analysis.

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_orderbook(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = []
    for day in [-2, -1, 0]:
        path = data_dir / f"prices_round_1_day_{day}.csv"
        df = pd.read_csv(path, sep=";")
        df["file_day"] = day
        frames.append(df)
    ob = pd.concat(frames, ignore_index=True)

    # Rename 'day' column (values -2,-1,0) to avoid confusion
    ob.rename(columns={"day": "sim_day"}, inplace=True)

    # Numeric coercion (some bid/ask levels can be empty)
    price_cols = [c for c in ob.columns if "price" in c or "volume" in c]
    for c in price_cols:
        ob[c] = pd.to_numeric(ob[c], errors="coerce")

    # Derived columns
    ob["spread"]          = ob["ask_price_1"] - ob["bid_price_1"]
    ob["relative_spread"] = ob["spread"] / ob["mid_price"]
    ob["bid_depth_1"]     = ob["bid_volume_1"].fillna(0)
    ob["ask_depth_1"]     = ob["ask_volume_1"].fillna(0)

    # Total depth across 3 levels
    for side in ("bid", "ask"):
        vols = [ob[f"{side}_volume_{lvl}"].fillna(0) for lvl in [1, 2, 3]]
        ob[f"total_{side}_depth"] = sum(vols)

    # Order Book Imbalance  OBI ∈ (-1, 1)
    total_bid = ob["total_bid_depth"]
    total_ask = ob["total_ask_depth"]
    denom = total_bid + total_ask
    ob["obi"] = np.where(denom > 0, (total_bid - total_ask) / denom, 0.0)

    return ob


def load_trades(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = []
    for day in [-2, -1, 0]:
        path = data_dir / f"trades_round_1_day_{day}.csv"
        df = pd.read_csv(path, sep=";")
        df["file_day"] = day
        frames.append(df)
    trades = pd.concat(frames, ignore_index=True)
    trades["price"]    = pd.to_numeric(trades["price"],    errors="coerce")
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce")
    return trades


def merge_trades_with_ob(trades: pd.DataFrame, ob: pd.DataFrame) -> pd.DataFrame:

    result_frames = []
    for product in trades["symbol"].unique():
        for day in trades["file_day"].unique():
            t_sub = trades[
                (trades["symbol"] == product) & (trades["file_day"] == day)
            ].copy()
            o_sub = ob[
                (ob["product"] == product) & (ob["file_day"] == day)
            ][["timestamp", "mid_price", "spread", "relative_spread",
               "obi", "total_bid_depth", "total_ask_depth"]].copy()

            if t_sub.empty or o_sub.empty:
                continue

            t_sub.sort_values("timestamp", inplace=True)
            o_sub.sort_values("timestamp", inplace=True)

            merged = pd.merge_asof(
                t_sub, o_sub,
                on="timestamp",
                direction="backward",
            )
            result_frames.append(merged)

    return pd.concat(result_frames, ignore_index=True)


if __name__ == "__main__":
    ob     = load_orderbook()
    trades = load_trades()
    merged = merge_trades_with_ob(trades, ob)
    print("Order book shape:", ob.shape)
    print("Trades shape:    ", trades.shape)
    print("Merged shape:    ", merged.shape)
    print("\nOrder book sample:\n", ob.head(3))
    print("\nMerged sample:\n",     merged.head(3))