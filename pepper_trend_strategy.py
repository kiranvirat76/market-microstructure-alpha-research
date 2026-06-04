import pandas as pd
import numpy as np

def build_pepper_trend_signal(ob):

    pepper = ob[
        ob["product"] == "INTARIAN_PEPPER_ROOT"
    ].copy()

    pepper["ma50"] = pepper["mid_price"].rolling(50).mean()
    pepper["ma200"] = pepper["mid_price"].rolling(200).mean()

    pepper["score"] = np.where(
        pepper["ma50"] > pepper["ma200"],
        1,
        -1
    )
    return pepper