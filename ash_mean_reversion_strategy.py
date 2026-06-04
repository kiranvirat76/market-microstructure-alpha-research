import pandas as pd
import numpy as np

def build_ash_mean_reversion_signal(
        ob,
        threshold=3
):

    ash = ob[
        ob["product"] == "ASH_COATED_OSMIUM"
    ].copy()

    # Dynamic fair value
    ash["fair_value"] = (
        ash["mid_price"]
        .rolling(500, min_periods=50)
        .mean()
    )

    deviation = (
        ash["mid_price"]
        - ash["fair_value"]
    )

    ash["score"] = 0

    ash.loc[
        deviation > threshold,
        "score"
    ] = -1

    ash.loc[
        deviation < -threshold,
        "score"
    ] = 1

    ash["score"] = (
        ash["score"]
        .fillna(0)
    )

    return ash