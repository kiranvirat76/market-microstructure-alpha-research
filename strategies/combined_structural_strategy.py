from strategies.pepper_trend_strategy import build_pepper_trend_signal
from strategies.ash_mean_reversion_strategy import build_ash_mean_reversion_signal

import pandas as pd

def build_structural_strategy(ob):

    pepper = build_pepper_trend_signal(ob)

    ash = build_ash_mean_reversion_signal(ob)

    return pd.concat(
        [pepper, ash],
        ignore_index=True
    )
