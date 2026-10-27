"""Per-asset features: deviation from the asset's own baseline and trend.

The baseline is the first BASELINE_HOURS of each asset, assumed healthy
(as an engineer would after commissioning or an overhaul).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASELINE_HOURS = 150
SENSORS = ["vibration", "temperature", "pressure", "current"]
FEATURES = (
    [f"{s}_z" for s in SENSORS]
    + [f"{s}_slope24" for s in SENSORS]
    + ["vibration_ratio"]
)


def _slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window) - (window - 1) / 2
    denom = (x ** 2).sum()
    return series.rolling(window, min_periods=window).apply(
        lambda y: float((x * (y - y.mean())).sum() / denom), raw=True
    )


def asset_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("hour").copy()
    base = df.iloc[:BASELINE_HOURS]
    for s in SENSORS:
        mu, sd = base[s].mean(), base[s].std() or 1e-6
        smooth = df[s].rolling(6, min_periods=1).mean()
        df[f"{s}_z"] = (smooth - mu) / sd
        df[f"{s}_slope24"] = _slope(df[s], 24) / sd
    df["vibration_ratio"] = df["vibration"].rolling(6, min_periods=1).mean() / base["vibration"].mean()
    df = df.iloc[BASELINE_HOURS:]  # features only after the baseline window
    return df.fillna(0.0)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [asset_features(g) for _, g in raw.groupby("asset_id")], ignore_index=True
    )
