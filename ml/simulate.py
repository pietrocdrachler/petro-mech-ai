"""Synthetic run-to-failure data for pumps and compressors.

Each asset runs healthy for a random period, then develops bearing wear:
vibration and bearing temperature drift upward (roughly exponential) until
vibration reaches the trip limit, which we treat as the failure.

All numbers are synthetic. The point is to exercise the pipeline end to end,
not to claim a validated model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRIP_VIBRATION = 7.0  # mm/s, treated as failure
ALERT_VIBRATION = 4.5  # mm/s, manual alarm limit (used only for reference)


def simulate_asset(asset_id: str, rng: np.random.Generator) -> pd.DataFrame:
    """Hourly readings for one asset until failure."""
    healthy_hours = int(rng.integers(300, 900))
    base_vib = rng.uniform(1.4, 2.4)       # mm/s RMS
    base_temp = rng.uniform(55.0, 68.0)    # deg C, bearing
    base_pres = rng.uniform(18.0, 24.0)    # bar, discharge
    base_curr = rng.uniform(80.0, 120.0)   # A, motor
    growth = rng.uniform(0.008, 0.02)      # degradation rate per hour

    vib, temp, pres, curr, degrading = [], [], [], [], []
    t = 0
    while True:
        d = 0.0 if t < healthy_hours else np.expm1(growth * (t - healthy_hours))
        v = base_vib * (1 + 0.35 * d) + rng.normal(0, 0.12 * base_vib * 0.5)
        vib.append(max(v, 0.1))
        temp.append(base_temp + 3.0 * d + rng.normal(0, 0.6))
        pres.append(base_pres - 0.15 * d + rng.normal(0, 0.3))
        curr.append(base_curr * (1 + 0.02 * d) + rng.normal(0, 1.0))
        degrading.append(t >= healthy_hours)
        t += 1
        if v >= TRIP_VIBRATION or t > 5000:
            break

    n = len(vib)
    return pd.DataFrame({
        "asset_id": asset_id,
        "hour": np.arange(n),
        "vibration": vib,
        "temperature": temp,
        "pressure": pres,
        "current": curr,
        "degrading": degrading,
        "rul": np.arange(n)[::-1].astype(float),  # hours until failure
    })


def simulate_fleet(n_assets: int = 90, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [simulate_asset(f"A{i:03d}", rng) for i in range(n_assets)]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = simulate_fleet()
    print(df.groupby("asset_id").size().describe())
