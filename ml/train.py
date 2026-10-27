"""Train and evaluate: anomaly detection (Isolation Forest) + RUL (Gradient Boosting).

Run:  python -m ml.train
Writes models/ and metrics.json.

The train/test split is by asset, so no hour of a test asset is ever seen
during training.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest

from ml.features import FEATURES, build_features
from ml.simulate import simulate_fleet

RUL_CAP = 300.0          # hours; beyond this, "far from failure"
PERSISTENCE = 6          # consecutive anomalous hours before an alert
TARGET_QUANTILE = 0.995  # score threshold from healthy training data


def split_assets(ids: np.ndarray, n_test: int, seed: int) -> tuple[set, set]:
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(ids))
    rng.shuffle(ids)
    return set(ids[n_test:]), set(ids[:n_test])


def persistent_alerts(flags: np.ndarray, k: int) -> np.ndarray:
    """True from the hour where k consecutive flags are reached."""
    out = np.zeros_like(flags, dtype=bool)
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        out[i] = run >= k
    return out


def evaluate(test: pd.DataFrame, iso: IsolationForest, threshold: float,
             gbr: GradientBoostingRegressor) -> dict:
    detected, leads, healthy_hours, false_alarms = 0, [], 0, 0
    rul_err = []
    for _, g in test.groupby("asset_id"):
        g = g.sort_values("hour")
        score = -iso.score_samples(g[FEATURES])
        alert = persistent_alerts(score > threshold, PERSISTENCE)

        healthy = ~g["degrading"].to_numpy()
        healthy_hours += int(healthy.sum())
        # false alarm = start of an alert episode while still healthy
        starts = alert & ~np.r_[False, alert[:-1]]
        false_alarms += int((starts & healthy).sum())

        degr_alert = alert & ~healthy
        if degr_alert.any():
            detected += 1
            first = int(np.argmax(degr_alert))
            leads.append(float(g["rul"].iloc[first]))

        last = g[g["rul"] <= 100]
        pred = np.clip(gbr.predict(last[FEATURES]), 0, RUL_CAP)
        rul_err.extend(np.abs(pred - last["rul"].clip(upper=RUL_CAP)))

    n = test["asset_id"].nunique()
    return {
        "test_assets": n,
        "detected_before_failure": detected,
        "median_lead_time_h": round(float(np.median(leads)), 1) if leads else None,
        "rul_mae_last_100h": round(float(np.mean(rul_err)), 1),
        "false_alarms_per_1000_healthy_h": round(1000 * false_alarms / max(healthy_hours, 1), 2),
    }


def main(n_assets: int = 90, n_test: int = 18, seed: int = 42, out: str = ".") -> dict:
    raw = simulate_fleet(n_assets=n_assets, seed=seed)
    feats = build_features(raw)
    train_ids, test_ids = split_assets(feats["asset_id"].unique(), n_test, seed)
    train = feats[feats["asset_id"].isin(train_ids)]
    test = feats[feats["asset_id"].isin(test_ids)]

    # Anomaly model learns only what "healthy" looks like.
    healthy_train = train[~train["degrading"]]
    iso = IsolationForest(n_estimators=200, random_state=seed)
    iso.fit(healthy_train[FEATURES])
    threshold = float(np.quantile(-iso.score_samples(healthy_train[FEATURES]), TARGET_QUANTILE))

    gbr = GradientBoostingRegressor(random_state=seed)
    gbr.fit(train[FEATURES], train["rul"].clip(upper=RUL_CAP))

    metrics = evaluate(test, iso, threshold, gbr)
    metrics.update({"train_assets": len(train_ids), "seed": seed})

    out_dir = Path(out)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump({"iso": iso, "threshold": threshold}, out_dir / "models" / "anomaly.joblib")
    joblib.dump(gbr, out_dir / "models" / "rul.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
