import numpy as np

from ml.features import BASELINE_HOURS, FEATURES, build_features
from ml.simulate import TRIP_VIBRATION, simulate_asset, simulate_fleet
from ml.train import main, persistent_alerts, split_assets


def test_asset_runs_to_failure():
    df = simulate_asset("T1", np.random.default_rng(0))
    assert df["vibration"].iloc[-1] >= TRIP_VIBRATION
    assert df["rul"].iloc[-1] == 0
    assert not df["degrading"].iloc[0]


def test_features_have_no_nans_and_skip_baseline():
    raw = simulate_fleet(n_assets=3, seed=1)
    feats = build_features(raw)
    assert not feats[FEATURES].isna().any().any()
    assert feats.groupby("asset_id")["hour"].min().min() == BASELINE_HOURS


def test_split_is_by_asset():
    train, test = split_assets(np.array([f"A{i}" for i in range(20)]), 5, seed=0)
    assert len(test) == 5 and not (train & test)


def test_persistent_alerts_need_consecutive_flags():
    flags = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
    assert persistent_alerts(flags, 3).tolist() == [False] * 5 + [True]


def test_end_to_end_small(tmp_path):
    m = main(n_assets=20, n_test=5, seed=7, out=str(tmp_path))
    assert m["test_assets"] == 5
    assert m["detected_before_failure"] >= 4
    assert (tmp_path / "metrics.json").exists()
