"""The agent's four tools.

read_sensors runs the real ML pipeline (features, Isolation Forest, RUL model)
on the asset's readings. History and manual are small synthetic stand-ins for
a CMMS and an equipment manual.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest

from ml.features import FEATURES, asset_features
from ml.simulate import ALERT_VIBRATION, TRIP_VIBRATION
from ml.train import PERSISTENCE, RUL_CAP, persistent_alerts

MANUAL = {
    "7.1": ("Alarm and trip limits",
            f"Vibration alarm at {ALERT_VIBRATION} mm/s RMS, trip at {TRIP_VIBRATION} mm/s RMS. "
            "Above alarm: plan intervention. At trip: stop the machine."),
    "7.2": ("Bearing wear",
            "Sustained vibration rise above 3 sigma together with rising bearing temperature "
            "indicates bearing wear. Inspect the bearing within 72 h and take an oil sample."),
    "7.4": ("Misalignment",
            "Vibration rise without a matching bearing temperature rise may indicate "
            "misalignment. Confirm with a frequency spectrum (1x/2x running speed)."),
}


@dataclass
class Fleet:
    raw: pd.DataFrame
    iso: IsolationForest
    threshold: float
    gbr: GradientBoostingRegressor
    work_orders: list = field(default_factory=list)


def read_sensors(fleet: Fleet, asset_id: str, at_hour: int) -> dict:
    g = fleet.raw[(fleet.raw["asset_id"] == asset_id) & (fleet.raw["hour"] <= at_hour)]
    f = asset_features(g)
    score = -fleet.iso.score_samples(f[FEATURES])
    alert = persistent_alerts(score > fleet.threshold, PERSISTENCE)
    vz = f["vibration_z"].to_numpy()
    above = 0
    for z in vz[::-1]:
        if z <= 3:
            break
        above += 1
    base = g.iloc[:150]
    last = f.iloc[-1]
    vib_now = float(g["vibration"].tail(6).mean())
    vib_base = float(base["vibration"].mean())
    t_now = float(g["temperature"].tail(6).mean())
    t_base = float(base["temperature"].mean())
    rul = float(np.clip(fleet.gbr.predict(f[FEATURES].tail(1))[0], 0, RUL_CAP))
    return {
        "asset_id": asset_id,
        "hour": int(at_hour),
        "vibration_mm_s": round(vib_now, 2),
        "vibration_baseline_mm_s": round(vib_base, 2),
        "vibration_change_pct": round(100 * (vib_now / vib_base - 1), 1),
        "hours_above_3sigma": int(above),
        "bearing_temp_c": round(t_now, 1),
        "bearing_temp_baseline_c": round(t_base, 1),
        "bearing_temp_change_c": round(t_now - t_base, 1),
        "anomaly_score": round(float(score[-1]), 3),
        "anomaly_threshold": round(fleet.threshold, 3),
        "alert_active": bool(alert[-1]),
        "rul_estimate_h": round(rul, 0),
        "rul_capped": rul >= RUL_CAP - 1,
        "vibration_z": round(float(last["vibration_z"]), 1),
    }


def get_history(asset_id: str) -> dict:
    """Deterministic synthetic maintenance history for an asset."""
    h = int(hashlib.sha256(asset_id.encode()).hexdigest(), 16)
    return {
        "asset_id": asset_id,
        "last_bearing_change_months": 6 + h % 18,
        "last_oil_analysis_months": 1 + (h // 7) % 8,
        "last_oil_analysis_result": "normal",
        "last_alignment_months": 2 + (h // 11) % 10,
        "failures_last_12_months": 0,
    }


def search_manual(query: str) -> list[dict]:
    q = query.lower()
    keys = {
        "bearing": ["7.2"], "rolamento": ["7.2"],
        "misalignment": ["7.4"], "desalinhamento": ["7.4"],
        "limit": ["7.1"], "alarm": ["7.1"], "trip": ["7.1"],
    }
    hits = sorted({s for k, secs in keys.items() if k in q for s in secs}) or ["7.1"]
    return [{"section": s, "title": MANUAL[s][0], "text": MANUAL[s][1]} for s in hits]


def create_work_order(fleet: Fleet, asset_id: str, priority: str, actions: str, evidence: str) -> dict:
    wo = {
        "id": f"WO-{1000 + len(fleet.work_orders) + 1}",
        "asset_id": asset_id,
        "priority": priority,
        "actions": actions,
        "evidence": evidence,
    }
    fleet.work_orders.append(wo)
    return wo


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "read_sensors",
        "description": "Current vibration and bearing temperature vs. baseline, anomaly score, alert flag and remaining useful life from the ML pipeline.",
        "parameters": {"type": "object", "properties": {"asset_id": {"type": "string"}}, "required": ["asset_id"]}}},
    {"type": "function", "function": {
        "name": "get_history",
        "description": "Maintenance history: last bearing change, oil analysis, alignment, failures.",
        "parameters": {"type": "object", "properties": {"asset_id": {"type": "string"}}, "required": ["asset_id"]}}},
    {"type": "function", "function": {
        "name": "search_manual",
        "description": "Search the equipment manual (limits, bearing wear, misalignment).",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "create_work_order",
        "description": "Open a work order. Only when the evidence justifies it.",
        "parameters": {"type": "object", "properties": {
            "asset_id": {"type": "string"},
            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
            "actions": {"type": "string"}, "evidence": {"type": "string"}},
            "required": ["asset_id", "priority", "actions", "evidence"]}}},
]
