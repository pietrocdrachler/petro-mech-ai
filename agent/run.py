"""Run the agent on one asset, or on the three bundled examples.

    python -m agent.run --examples          # writes examples/ (input, output, trace)
    python -m agent.run --asset A010 --hour 700

Planner:
  * If METERLESS_BASE_URL is set (e.g. http://127.0.0.1:8088/v1), the agent
    calls Meterless Nightshift through its OpenAI-compatible chat/completions
    endpoint with tool calling, and logs every request and receipt.
  * Otherwise it uses a transparent rule-based planner, so the repository runs
    anywhere with no keys. The output always says which planner was used.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import joblib

from agent import tools
from ml.simulate import ALERT_VIBRATION, TRIP_VIBRATION, simulate_fleet
from ml.train import main as train_main

ROOT = Path(__file__).resolve().parent.parent
LIMITS = [
    "No frequency spectrum, so bearing wear cannot be fully separated from misalignment.",
    "No lubrication or load data.",
    "All readings are synthetic.",
]

# The three bundled examples: test assets the models never saw, at chosen hours.
EXAMPLES = [
    {"name": "late_bearing_wear", "asset_id": None, "hours_before_failure": 40},
    {"name": "early_deviation", "asset_id": None, "hours_after_onset": 30},
    {"name": "healthy", "asset_id": None, "hours_after_baseline": 150},
]


class Trace:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        self.step = 0

    def log(self, kind: str, **data):
        self.step += 1
        rec = {"step": self.step, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "kind": kind, **data}
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_fleet(seed: int = 42) -> tools.Fleet:
    models = ROOT / "models"
    if not (models / "anomaly.joblib").exists():
        train_main(seed=seed, out=str(ROOT))
    anomaly = joblib.load(models / "anomaly.joblib")
    gbr = joblib.load(models / "rul.joblib")
    return tools.Fleet(raw=simulate_fleet(seed=seed), iso=anomaly["iso"],
                       threshold=anomaly["threshold"], gbr=gbr)


def call_tool(fleet, trace, name, args, at_hour):
    t0 = time.perf_counter()
    if name == "read_sensors":
        out = tools.read_sensors(fleet, args["asset_id"], at_hour)
    elif name == "get_history":
        out = tools.get_history(args["asset_id"])
    elif name == "search_manual":
        out = tools.search_manual(args["query"])
    elif name == "create_work_order":
        out = tools.create_work_order(fleet, **args)
    else:
        out = {"error": f"unknown tool {name}"}
    trace.log("tool_call", tool=name, args=args, result=out,
              ms=round(1000 * (time.perf_counter() - t0), 1))
    return out


# ---------------------------------------------------------------- rules planner
def plan_with_rules(fleet, trace, asset_id, at_hour) -> dict:
    s = call_tool(fleet, trace, "read_sensors", {"asset_id": asset_id}, at_hour)
    h = call_tool(fleet, trace, "get_history", {"asset_id": asset_id}, at_hour)

    if not s["alert_active"] and s["hours_above_3sigma"] == 0:
        call_tool(fleet, trace, "search_manual", {"query": "alarm limits"}, at_hour)
        trace.log("decision", rule="no persistent anomaly and vibration within 3 sigma -> no work order")
        return {
            "status": "normal",
            "diagnosis": (f"Operating normally. Vibration {s['vibration_mm_s']} mm/s vs. baseline "
                          f"{s['vibration_baseline_mm_s']} ({s['vibration_change_pct']:+}%), never above 3 sigma "
                          f"in the current window; bearing temperature change {s['bearing_temp_change_c']:+} °C. "
                          f"{h['failures_last_12_months']} failures in the last 12 months."),
            "confidence": "High for the current state.",
            "action": "No work order. Continue routine monitoring.",
            "work_order": None,
        }

    m_bearing = call_tool(fleet, trace, "search_manual", {"query": "bearing wear"}, at_hour)
    m_misal = call_tool(fleet, trace, "search_manual", {"query": "misalignment"}, at_hour)
    temp_rising = s["bearing_temp_change_c"] >= 2.0
    cause = ("Probable bearing wear (manual §7.2)." if temp_rising else
             "Early deviation; bearing temperature is not rising yet, so the cause is open between "
             "early bearing wear (manual §7.2) and misalignment (manual §7.4).")
    trace.log("reasoning", temp_rising=temp_rising, cause=cause,
              manual_sections=[m_bearing[0]["section"], m_misal[0]["section"]])

    above_alarm = s["vibration_mm_s"] >= ALERT_VIBRATION
    soon = s["rul_estimate_h"] <= 100
    priority = "high" if (above_alarm or soon) else "medium"
    rul_txt = ("more than 300 h" if s["rul_capped"] else f"about {int(s['rul_estimate_h'])} h")
    evidence = (f"Vibration {s['vibration_mm_s']} mm/s ({s['vibration_change_pct']:+}% vs. baseline "
                f"{s['vibration_baseline_mm_s']}), {s['hours_above_3sigma']} h above 3 sigma, bearing temperature "
                f"{s['bearing_temp_baseline_c']} -> {s['bearing_temp_c']} °C, estimated remaining life {rul_txt}.")
    actions = ("Inspect the bearing and take an oil sample within 72 h; monitor vibration and temperature "
               f"continuously; stop the machine if vibration reaches {TRIP_VIBRATION} mm/s."
               if priority == "high" else
               "Inspect the bearing and take an oil sample within 7 days; check alignment; "
               "keep monitoring.")
    trace.log("decision", rule=f"alert active; above alarm={above_alarm}; RUL<=100h={soon} -> {priority}")
    wo = call_tool(fleet, trace, "create_work_order",
                   {"asset_id": asset_id, "priority": priority, "actions": actions, "evidence": evidence},
                   at_hour)
    conf = ("Moderate to high" if (temp_rising and priority == "high") else "Moderate")
    return {
        "status": f"{priority} risk",
        "diagnosis": f"{cause} {evidence}",
        "confidence": (f"{conf}. Last bearing change {h['last_bearing_change_months']} months ago, last oil "
                       f"analysis {h['last_oil_analysis_months']} months ago ({h['last_oil_analysis_result']})."),
        "action": f"Opened {wo['id']} ({priority} priority): {actions}",
        "work_order": wo,
    }


# ------------------------------------------------------------ nightshift planner
SYSTEM_PROMPT = (
    "You are a maintenance engineer's assistant for pumps and compressors. Investigate the asset "
    "with the tools, deciding which ones to call. Open a work order only when the evidence justifies "
    "it. Answer in three parts: 1) probable diagnosis with the numbers, 2) confidence and what you "
    "cannot know from the data, 3) action. All data is synthetic."
)


def plan_with_nightshift(fleet, trace, asset_id, at_hour) -> dict:
    base = os.environ["METERLESS_BASE_URL"].rstrip("/")
    key = os.environ.get("METERLESS_API_KEY", "")
    model = os.environ.get("METERLESS_MODEL", "default")
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Investigate asset {asset_id}."}]
    wo = None
    for _ in range(8):
        body = {"model": model, "messages": messages, "tools": tools.TOOL_SCHEMAS}
        req = urllib.request.Request(
            f"{base}/chat/completions", data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                     "meterless-app": "petro-mech-ai", "meterless-scope": "user"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
        msg = resp["choices"][0]["message"]
        trace.log("model_call", provider="meterless-nightshift", model=resp.get("model", model),
                  ms=round(1000 * (time.perf_counter() - t0)), usage=resp.get("usage"),
                  receipt={k: v for k, v in resp.items() if k not in ("choices",)})
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return {"status": "see report", "diagnosis": msg.get("content", ""),
                    "confidence": "", "action": "", "work_order": wo}
        for c in calls:
            args = json.loads(c["function"]["arguments"] or "{}")
            out = call_tool(fleet, trace, c["function"]["name"], args, at_hour)
            if c["function"]["name"] == "create_work_order":
                wo = out
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": json.dumps(out, ensure_ascii=False)})
    return {"status": "stopped", "diagnosis": "Too many steps.", "confidence": "", "action": "",
            "work_order": wo}


def investigate(fleet, asset_id, at_hour, trace_path: Path) -> dict:
    trace = Trace(trace_path)
    trace.log("input", asset_id=asset_id, hour=at_hour)
    planner = "rules"
    if os.environ.get("METERLESS_BASE_URL"):
        try:
            result = plan_with_nightshift(fleet, trace, asset_id, at_hour)
            planner = "meterless-nightshift"
        except (urllib.error.URLError, KeyError, TimeoutError, json.JSONDecodeError) as e:
            trace.log("error", where="nightshift", error=str(e), fallback="rules")
            result = plan_with_rules(fleet, trace, asset_id, at_hour)
    else:
        result = plan_with_rules(fleet, trace, asset_id, at_hour)
    result = {"asset_id": asset_id, "hour": at_hour, "planner": planner, **result}
    trace.log("output", **result)
    return result


def pick_examples(fleet: tools.Fleet) -> list[dict]:
    """Choose three test-set assets (never seen in training) and hours to inspect."""
    from ml.train import split_assets
    _, test_ids = split_assets(fleet.raw["asset_id"].unique(), 18, 42)
    test_ids = sorted(test_ids)
    raw = fleet.raw
    picked = []
    for ex, aid in zip(EXAMPLES, test_ids[:3]):
        g = raw[raw["asset_id"] == aid]
        last = int(g["hour"].max())
        onset = int(g.loc[g["degrading"], "hour"].min())
        if "hours_before_failure" in ex:
            hour = last - ex["hours_before_failure"]
        elif "hours_after_onset" in ex:
            # first hour after onset with an active alert and 6+ h above 3 sigma
            hour = onset + ex["hours_after_onset"]
            while hour < last:
                s = tools.read_sensors(fleet, aid, hour)
                if s["alert_active"] and s["hours_above_3sigma"] >= 6:
                    break
                hour += 6
        else:
            hour = 150 + ex["hours_after_baseline"]
        picked.append({"name": ex["name"], "asset_id": aid, "hour": hour})
    return picked


def to_markdown(r: dict) -> str:
    lines = [f"# {r['asset_id']} at hour {r['hour']}: {r['status']}", "",
             f"Planner: `{r['planner']}`", "",
             f"**1) Diagnosis.** {r['diagnosis']}", "",
             f"**2) Confidence.** {r['confidence']} What the agent cannot know:"]
    lines += [f"- {x}" for x in LIMITS]
    lines += ["", f"**3) Action.** {r['action']}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", action="store_true")
    ap.add_argument("--asset")
    ap.add_argument("--hour", type=int)
    a = ap.parse_args()
    fleet = load_fleet()
    if a.examples:
        out = ROOT / "examples"
        for ex in pick_examples(fleet):
            d = out / ex["name"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "input.json").write_text(json.dumps({"asset_id": ex["asset_id"], "hour": ex["hour"]}, indent=2))
            r = investigate(fleet, ex["asset_id"], ex["hour"], d / "trace.jsonl")
            (d / "output.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
            (d / "output.md").write_text(to_markdown(r))
            print(to_markdown(r))
    else:
        r = investigate(fleet, a.asset, a.hour, ROOT / "logs" / f"{a.asset}_{a.hour}" / "trace.jsonl")
        print(to_markdown(r))


if __name__ == "__main__":
    main()
