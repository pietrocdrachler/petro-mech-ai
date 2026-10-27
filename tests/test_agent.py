import json

import pytest

from agent.run import investigate, load_fleet, pick_examples


@pytest.fixture(scope="module")
def fleet():
    return load_fleet()


@pytest.fixture(scope="module")
def cases(fleet):
    return {c["name"]: c for c in pick_examples(fleet)}


def run(fleet, case, tmp_path):
    return investigate(fleet, case["asset_id"], case["hour"], tmp_path / "trace.jsonl")


def test_late_wear_opens_high_priority_order(fleet, cases, tmp_path):
    r = run(fleet, cases["late_bearing_wear"], tmp_path)
    assert r["work_order"]["priority"] == "high"


def test_healthy_asset_opens_no_order(fleet, cases, tmp_path):
    r = run(fleet, cases["healthy"], tmp_path)
    assert r["work_order"] is None


def test_trace_records_every_tool_call(fleet, cases, tmp_path):
    run(fleet, cases["early_deviation"], tmp_path)
    steps = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    kinds = [s["kind"] for s in steps]
    assert kinds[0] == "input" and kinds[-1] == "output"
    tools_used = [s["tool"] for s in steps if s["kind"] == "tool_call"]
    assert tools_used[0] == "read_sensors"
    assert "create_work_order" in tools_used
