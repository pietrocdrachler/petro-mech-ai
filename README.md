# Petro Mech AI

A predictive maintenance agent for rotating equipment (pumps and compressors) in Oil & Gas.
Open Agent Hackathon 2026, track **The Agent That Can Explain Why**.

- **Demo video (2:44, narration in Portuguese):** https://youtu.be/ylcaJ8-7U20

> **All data is synthetic.** The results below show that the pipeline and the agent work end to
> end, not that the model is validated in the field. The real next step is a pilot with plant data.

## What the agent does

For each asset, the agent decides which tools to call and then explains its decision:

| Tool | What it returns |
|---|---|
| `read_sensors` | Runs the ML pipeline: vibration and bearing temperature vs. baseline, hours above 3σ, anomaly score, alert flag, remaining useful life |
| `get_history` | Last bearing change, last oil analysis, last alignment, failures |
| `search_manual` | Alarm and trip limits, bearing wear, misalignment |
| `create_work_order` | Opens an order, only when the evidence justifies it |

Every answer has three parts: a **diagnosis** backed by numbers, a **confidence level** that
states what the agent cannot know (for example, no frequency spectrum to separate bearing wear
from misalignment), and an **action**, which can be "no work order".

## Run it

```bash
pip install -r requirements.txt
python -m ml.train               # simulate 90 assets, train, evaluate -> models/, metrics.json
python -m agent.run --examples   # run the agent on the 3 examples -> examples/
python -m agent.run --asset A002 --hour 737
pytest -q
```

Everything runs offline in under a minute, with no API keys.

## Three examples: input → output

All three are test assets the models never saw in training. Full outputs and step-by-step
traces are in [`examples/`](examples/).

**1. Late bearing wear** · input `{"asset_id": "A002", "hour": 737}`
- Diagnosis: probable bearing wear (manual §7.2). Vibration 3.68 mm/s (+106% vs. baseline 1.79),
  48 h above 3σ, bearing temperature 55.7 → 64.9 °C, estimated remaining life about 38 h.
- Confidence: moderate to high; no spectrum, lubrication or load data.
- Action: **opened WO-1001, high priority**: inspect bearing and oil sample within 72 h,
  stop if vibration reaches 7 mm/s.

**2. Early deviation** · input `{"asset_id": "A004", "hour": 767}`
- Diagnosis: probable bearing wear. Vibration 1.85 mm/s (+22%), 6 h above 3σ, bearing
  temperature +2.0 °C, estimated remaining life about 155 h.
- Confidence: moderate.
- Action: **opened WO-1002, medium priority**: inspect within 7 days, check alignment.

**3. Healthy** · input `{"asset_id": "A007", "hour": 300}`
- Diagnosis: operating normally. Vibration 2.16 mm/s vs. baseline 2.23 (−3%), never above 3σ,
  no failures in 12 months.
- Confidence: high for the current state.
- Action: **no work order**, routine monitoring.

## Agent traces

Each run writes a `trace.jsonl` with one line per step: the input, every tool call with its
arguments, result and timing, the reasoning and decision rule, every model call (when a model
is used) and the final output. Example: [`examples/early_deviation/trace.jsonl`](examples/early_deviation/trace.jsonl).

## Planner and Meterless Nightshift

- **Default: rules planner.** A transparent rule-based planner, so the repository runs anywhere.
  Every output states `planner: rules`.
- **Meterless Nightshift.** If `METERLESS_BASE_URL` is set (for example
  `http://127.0.0.1:8088/v1`), the agent sends each step to Nightshift's OpenAI-compatible
  `chat/completions` endpoint with tool calling, lets the model choose the tools, and logs each
  model call and the response metadata in the trace. Optional: `METERLESS_API_KEY`,
  `METERLESS_MODEL`. If the call fails, the agent logs the error and falls back to the rules
  planner. **This client is implemented but was not tested against a live Nightshift instance
  before the deadline.**

## The web demo

The web demo shown in the video is a separate front end built on claude.ai: the same four tools
over its own simulated readings, with Claude choosing the tools. The agent in this repository
is the version connected to the ML pipeline.

## ML pipeline results (synthetic data)

90 simulated assets, split **by asset**: 72 for training, 18 for testing that are never seen
during training, so no test hour leaks into the model. Seed 42 (the default):

| Metric | Result |
|---|---|
| Failures detected before trip | 18 / 18 |
| Median warning before failure | 161 h |
| RUL error (MAE) in the last 100 h | 12.1 h |
| False alarm episodes in healthy operation | 0 per 1,000 h |

Across five seeds (1 to 4 and 42) the results stay in the same range: 18/18 detected,
median warning 124 to 161 h, RUL error 11 to 15 h, no false alarm episodes.

**Read these numbers with care.** The simulator produces one failure mode with a clean,
monotonic signature, and an alert needs 6 consecutive anomalous hours. Real plant data is
noisier, has several failure modes and operating changes, so real performance will be lower
and the alert threshold will need recalibration.

## How the pipeline works

1. **Simulate.** Each asset runs healthy for 300 to 900 h, then bearing wear makes vibration,
   bearing temperature, pressure and motor current drift until vibration reaches the
   7 mm/s trip limit.
2. **Features.** The first 150 h of each asset are its baseline. Each hour gets the z-score of
   every sensor against that baseline, a 24 h trend, and the vibration ratio.
3. **Anomaly.** An Isolation Forest learns only healthy behavior. The alert threshold is the
   99.5th percentile of healthy training scores, and an alert needs 6 consecutive hours above it.
4. **Remaining useful life.** A Gradient Boosting regressor estimates hours to failure,
   capped at 300 h.

## Limitations

- One failure mode (bearing wear); history and manual are synthetic stand-ins.
- No real data, no vibration spectrum, no validation with maintenance experts.
- Alert threshold calibrated only on simulated healthy data.
- Nightshift client not tested live; Zetaris not integrated yet.

## Roadmap

1. Run the agent live on Meterless Nightshift.
2. Zetaris as the data layer: federated SQL over historian, maintenance records and manuals.
3. FastAPI, PostgreSQL and Docker Compose.
4. More failure modes and public degradation datasets.
5. Pilot with real plant data.

## License

MIT
