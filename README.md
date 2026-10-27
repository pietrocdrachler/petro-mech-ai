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
python -m agent.run --asset A002 --hour 737
pytest -q
```

Without any keys this runs offline in under a minute with the rules planner.

**With Meterless Nightshift** (the model chooses the tools and writes the explanation):

```bash
export METERLESS_BASE_URL=http://127.0.0.1:8088/v1
export METERLESS_API_KEY=<Nightshift runtime auth token>
export METERLESS_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
python -m agent.run --examples   # regenerates examples/ with live model calls and receipts
```

## Three examples: input → output

All three are test assets the models never saw in training. Each folder in
[`examples/`](examples/) has `input.json`, the agent's answer (`output.md`, `output.json`) and
the step-by-step `trace.jsonl`. The first lines of each `output.md` say which planner produced it.

| Example | Input | What the tools return | Expected decision |
|---|---|---|---|
| Late bearing wear | `{"asset_id": "A002", "hour": 737}` | vibration 3.68 mm/s (+106% vs. 1.79), 48 h above 3σ, bearing 55.7 → 64.9 °C, RUL ≈ 38 h | high-priority work order |
| Early deviation | `{"asset_id": "A004", "hour": 767}` | vibration 1.85 mm/s (+22%), 6 h above 3σ, bearing +2.0 °C, RUL ≈ 155 h | medium-priority work order |
| Healthy | `{"asset_id": "A007", "hour": 300}` | vibration 2.16 mm/s (−3%), never above 3σ, no failures in 12 months | no work order |

## Agent traces

Each run writes a `trace.jsonl` with one line per step: the input, every tool call with its
arguments, result and timing, the reasoning and decision rule, every model call with the
model Nightshift routed to and its Meterless compute receipt id, and the final output. Example: [`examples/early_deviation/trace.jsonl`](examples/early_deviation/trace.jsonl).

## Meterless Nightshift and NVIDIA Nemotron

The agent's model calls go through **Meterless Nightshift** running on a MacBook Air (M2),
using its OpenAI-compatible endpoint at `http://127.0.0.1:8088/v1`. Nightshift holds the
NVIDIA key and routes the calls to **NVIDIA Nemotron** (`nemotron-3.5-lightning-30b-a3b`), and
returns a signed compute receipt for each call, which the agent stores in the trace.

- **`model-chose-tools` mode:** the model gets the four tool definitions and decides which to
  call, in what order, and whether to open a work order.
- **`evidence-first` mode:** if the routed model answers without using tools, the agent runs
  the tools itself, takes the work-order decision with the rules planner, and asks the model
  only to write the explanation from that evidence. The switch is logged in the trace.
- **`rules` fallback:** with no `METERLESS_BASE_URL`, or if a call fails, the agent logs the
  error and uses the rule-based planner, so the repository always runs.

Every output says which planner and mode produced it.

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
- The model sometimes describes a work order instead of opening it; the trace shows what it actually called.
- Zetaris not integrated yet.

## Roadmap

1. Zetaris as the data layer: federated SQL over historian, maintenance records and manuals.
2. FastAPI, PostgreSQL and Docker Compose.
3. More failure modes and public degradation datasets.
4. Pilot with real plant data.

## License

MIT
