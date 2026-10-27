# Petro Mech AI

A predictive maintenance agent for rotating equipment (pumps and compressors) in Oil & Gas.
Open Agent Hackathon 2026, track **The Agent That Can Explain Why**.

- **Live demo:** [add the public demo link]
- **Demo video:** [add the YouTube link]

> **All data is synthetic.** The results below show that the pipeline works end to end,
> not that the model is validated in the field. The real next step is a pilot with plant data.

## What the agent does

For each asset, the agent decides which tools to call and then explains its decision:

| Tool | What it returns |
|---|---|
| `read_sensors` | Vibration and bearing temperature against a 3σ baseline |
| `get_history` | Past failures, last bearing change, last oil analysis |
| `search_manual` | Alarm and trip limits, failure modes |
| `create_work_order` | Opens an order, only when the evidence justifies it |

Every answer has three parts: a **diagnosis** backed by numbers, a **confidence level** that
states what the agent cannot know (for example, no frequency spectrum to separate bearing wear
from misalignment), and an **action**. In the demo, compressor C-102 gets a high-priority work
order, pump P-205 a medium one, and pump P-310 is judged normal with no order opened.

## What exists today

| Layer | Status |
|---|---|
| Run-to-failure simulator (bearing wear, 4 sensors) | done, `ml/simulate.py` |
| Per-asset features (deviation from baseline, 24 h trend) | done, `ml/features.py` |
| Anomaly detection (Isolation Forest) + remaining useful life (Gradient Boosting) | done, `ml/train.py` |
| Automated tests | done, `tests/` |
| Agent with tools, web demo | published on claude.ai (see `demo/`) |
| FastAPI, PostgreSQL, Docker | not started |

The web demo and the ML pipeline are separate today: the demo uses its own simulated readings.
Serving the pipeline to the agent through an API is the first item on the roadmap.

## How to run

```bash
pip install -r requirements.txt
python -m ml.train   # simulates 90 assets, trains, evaluates, writes models/ and metrics.json
pytest -q
```

Takes about 20 seconds on a laptop.

## Results (synthetic data)

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

## How it works

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

- One failure mode (bearing wear).
- No real data, no vibration spectrum, no validation with maintenance experts.
- Alert threshold calibrated only on simulated healthy data.
- The demo agent does not yet call this pipeline.

## Roadmap

1. FastAPI serving the model to the agent.
2. PostgreSQL for readings, alerts and work orders; Docker Compose.
3. More failure modes and public degradation datasets.
4. A federated data layer over historian, maintenance history and manuals.
5. Pilot with real plant data.

## License

MIT
