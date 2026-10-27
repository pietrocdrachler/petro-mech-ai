# A002 at hour 737: medium risk, work order opened

Planner: `meterless-nightshift` (model-chose-tools)

1) **Probable diagnosis** – Asset A002 shows a vibration of 3.68 mm/s (105.9 % above the 1.79 mm/s baseline) and has exceeded the 3‑sigma threshold for 48 h. Bearing temperature is 64.9 °C, 9.2 °C above the 55.7 °C baseline. The anomaly score (0.729) exceeds the 0.585 threshold and the ML‑estimated remaining useful life is 38 h. Manual section 7.2 flags sustained vibration rise above 3 sigma with rising bearing temperature as bearing‑wear indicators.

2) **Confidence & limits** – High confidence that the machine is trending toward a bearing‑wear fault given the vibration/temperature rise and anomaly score. Cannot definitively confirm bearing damage without a physical inspection and oil analysis; other causes (misalignment, looseness) are possible.

3) **Action taken** – Opened work order **WO‑1001** (medium priority) to inspect the bearing and take an oil sample per manual section 7.2, with continued monitoring of vibration and temperature.

Work order: WO-1001 (medium priority)

Every model call and its Meterless receipt is in trace.jsonl.
