# A004 at hour 767: medium risk, work order opened

Planner: `meterless-nightshift` (model-chose-tools)

**1) Probable diagnosis (with numbers)**  
- Vibration is **1.85 mm/s RMS**, 21.7 % above the 1.52 mm/s baseline and has exceeded the 3‑sigma threshold for **6 hours** (anomaly score 0.672 > threshold 0.585).  
- Bearing temperature is **57 °C**, 2 °C above the 55 °C baseline.  
- The manual (Section 7.2) states that sustained vibration rise above 3 σ together with a rising bearing temperature signals bearing wear and calls for inspection within 72 h and an oil sample.  
- The ML‑estimated remaining useful life (RUL) is **155 h**, indicating the unit is still functional but trending toward a failure point.

**2) Confidence and limits**  
- High confidence that the asset is showing early‑stage bearing‑wear symptoms per the manual’s criteria.  
- We cannot determine the exact wear mechanism (e.g., spalling, contamination) without a physical bearing inspection and oil analysis.  
- The RUL estimate is model‑based and may shift once actual condition data are collected.

**3) Action taken**  
- Opened work order **WO‑1002** (medium priority) to **inspect the bearing and take an oil sample** per the manual’s Section 7.2, justified by the vibration rise above 3 σ, sustained 6‑hour duration, and the 2 °C temperature increase.

Work order: WO-1002 (medium priority)

Every model call and its Meterless receipt is in trace.jsonl.
