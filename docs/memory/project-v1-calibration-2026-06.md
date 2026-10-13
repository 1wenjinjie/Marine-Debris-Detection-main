---
name: project-v1-calibration-2026-06
description: "Platt scaling on honest BiFPN; held-out test ECE 0.051→0.029 (1.75×); val in-sample 0.103→0.029 (3.5×) was the original overstated headline, corrected"
metadata: 
  node_type: memory
  type: project
  originSessionId: a3ca46b2-90f2-46df-b0b0-eb6ecf921caf
---

Confidence calibration via Platt scaling on the honest BiFPN+1024 model. Fit: a=0.4078, b=-0.5091 (`docs/paper/calibration/platt_fit.json`; figure `reliability_diagram.png`; sweep `threshold_sweep_test.json`). Script `scripts/v1_calibration_platt.py` produces a 2×2 reliability diagram (rows=val/test, cols=raw/calibrated) with empty count-bins filtered out.

**Honesty fix (2026-05-23):** the original headline "ECE 0.103→0.029 (3.5×)" was the **val in-sample** number (Platt was fit on val → in-sample). The defensible claim is the **held-out test**: ECE **0.0514→0.0293 (1.75×)**. Paper §4.5 now reports both with the in-sample/held-out distinction explicit. Deployment threshold τ=0.25 chosen from the calibrated sweep → F1=0.604.

**Why:** writing the val number as the headline would overstate calibration gain — a reviewer-fatal honesty slip the user/I caught and corrected. **How to apply:** always quote test ECE 1.75× as the result; mention val 3.5× only as in-sample reference. See [[project-status-2026-06-22]].
