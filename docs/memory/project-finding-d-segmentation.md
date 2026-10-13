---
name: project-finding-d-segmentation
description: "Finding D: U-Net instance recall ~0.59 vs YOLO ~0.15 on Marine_Debris — detection has a hard ceiling on sub-4px targets; long-term direction is segmentation-first"
metadata: 
  node_type: memory
  type: project
  originSessionId: a3ca46b2-90f2-46df-b0b0-eb6ecf921caf
---

**Finding D:** on the same honest v2split, a U-Net segmentation baseline reaches Marine_Debris instance recall ≈0.59 vs YOLO ≈0.15 — a ~4× gap. Sentinel-2 Marine_Debris boxes are ≈12–14px (sub-4px effective at native 256), below the detection framework's reliable floor. This is corroborated by the field (see [[reference-marine-debris-field-2026]]): the SOTA line is pixel-wise segmentation (MariNeXt, 2026 binary-reformulation U-Net pixel F1≈0.90), not box detection.

**Strategic implication:** the current YOLO paper can still be written (the unique angle is *instance-level + calibrated + deployment-threshold + leak-attribution*, which the pixel-F1 literature lacks), but the **next** project should not keep grinding on YOLO. Direction: segmentation-first → connected-component → instance post-processing, compared against YOLO under one shared instance protocol.

**Why:** keeps future effort from being spent optimizing a framework with a structural ceiling. **How to apply:** frame Finding D as a deliberate cross-paradigm comparison in the paper, not a footnote; pixel F1 (lit.) and instance F1 (ours) are NOT directly comparable — state this explicitly. See [[project-status-2026-06-22]].
