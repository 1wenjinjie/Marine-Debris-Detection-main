---
name: project-leak-attribution-2026-06
description: "The GT-leakage finding: oracle prior ch7 was responsible for most of the original published F1; 4-cell attribution quantifies it"
metadata: 
  node_type: memory
  type: project
  originSessionId: a3ca46b2-90f2-46df-b0b0-eb6ecf921caf
---

The original strong v1 results were inflated by **GT leakage**, not honest signal. Two leaks: (1) scene-level split leak (fixed by v2split), (2) the real culprit — ch7 was an **oracle prior derived from GT masks**, leaking labels into the input.

**4-cell attribution** (Marine_Debris metric; exact values in `docs/paper/v1_*_aggregate.md` + `docs/paper/figures/leak_inflation.png` — verify metric definition there before quoting): original-split + oracle ≈ 0.524 → v2split + oracle ≈ 0.466 → v2split + oracle zeroed ≈ 0.141 → v2split + honest NDWI-inverse ch7 ≈ 0.349. Conclusion: oracle prior contributed the large majority of the headline; scene leak only a small slice. The honest ch7 (NDWI-inverse, physics-based, no GT) is what the paper uses.

Supporting controls: leaky-control (BiFPN gain vanishes under honest setting), native-256 control (letterbox upsampling is the real mechanism, not sensor resolution), unetprior counter-example (`v1_unetprior_*` — ch7 must be an independent physical signal).

**Why:** this is the most original analysis in the project AND a near-fraud landmine; must never be misremembered as honest. **How to apply:** user decided oracle prior goes in 汇报 only, NOT the paper (treated as an error, not a contribution) — but I argued the leak-attribution *methodology* is the project's strongest academic angle and worth reopening with the 老师. See [[feedback-paper-presentation]] [[project-status-2026-06-22]].
