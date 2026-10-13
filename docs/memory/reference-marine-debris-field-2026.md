---
name: reference-marine-debris-field-2026
description: "Field state 2024-2026: MADOS replaced MARIDA as standard benchmark; SOTA is segmentation + cross-dataset generalization; key papers + URLs"
metadata: 
  node_type: memory
  type: reference
  originSessionId: a3ca46b2-90f2-46df-b0b0-eb6ecf921caf
---

Field scan done 2026-06-22 for the marine-debris-from-Sentinel-2 area.

- **MADOS** (ISPRS J. 2024) has become the standard benchmark over MARIDA: 174 scenes, ~1.5M annotated px, 15 classes incl. Marine_Debris + Oil_Spill. Reviewers will ask "why not MADOS". Paper https://www.sciencedirect.com/science/article/pii/S0924271624000625 · code https://github.com/gkakogeorgiou/mados · data https://zenodo.org/records/10664073
- **MariNeXt** (same group) beats baselines ≥12% F1/mIoU — the segmentation SOTA.
- **Binary reformulation** (Frontiers Mar. Sci., Feb 2026): U-Net binary seg + imbalance-aware loss + rarity sampling; within-dataset pixel F1=0.901; MARIDA↔MADOS cross-dataset avg drop only 4.4%. Closest direct comparator. https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2026.1765021/full . NOTE: their F1 is **pixel-level**, ours is **instance-level @ IoU≥0.5** — not directly comparable.
- **PANGAEA** GFM benchmark includes MADOS; CROMA ~67.5 mIoU, GFMs win in 10%-label regime (relevant to our 371-tile low-label setting). https://arxiv.org/pdf/2412.04204
- S2+YOLO recent work is mostly underwater/UAV; the few S2 cases confirm resolution limits hurt small-object recall.

**My recommendations (priority order):** ① add MADOS as external/zero-shot instance-level test (fixes our small-test-set risk; instance-level cross-dataset is a gap nobody filled); ④ reposition paper as rigorous instance-level + calibration eval, cite these 3; ② promote Finding D to a fair U-Net-vs-YOLO instance comparison; ③ add an imbalance-aware-loss ablation cell; ⑤ GFM fine-tuning as future work. See [[project-finding-d-segmentation]] [[project-status-2026-06-22]].
