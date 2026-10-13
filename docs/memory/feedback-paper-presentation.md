---
name: feedback-paper-presentation
description: "User's honesty/framing decisions for paper vs 汇报: oracle prior 汇报-only, honest ECE wording, imgsz wording, which classes to report"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a3ca46b2-90f2-46df-b0b0-eb6ecf921caf
---

How the user wants results presented (their explicit calls):

- **Oracle prior leak** goes in 汇报 / private discussion only, NOT the paper — "这就是错误，有必要说吗". (I pushed back that the leak-attribution methodology is the strongest academic angle; unresolved — worth raising with 老师.) See [[project-leak-attribution-2026-06]].
- **Calibration** must use the held-out test ECE (1.75×), not the in-sample val number (3.5×). See [[project-v1-calibration-2026-06]].
- **imgsz=1024** must be described as "more feature-map cells / Marine_Debris boxes inside P3 head's design range", never as "higher resolution" (it's letterbox upsampling). Also in CLAUDE.md.
- **Class reporting:** headline = Marine_Debris AP50/F1 (the hard class); secondary = mAP50 noted "MD dominates"; only plot/list classes with GT (MD, Natural_Organic, Ship, Foam); never report v1-test 0-GT classes.
- Active line is **v1 only**; v2 3-class is shelved.

**Why:** the user prioritizes reviewer-proof honesty over squeezing performance numbers; mislabeling in-sample/leaky results as honest is the cardinal sin here. **How to apply:** when drafting paper text, default to the conservative honest framing above; flag any number whose honest provenance I can't confirm.
