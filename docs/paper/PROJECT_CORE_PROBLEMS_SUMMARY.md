# Marine Debris Detection: Project Core Problems Summary

## Executive Summary
截至2026-03-24，项目存在四大核心问题：
1. **10ch策略全面失败**：性能相比7ch下降7.3%，未能实现预期突破。
2. **知识热力图设计缺陷**：在10ch+SIoU组合中导致Recall下降20.6%，产生监督信号冲突。
3. **实验设计层级混乱**：多个设计变量未完全隔离，归因链条不完整。
4. **论文发表策略被困**：原定10ch突破叙事未能实现，需要战略重调整。

---

## Problem 1: 10-Channel Route Has Failed to Surpass 7-Channel Baseline[-7.3% F1 Gap]

### Evidence
| Configuration | F1 | Precision | Recall | mAP50 | vs SOTA | Status |
|---|---:|---:|---:|---:|---|---|
| **7ch Baseline (SOTA-aligned)** | **0.6774** | **0.6281** | **0.7351** | **0.6673** | ✅ +0.3% | ✓ Current Best |
| Exp-C: 10ch + SIoU | 0.6280 | 0.6360 | 0.6200 | 0.4810 | ❌ -7.3% | Best in 10ch |
| Exp-D: 10ch + SIoU + Heatmap | 0.5260 | 0.5640 | 0.4930 | 0.4890 | ❌ -22.1% | Failed |
| Exp-A/B: 10ch + CIoU/EIoU | 0.4620 | 0.8740 | 0.3140 | 0.4370 | ❌ -31.6% | Baseline |

### Problem Statement
Adding FDI + NDVI + additional S2 bands, contrary to expectations, resulted in absolute performance degradation. The best 10ch result (Exp-C: F1=0.628) still lags the 7ch baseline (F1=0.6774) by 7.3 percentage points.

### Root Cause Analysis

#### A. Data-Level Issues
- Spectral noise: FDI/NDVI may introduce noise in MARIDA dataset where RGB+NIR already saturate
- Sample-complexity mismatch: 347 training samples constant, but 10ch increases model degrees of freedom
- Normalization bias: NDVI/RedEdge lose physical meaning when forced into [0,1] range

#### B. Feature-Level Issues
- Heatmap prior dominance: 7th channel encapsulates GT spatial priors strongly
- RGB+NIR sufficiency: Marine plastic patterns may be fully characterized by visible+NIR
- Feature interaction: 10ch requires architecture redesign without it

### Practical Impact
- Cannot justify 10ch as primary contribution unless resolved
- Publication mainline must shift to 7ch-centric narrative

---

## Problem 2: Knowledge Heatmap Design Contains Fundamental Flaw

### Evidence
```
Comparison of Heatmap Impact:
  Without Heatmap (Exp-C):    F1 = 0.628, Recall = 0.620
  With Heatmap (Exp-D):       F1 = 0.526, Recall = 0.493
  ─────────────────────────────────────────────────────────
  Degradation:                ΔF1 = -16.3%, ΔRecall = -20.6%
```

### Problem Statement
The heatmap supervision, previously supporting 7ch training, exhibits destructive interaction with 10ch+SIoU. Adding heatmap causes 20.6% Recall collapse, contradicting original design intent.

### Root Cause Analysis

#### A. Supervision Signal Conflict
- Optimization target tension: BBox detector optimizes for box placement; heatmap enforces center concentration
- Geometric mismatch: Can't satisfy both constraints without severe compromise
- Gradient divergence: Gradients from different losses may partially cancel

#### B. Prior Information Redundancy
- GT-based heatmap is static offline prior, unlike learned channels
- Test-time absence: Heatmap unavailable at inference (no GT); model must learn to ignore it
- Loss of learned capacity: Model allocates capacity to heatmap rather than robust features

### Practical Impact
- Heatmap supervision is NOT universally beneficial
- Stacking heatmap+SIoU+10ch is invalid; each must be isolated first

---

## Problem 3: Experimental Attribution Remains Incompletely Isolated

### Current Experiment Structure
```
Current (mixed variables):
  Exp-A/B:  10ch + {CIoU | EIoU}              ← Baseline
  Exp-C:    10ch + SIoU                       ← Best 10ch
  Exp-D:    10ch + SIoU + Heatmap             ← Failed

Missing (critical for attribution):
  ✗ Exp-E: 7ch + SIoU                    ← Does SIoU help 7ch?
  ✗ Exp-F: 10ch + CIoU + tuned_opts      ← Is 10ch intrinsically worse?
  ✗ Exp-G: 10ch + SIoU + minimal_aug    ← Is augmentation the culprit?
```

### Problem Statement
Without strict One-Factor-At-a-Time isolation, cannot conclusively answer:
- Does SIoU loss benefit 7ch? (Only 10ch+SIoU tested)
- Is 10ch fundamentally incompatible? (Confounded by loss+augmentation)
- Can hyperparameter tuning alone recover 10ch? (Not isolated)

### Practical Impact
- Causal claims become indefensible in reviews
- Incomplete ablation may trigger major revisions or desk rejection

### Missing Experiments
| ID | Config | Goal | Expected |
|---|---|---|---|
| Exp-E | 7ch + SIoU | Isolate SIoU | If F1 > 0.675: compatible |
| Exp-F | 10ch + tuned | Isolate 10ch | If F1 < 0.65: data-limited |
| Exp-G | 10ch + minimal_aug | Isolate augment | If F1 > 0.60: aug is culprit |

---

## Problem 4: Publication Narrative Strategy Has Become Trapped

### Current Strategic Dilemma
```
Original target: F1 > 0.70 via 10ch
Current reality:
  - 7ch: F1 = 0.6774 (valid result)
  - 10ch: F1 = 0.628  (fails main narrative)
  
Options:
  A: Publish 7ch only (limited novelty)
  B: Continue 10ch (high risk)
  C: Reframe narrative (low risk)
```

### Narrative Reframing Strategies

#### Strategy A: Channel Optimization ✓ Valid
"Why 7 channels outperform 10 channels in multispectral marine debris detection"
- Contribution: Systematic channel selection principle
- Venues: Remote Sensing, IEEE TGRS
- Evidence: Existing 7ch vs 10ch + failed experiments

#### Strategy B: Loss Function Evaluation ✓ Valid
"Comparative study of IoU loss functions (CIoU, EIoU, SIoU)"
- Contribution: Rigorous loss ablation on 2 channel settings
- Venues: IEEE GRSL, ML in Remote Sensing
- Evidence: FINAL_ABLATION_REPORT.json complete data

#### Strategy C: Negative Results ✓ Valid
"Lessons from unsuccessful attempts to extend with spectral indices"
- Contribution: Failure modes documented
- Venues: Remote Sensing, Sensors
- Evidence: Exp-D failure analysis

#### Strategy D: Knowledge Distillation (If time permits)
"Lightweight detection via knowledge distillation"
- Prerequisites: Stable 7ch baseline (have it)
- Risk: Medium

### Practical Impact
- Not publishing worse than honest reframing
- Waiting indefinitely risks missing publication window
- Reframing immediately makes 7ch result publishable+novel

---

## Key Metrics Reference

### Full Experiment Comparison
| Exp | Ch | Loss | Heat | F1 | Prec | Rec | mAP50 | Status | Gap |
|---|---|---|---|---:|---:|---:|---:|---|---|
| 7ch BL | 7 | CIoU | ✓ | **0.6774** | 0.6281 | 0.7351 | 0.6673 | Best | 0% |
| Exp-A | 10 | CIoU | ✗ | 0.462 | 0.874 | 0.314 | 0.437 | Fail | -31.8% |
| Exp-B | 10 | EIoU | ✗ | 0.462 | 0.874 | 0.314 | 0.437 | Fail | -31.8% |
| Exp-C | 10 | SIoU | ✗ | 0.628 | 0.636 | 0.620 | 0.481 | Best10 | -7.3% |
| Exp-D | 10 | SIoU | ✓ | 0.526 | 0.564 | 0.493 | 0.489 | Fail | -22.4% |

---

## Immediate Action Items

### Priority 1: Lock 7ch Reproducibility (This Week)
- [ ] Archive config, seed, best.pt to immutable storage
- [ ] Write reproducibility script with fixed seeds
- [ ] Generate final metrics snapshot
- [ ] Document Ultralytics/CUDA/PyTorch versions

### Priority 2: Isolation Experiments (Next 2 Weeks)
- [ ] **Exp-E**: Train 7ch + SIoU (SIoU compatibility)
- [ ] **Exp-F**: Retrain 10ch + reduced augmentation
- [ ] **Exp-G**: 10ch + SIoU + minimal aug (salvage attempt)

### Priority 3: Narrative & Writing (Parallel)
- [ ] Decide publication strategy (A/B/C/D)
- [ ] Draft methods section
- [ ] Prepare ablation tables+figures
- [ ] Write introduction for target venue

### Priority 4: Decision Gate (Next Week End)
- [ ] If Exp-E/G: F1 > 0.67 → continue 10ch with confidence
- [ ] If all < 7ch → commit to 7ch narrative immediately
- [ ] Each experiment needs clear decision criteria

---

## Summary & Recommendation

**Status**: Validated 7ch baseline (F1=0.6774, SOTA+0.3%); 10ch unproven due to risks.

**Risk Assessment**:
- **High**: Waiting indefinitely for 10ch breakthrough
- **Medium**: Publishing 7ch as "incremental" without context
- **Low**: Pivoting to channel/loss narrative using existing data

**Recommendations**:
1. Execute Exp-E (7ch+SIoU) immediately to close attribution gap
2. Decide narrative this week based on current+Exp-E data
3. Begin writing in parallel (no idle time)
4. Commit to 3-4 week submission target

**Executive summary**: 
Reproducible 7ch result exists; 10ch needs decisive experiments or pivot to 7ch-centric publication narrative.

---

Last updated: 2026-03-24
Data sources: docs/context.md, PROJECT_ARCHITECT_PM_SUMMARY.md, runs_marida/FINAL_ABLATION_REPORT.json
