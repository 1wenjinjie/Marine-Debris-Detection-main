# 海洋垃圾检测项目年度研究计划（2026-03-28 重写版）

> 本版基于当前已冻结的 `MARIDA v2` 实验结果重写，替代旧的 `PKD` 主线版本。
> 当前项目的可信结论以 `MARIDA_V2_RESULTS_FREEZE_2026-03-28.md` 为准。

---

## 一、当前项目定位

本项目聚焦于基于 Sentinel-2 多光谱遥感影像的海洋垃圾检测，当前已经从“泛化的模型改进尝试”收敛为一个更明确的研究问题：

**在 leak-free 的 MARIDA v2 基准上，如何稳定提升极小海洋垃圾目标（Marine_Debris）的可检测性，并把结果整理成可复现、可解释、可写论文的证据链。**

当前最重要的变化是：

- 研究主线已经不再是 `PKD`。
- 研究主线已经转为：
  - `MARIDA v2` 数据集重建与可信评估
  - Marine_Debris 检测失败机制诊断
  - `imgsz=1024 + focal loss` 的稳定改进
  - post-hoc calibration 作为部署导向收尾

---

## 二、截至目前的冻结结论

### 2.1 数据集层面

- `MARIDA v2` 已完成重建与验证。
- 采用 scene-level split，避免了旧版本中的 same-image leakage。
- 当前冻结数据集包含 `1381` 个 patch，其中 `367` 个含目标、`1014` 个纯背景。
- 该数据集已经可以作为论文中的正式实验基准。

### 2.2 基线结论

6 通道 baseline（3-seed）已经冻结：

- `mAP@0.5 = 0.193 +- 0.018`
- `Marine_Debris AP50 = 0.016 +- 0.006`
- 在实用阈值下，Marine_Debris 几乎不可检测

这说明：

- 标准 6ch YOLO 在该任务上并非“训练坏了”
- 但对极小海洋垃圾目标的实用检测能力不足

### 2.3 当前最优结果

当前最优 6 通道模型已经冻结为：

- `imgsz=1024`
- `focal loss (gamma=2.0, alpha=0.25)`
- 3-seed 验证完成

其核心结果是：

- `conf=0.05` 下，Marine_Debris `TP = 5.7 +- 0.6 / 13`
- 对应实用 recall 约 `44%`
- 相比 baseline 的 `0/13`，是清晰且稳定的改进

这已经构成当前论文主结果。

### 2.4 calibration 结论

在冻结 best 6ch model 后，进一步完成了一个小型 post-hoc calibration 实验：

- `Platt scaling` 在 `conf=0.05` 下提升很小
- 但在 `conf=0.1` 下效果明显：
  - 未校准：`0.7 +- 0.5`
  - 校准后：`5.0 +- 0.0`

这说明 calibration 的价值主要在于：

- 提升 score usability
- 稳定部署阈值
- 使较高阈值也能保留有效 TP

因此 calibration 是一个有价值的收尾实验，但不是新的主贡献。

---

## 三、当前论文主线

### 3.1 建议的核心叙事

论文最合理的主线应为：

1. 旧 MARIDA 处理流程和旧 split 存在评估风险，因此需要 `MARIDA v2`
2. 在 leak-free 的 `MARIDA v2` 上，标准 6ch baseline 对 Marine_Debris 在实用阈值下失效
3. 通过机制诊断可以看出，问题不只是“没有学到”，而是“小目标 + 低置信度 + operating point 敏感”
4. 更高输入分辨率 (`1024`) 与 focal loss 能稳定提升 Marine_Debris 检测能力
5. calibration 可进一步改善 score 的部署可用性

### 3.2 论文中各条支线的角色

#### 主线内容

- `MARIDA v2` 数据集与构建说明
- baseline 与失败机制诊断
- `imgsz=1024 + focal loss` 主结果
- calibration 收尾分析

#### 分析性旁线

- `7ch oracle teacher`
- `P0-B` 的 shortcut collapse
- `PKD` 的失败与分析

这些内容仍然有价值，但更适合写成：

- motivation
- upper bound analysis
- negative finding
- discussion

而不是论文主结果。

#### 当前不建议作为主线继续深挖

- 重新回到 `PKD` 作为核心方法
- 把 `FDI` 作为当前最优先方向
- 大规模继续尝试新 detection head
- 同时并行扩展多个新实验方向

---

## 四、未来 1-3 个月的核心安排

### Phase A：结果冻结与论文写作准备（最高优先级）

目标：把当前已经站住的结果整理成论文可直接使用的材料。

需要完成：

1. 主表、附表、图注、误差案例图整理
2. `MARIDA v2` 数据集章节草稿
3. baseline 与 best 6ch model 的结果章节草稿
4. calibration 小节草稿
5. discussion 中关于 threshold sensitivity 的表述

产出：

- 一版完整的论文实验章节草稿
- 一套统一口径的图表素材

### Phase B：有限补证据（只做高价值、低风险项）

目标：补足论文说服力，但不打开新的无底洞实验线。

优先候选：

1. 更系统的错误分析
2. class-wise / scene-wise 案例图整理
3. 若时间允许，做一个轻量外部验证或跨场景分析

执行原则：

- 只补“增强说服力”的证据
- 不再为了追求新方向而重开大规模实验

### Phase C：论文成稿与导师汇报

目标：从“实验完成”转向“论文表达完整”。

需要完成：

1. 引言重写
2. 方法章节重写
3. 实验章节整合
4. 结论与 limitation 明确化

---

## 五、当前不再作为主线推进的内容

以下方向可以保留为补充讨论，但不再作为近期主任务：

### 5.1 PKD

原因：

- 当前实证结果并未支持它作为主结果
- 与当前冻结结论不一致
- 若继续深挖，成本高且风险大

论文定位：

- negative finding
- privileged upper bound analysis

### 5.2 FDI

原因：

- 当前 3-seed 结果表明，大部分困难实例并非“完全不可分”
- 剩余 truly unrecoverable 实例仅有 `2` 个
- 目前投入产出比不高

论文定位：

- 未来工作
- 可选补充分析

### 5.3 继续堆结构模块

例如：

- P2 head
- 新 neck
- 注意力模块堆叠

原因：

- 当前项目的核心矛盾已经较明确
- 再堆模块容易让论文主线变散

---

## 六、建议的论文结构

### 6.1 章节结构

1. **Introduction**
2. **Related Work**
3. **MARIDA v2 Dataset and Evaluation Protocol**
4. **Baseline Failure Diagnosis**
5. **Improved 6ch Detection with 1024 Resolution and Focal Loss**
6. **Post-hoc Score Calibration**
7. **Discussion and Limitations**
8. **Conclusion**

### 6.2 表格建议

- Table 1：`MARIDA v2` 数据集统计
- Table 2：baseline 与 best 6ch model 主结果
- Table 3：Marine_Debris threshold sweep
- Table 4：scene-level stability / case-level stability
- Table 5：post-hoc calibration 对比

### 6.3 图像建议

- baseline 训练曲线
- best 6ch model 训练曲线
- baseline vs best qualitative predictions
- debris patch recoverability cases
- calibration 前后 operating point 对比图

---

## 七、风险判断

### 当前主要风险

1. 继续沿用旧 PKD 叙事，导致论文主线和现有结果错位
2. 为了“追更大贡献”而重新打开多个高风险实验方向
3. 图表和文字口径不统一，影响导师和评审理解

### 当前最大优势

1. 已有 leak-free 数据集基础
2. 已有稳定 3-seed 主结果
3. 已有明确的机制分析链条
4. 已有 calibration 收尾，可形成较完整的实验闭环

---

## 八、下一步执行建议

接下来最合理的顺序是：

1. 冻结当前实验口径，不再改主配置
2. 继续完善论文主表、图注和结果章节
3. 仅补少量高价值分析
4. 尽快进入完整写作阶段

一句话总结：

**项目现在最该做的不是再找新的模型突破，而是把已经拿到的“可信数据集 + 稳定主结果 + 可解释收尾”整理成一篇完整且自洽的论文。**
