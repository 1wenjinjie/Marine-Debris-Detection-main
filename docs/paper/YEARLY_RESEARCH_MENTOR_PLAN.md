# 海洋垃圾多光谱检测 — 年度研究计划（第二次修订）

> 本文件基于 2026-03-25 Phase 0 实验结果及后续深度讨论全面修订。
> Phase 0 已完成并产出明确结论，项目方向从"LUPI 直接拼接"转向"Privileged Knowledge Distillation (PKD)"。

---

## 一、Phase 0 实验结果（已完成）

### 1.1 三组对照实验

| 实验 | 配置 | Precision | Recall | mAP50 | mAP50-95 |
|------|------|-----------|--------|-------|----------|
| **P0-A** | 6ch 训练 + 6ch 测试（纯 baseline） | 0.675 | 0.317 | 0.312 | 0.171 |
| **P0-B** | 7ch 训练 + 6ch 测试（heatmap 置零） | 0.026 | 0.018 | 0.023 | 0.005 |
| **P0-C** | 7ch 训练 + 7ch 测试（GT heatmap） | 0.677 | 0.450 | 0.605 | 0.284 |

> 训练配置：YOLOv11n, epochs=50, batch=8, imgsz=512, seed=0, RTX 3090 Ti

### 1.2 Phase 0 结论（已冻结，不可推翻）

**结论 1：Direct LUPI concatenation fails.**

P0-B Recall = 0.018，相比 P0-A Recall = 0.317 暴跌 94%。7ch 模型在 heatmap 置零后几乎完全失效，说明模型对 heatmap 通道形成了 **catastrophic dependency**——它学到的是"依赖先验定位"而非"光谱特征识别"。

**结论 2：GT heatmap 确实携带极强的空间先验信号。**

P0-C mAP50 = 0.605 vs P0-A mAP50 = 0.312，差距 0.293。这个 gap 足以证明 heatmap 先验的价值，但这个价值在推理时不可获得（需要 GT），因此不能作为可部署方案。

**结论 3：历史结果 F1=0.6774 存在信息泄漏。**

P0-C 的配置即为历史实验的配置（7ch train + 7ch test with GT heatmap），该结果在科学上不可作为公平评估指标。

---

## 二、四项冻结决策

基于 Phase 0 结果，以下决策永久冻结：

1. **6ch 为唯一官方可部署基线。** 所有最终模型必须以 6ch（B4, B3, B2, B8, B11, B12）作为推理输入。
2. **7ch + GT heatmap 仅作为 oracle / teacher upper bound。** 不作为主模型，不作为论文主结果。
3. **Phase 0 结论写死：direct LUPI concatenation fails.** 不再尝试"7ch 训练 + 6ch 推理"的直接拼接路线。
4. **P0-B 的失败模式是论文核心证据。** 7ch 模型在 heatmap 置零后 Recall 从 0.450 暴跌至 0.018 的事实，是整篇论文 motivation section 的定量支撑，不可被后续实验覆盖或淡化。

---

## 三、论文叙事重定义

### 3.1 叙事升级

| 层级 | 旧叙事 | 新叙事 |
|------|--------|--------|
| 表层 | "多一个通道效果好" | → 任何人都能做，没有 insight |
| 中层 | "GT prior 直接用导致依赖而非学习" | → Phase 0 negative finding，有科学价值 |
| **论文级** | **"通过蒸馏将不可部署的训练时特权先验转化为可部署模型的能力"** | → **方法论贡献** |

### 3.2 核心论文主张

> 直接将 GT 派生空间先验作为输入通道会导致模型对先验形成灾难性依赖，而非学习可迁移的光谱表征；
> 但这种强先验可以通过 Privileged Knowledge Distillation 框架，在训练阶段迁移到纯 6ch 可部署模型中，
> 使其在推理时无需任何 GT 信息即可获得显著的检测性能提升。

### 3.3 论文标题方向

*"Learning from Privileged Spatial Priors: A Knowledge Distillation Framework for Marine Debris Detection in Sentinel-2 Imagery"*

---

## 四、PKD 框架设计

### 4.1 核心架构

```
┌─────────────────────────────────────┐
│  Privileged Teacher (训练阶段)       │
│  Input: 6ch spectral + GT heatmap   │
│  → 学到: 空间注意力 + 光谱特征       │
│  → 输出: 强检测结果 + 内部特征表示    │
│  → 即 P0-C 模型 (mAP50=0.605)       │
└──────────────┬──────────────────────┘
               │ 蒸馏 (response → feature → attention)
               ▼
┌─────────────────────────────────────┐
│  Deployable Student (推理阶段)       │
│  Input: 6ch spectral ONLY           │
│  → 学到: 从 teacher 迁移的空间感知    │
│  → 输出: 无需 GT 即可检测             │
└─────────────────────────────────────┘
```

### 4.2 三级蒸馏（逐级验证）

| 级别 | 蒸馏方式 | 内容 | 验证优先级 |
|------|---------|------|-----------|
| L1 | Response-level | Student 模仿 Teacher 的 soft output distribution | ⭐⭐⭐ 最先验证 |
| L2 | Feature-level | Student 对齐 Teacher 中间层特征表示 | ⭐⭐ 第二步 |
| L3 | Attention-level | Teacher 的 heatmap 感知注意力图指导 Student 自生成注意力 | ⭐ 第三步 |

### 4.3 关键发现：跨输入空间蒸馏的特殊性

**原始风险假设：** Teacher/Student 输入空间不同（7ch vs 6ch），KD 可能失效。

**实验结论（Phase 2 已验证）：**

1. **Response-level KD 成功，且是最优路径。** 与预期相反，"松约束"（只对齐输出分布）比"强约束"（对齐中间层特征）更有效。原因：response-level 给 student 自由度用 6ch 特征重构 teacher 的检测行为。
2. **Feature-level KD 有正增益但偏弱。** Teacher neck 编码了 heatmap 通道信息，student 的 6ch neck 特征空间与之有结构性差异，强制 L2 对齐效果有限。
3. **联合蒸馏反而干扰。** 两种 loss 梯度方向冲突（response 鼓励"更多检测"，feature 约束"表征对齐"），联合后 student 无法同时满足两个目标。

**论文价值：** 这一组结论构成了一个关于"跨输入空间知识蒸馏中蒸馏模式选择"的 insight，区别于标准 KD 文献中 teacher/student 同输入的假设。

---

## 五、实验执行计划与进展

### 核心原则：先证明机制成立，再逐级加码，最后堆证据

### Phase 1：PKD 最小可行验证 ✅ 已完成（2026-03-26）

**目标：回答"privileged prior 能不能通过蒸馏被 6ch student 吸收？"**

| 实验编号 | 配置 | Precision | Recall | mAP50 | mAP50-95 |
|---------|------|-----------|--------|-------|----------|
| P0-A (baseline) | 6ch 训练 + 6ch 测试 | 0.675 | 0.317 | 0.312 | 0.171 |
| **P1-RKD** | **6ch student + response KD** | **0.450** | **0.454** | **0.401** | **0.219** |
| P0-C (oracle) | 7ch 训练 + 7ch 测试 | 0.677 | 0.450 | 0.605 | 0.284 |

**决策门：✅ PASS** — mAP50 提升 +0.089（0.312→0.401），远超 0.02 阈值。Recall 从 0.317 跃升至 0.454，追平 oracle。

**关键发现：** 蒸馏信号明确存在。Teacher 从 heatmap 学到的"哪里值得关注"的信息，通过 soft label 成功传递给了 6ch student。

**Conf sweep 补充分析（Step 2.0）：**
在 conf=0.1 下公平对比，P1-RKD 的 mAP50=0.470 vs P0-A=0.335（+40%），mAP50-95=0.287 vs 0.196（+46%）。Precision 下滑部分由置信度校准导致，不完全是真误检。

### Phase 2：蒸馏模式消融 ✅ 已完成（2026-03-26）

| 实验 | KD 模式 | Precision | Recall | mAP50 | mAP50-95 |
|------|---------|-----------|--------|-------|----------|
| **P0-A** | — (6ch baseline) | 0.675 | 0.317 | 0.312 | 0.171 |
| **P2-FKD** | feature only | 0.502 | 0.338 | 0.374 | 0.178 |
| **P2-RFKD** | response + feature | 0.573 | 0.290 | 0.346 | 0.166 |
| **P2-RKD** | **response only** | **0.450** | **0.454** | **0.401** | **0.219** |
| **P0-C** | — (7ch oracle) | 0.677 | 0.450 | 0.605 | 0.284 |

**Phase 2 结论：**

1. **Response KD 是最有效的特权先验迁移路径。** mAP50、mAP50-95、Recall 均为最优。
2. **Feature KD 有增益但弱于 Response KD。** mAP50 +20% vs baseline，但 Precision 更保守（0.502 vs RKD 的 0.450）。
3. **联合蒸馏（RFKD）出现干扰效应 — 有价值的 negative finding。** 两种 loss 梯度方向可能冲突，联合反而比单独差。Teacher neck features 编码了 heatmap 通道信息，强制 L2 对齐不同输入空间的特征会扭曲 student 表征。
4. **Response KD 的"松约束"更适合跨输入空间蒸馏。** 只要求输出分布对齐，给 student 自由度用自己的方式学到相似检测行为。

> 这三组消融构成论文 Table 2 的核心，展示了蒸馏模式选择对跨输入空间知识迁移的影响。

### Phase 2.5：3-seed 稳定性验证 🔄 进行中

对 P0-A（baseline）和 P2-RKD（best method）跑 seed=0,1,2，报告均值 ± 标准差。

### Phase 3：证据补全（下一步）

| 任务 | 内容 | 论文位置 |
|------|------|---------|
| 3.1 SOTA 对比 | 提取 MARIDA 文献指标，横向对比 | Table 3 |
| 3.2 误差分析 | Class-wise P/R/F1, FP/FN 可视化, 混淆分析 | Section 6 |
| 3.3 Cross-scene | 按地理/日期重划 train/test，验证非随机过拟合 | Table 4 |
| 3.4 MADOS 外部验证 | Zero-shot transfer → 分析域偏移 → 可选 fine-tune | Section 7 |

### Phase 4：论文撰写与投稿（3-4 月）

---

## 六、目标期刊

| 期刊 | IF | 适配度 | 说明 |
|------|-----|-------|------|
| **IEEE TGRS** | ~8 | ⭐⭐⭐⭐⭐ | 方法+遥感应用，完美匹配 PKD 故事 |
| **ISPRS Journal** | ~12 | ⭐⭐⭐⭐ | 偏方法论的遥感论文，需要更强证据 |
| Remote Sensing of Environment | ~13 | ⭐⭐⭐ | 偏应用，需要更多环境科学讨论 |
| IEEE JSTARS | ~5 | ⭐⭐⭐⭐ | 备选，接受率更高 |

---

## 七、论文最终结构

1. **Introduction**：海洋垃圾检测的挑战 → GT 空间先验的价值与不可部署性 → PKD 框架动机
2. **Related Work**：海洋垃圾遥感监测 / 多光谱目标检测 / LUPI 理论 / 知识蒸馏
3. **Problem Formulation**：MARIDA 数据特点 → 6ch/7ch 通道设计 → Phase 0 negative finding 作为动机
4. **Method**：PKD 框架 → Teacher 设计 → 三级蒸馏 → Student 部署
5. **Experiments**：
   - Table 1: Phase 0 对照（motivation）
   - Table 2: PKD 消融（核心贡献）
   - Table 3: SOTA 对比
   - Table 4: Cross-scene 泛化
6. **Error Analysis**：class-wise + scene-wise + 失败模式
7. **External Validation**：MADOS zero-shot 迁移
8. **Discussion & Limitations**：数据规模约束 / 域偏移 / 部署距离 / 小数据下 PKD 的价值
9. **Conclusion**

---

## 八、时间线

```
2026-03  Phase 0 ✅ Phase 1 ✅ Phase 2 消融 ✅ (大幅超前进度)
2026-04  Phase 2.5: 3-seed 稳定性 + Phase 3.1: SOTA 文献对比
2026-05  Phase 3.2: 误差分析 (class-wise, FP/FN 可视化)
2026-06  Phase 3.3: Cross-scene 内部泛化验证
2026-07  Phase 3.4: MADOS 外部验证
2026-08  Phase 4 开始: 论文初稿
2026-09  导师审阅 + 修改
2026-10  投稿
2026-11  审稿期 / 准备答辩材料
2026-12  Buffer / 修稿 / 补实验
```

---

## 九、执行规则

1. **每轮只推进一个主问题。** 同一阶段内不并行推进多个大分支。
2. **每个正式实验都要有实验卡。** 记录：目标、假设、配置、命令、seed、结果、结论、是否进入论文主表。
3. **Phase 0 结果在论文定稿前不可更改口径。** 它是 motivation section 的定量支撑。
4. **先验证机制再优化效果。** Teacher 优化排在蒸馏验证之后，而非之前。
5. **数据量小不是回避而是叙事。** 论文中正面讨论 347 样本的局限，并将其转化为 PKD 的优势场景——"正因为标注稀缺，privileged prior 的蒸馏才更有价值"。
6. **证据标准决定论文级别。** 仅有 baseline + student 提升不够一区；必须补齐 negative result + 分层消融 + 3-seed + cross-scene + 外部验证。

---

## 十、答辩核心三问

- **你解决了什么问题？**
  如何将 GT 派生的强空间先验从不可部署的训练时特权信息，转化为纯光谱可部署检测模型的能力。

- **你的方法多解决了什么？**
  证明了直接拼接 GT 先验会导致灾难性依赖（Phase 0 negative finding），并提出了基于三级知识蒸馏的 PKD 框架，使 6ch student 在无 GT 信息时仍能显著超越 baseline。

- **你的方法还不能解决什么？**
  数据量限制下的统计置信度；跨传感器/跨区域泛化；从 benchmark 到实际海洋监测部署的工程差距。

---

*最后更新：2026-03-26 | Phase 0-2 实验完成，3-seed 稳定性验证进行中*
*实验数据：runs_phase0/, runs_phase1/, runs_phase2/*
