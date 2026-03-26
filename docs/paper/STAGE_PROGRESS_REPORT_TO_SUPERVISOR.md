# 海洋垃圾多光谱遥感检测项目阶段性汇总报告

## 1. 汇总说明

本报告用于向导师汇报项目截至当前阶段的真实进展，所有结论均基于仓库内已经存在的训练日志、结果图像、分析文档和配置文件整理而成。

本报告的口径以 `2025-12-23` 至 `2025-12-27` 形成的阶段性主线为主，同时单独说明仓库中存在的后续结果线索。为避免误导，不同评估口径的结果不直接混合比较。

## 2. 研究目标与当前定位

本项目聚焦于基于 Sentinel-2 多光谱遥感影像的海洋垃圾检测，当前已经形成一条较清晰的工程与论文主线：

- 使用 YOLO 检测框架处理多通道 `.npy` 输入。
- 在多光谱通道基础上引入由标注 mask 构造的 `knowledge heatmap`，作为额外输入通道。
- 围绕 `7ch` 主线与 `10ch` 扩展主线开展对照与消融实验。

当前可以明确的是：

- 项目已经具备可运行的数据处理、训练、验证和结果可视化链路。
- 项目已经得到一组可支撑论文叙事的阶段性主结果。
- 项目已经完成一轮有结论的 `10ch` 消融分析。
- 项目暂时还不能称为“最终定稿结果”，因为后续还存在新的评估口径和更晚的结果记录。

## 3. 已完成工作的真实情况

截至目前，已经完成且有文件证据支持的工作主要包括以下几类。

### 3.1 数据与训练链路

- 已建立 `7ch` 与 `10ch` 数据配置。
- 当前记录的两套数据规模均为 `train=347 / val=114 / test=53`。
- 输入形式为多通道 `.npy` 张量。
- 训练依赖 Ultralytics，并配有自定义 dataset/trainer 以支持多通道输入。

对应文件：

- `configs/datasets/marida_mask_rescue_7ch.yaml`
- `configs/datasets/marida_mask_rescue_10ch.yaml`
- `ultralytics_plugins/mask_rescue_dataset.py`
- `ultralytics_plugins/mask_rescue_trainer.py`

### 3.2 预处理与知识通道

- 已有 knowledge heatmap 相关预处理实现与说明文档。
- 这一部分已经被整理为“重建说明”，因此在汇报中应如实表述为：当前脚本可支持复现项目流程，但其历史上属于重建实现，而不是最初版本源码的原样保留。

对应文件：

- `docs/PREPROCESSING_RECONSTRUCTION.md`
- `scripts/preprocessing/preprocess_generate_knowledge_channel.py`

### 3.3 阶段性文档整理

- 已形成项目背景恢复文档。
- 已形成阶段性项目快照总结。
- 已形成面向硕士毕业与投稿的路线图文档。

对应文件：

- `docs/context.md`
- `docs/paper/PROJECT_ARCHITECT_PM_SUMMARY.md`
- `docs/paper/MASTERS_THESIS_PUBLICATION_ROADMAP.md`

## 4. 阶段性核心实验结果

本部分只写有明确产物支撑的结果，并严格区分不同结果线。

### 4.1 历史主线结果：7ch + Knowledge Heatmap + CIoU

这是当前最适合写入“阶段性主线成果”的一组结果，也是项目在 `2025-12-23` 至 `2025-12-25` 左右形成的核心判断依据。

训练目录：

- `experiments/runs/baseline_sota_aligned_1223_2219`

关键指标如下：

| 指标 | 数值 |
| --- | --- |
| Precision | `0.6281` |
| Recall | `0.7351` |
| F1 | `0.6774` |
| mAP@0.5 | `0.6673` |
| mAP@0.5:0.95 | `0.3351` |
| Best Epoch | `38` |

阶段性结论：

- 该结果支撑了“`7ch + knowledge heatmap + CIoU` 是当时最优主线”的判断。
- 相比当时对齐的 SOTA 目标，项目在 Recall 上表现更突出。
- 这条主线更适合作为论文主体，因为结果、工程实现和叙事逻辑较完整。

证据来源：

- `experiments/runs/baseline_sota_aligned_1223_2219/results.csv`
- `docs/context.md`
- `docs/paper/PROJECT_ARCHITECT_PM_SUMMARY.md`

### 4.2 10ch 消融结果

`10ch` 系列实验主要用于验证更丰富光谱通道和不同 IoU Loss 的效果，当前已完成的阶段性消融结论如下。

| 实验组 | 配置 | Precision | Recall | F1 | mAP@0.5 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| Exp-A | `10ch + CIoU` | `0.874` | `0.314` | `0.462` | `0.437` | 基线 |
| Exp-B | `10ch + EIoU` | `0.874` | `0.314` | `0.462` | `0.437` | 无改进 |
| Exp-C | `10ch + SIoU` | `0.636` | `0.620` | `0.628` | `0.481` | 10ch 内最优 |
| Exp-D | `10ch + SIoU + Heatmap` | `0.564` | `0.493` | `0.526` | `0.489` | 回退 |

阶段性结论：

- `SIoU` 对 `10ch` 确实有效，显著修复了早期 Recall 偏低问题。
- 但 `10ch` 最优实验 `Exp-C` 的综合表现仍未超过 `7ch` 主线。
- 在 `10ch` 路线上继续叠加 heatmap 并未进一步提升，反而造成性能回退。

证据来源：

- `runs_marida/FINAL_ABLATION_REPORT.json`
- `runs_marida/README_ABLATION_STUDY.md`
- `runs_marida/exp_a_10ch_ciou_baseline/results.csv`
- `runs_marida/exp_c_10ch_siou3/results.csv`
- `runs_marida/exp_d_10ch_siou_heatmap/results.csv`

### 4.3 后续结果线索：README 中的 major classes 结果

仓库中还存在一条更晚的结果叙述，见 `README.md` 中的 `Current SOTA (Major Classes)`：

| 指标 | 数值 |
| --- | --- |
| Precision | `0.826` |
| Recall | `0.787` |
| F1 | `0.806` |
| Best threshold | `0.53` |

这一结果可以作为“后续项目演进线索”保留，但本次阶段性汇报不将其与 `7ch baseline` 和 `10ch ablation` 直接并入一张对比表，原因是：

- 它属于后续评估口径。
- 其描述中明确指向 `major classes`。
- 它更适合写成“后续补充结果”而不是“本阶段统一口径主结果”。

证据来源：

- `README.md`

## 5. 阶段性关键认识

基于目前已经形成的真实实验记录，可以较稳妥地总结出以下几点。

### 5.1 7ch 主线在阶段内最成熟

- `7ch + knowledge heatmap + CIoU` 是目前最适合当作阶段性论文主线的方案。
- 它不仅有最好的一组历史主结果，也有相对完整的训练分析和工程上下文。

### 5.2 Recall 对该任务非常关键

- 海洋垃圾检测任务中，漏检代价较高。
- 当前阶段最强结果的价值，不只是 F1 本身，更在于 Recall 的提升较明显。
- 这也是 `7ch` 主线被优先保留的重要原因。

### 5.3 更复杂输入并不必然带来更好效果

- `10ch` 相比 `7ch` 并没有自然取得整体优势。
- 说明这个项目已经进入“通道设计、损失函数、增强策略相互耦合”的阶段，而不是简单增加输入信息就能稳定提升。

### 5.4 数据增强策略会直接影响知识通道有效性

- 项目已有文档明确指出：`auto_augment=randaugment` 会破坏 knowledge heatmap 的空间先验。
- 这一发现是当前主线能够成立的重要工程经验之一。

## 6. 可直接用于汇报的图片与图像证据

以下图片均来自真实训练产物，适合直接插入汇报文档或 PPT。建议不要一次放太多，优先放 4 到 6 张最能讲故事的图。

### 图 1. 7ch 主线训练曲线

说明：用于展示历史主线的收敛过程和阶段性最优点。

![](../../experiments/runs/baseline_sota_aligned_1223_2219/results.png)

来源：

- `experiments/runs/baseline_sota_aligned_1223_2219/results.png`

### 图 2. 7ch 主线预测样例

说明：用于展示验证集上的真实检测效果。

![](../../experiments/runs/baseline_sota_aligned_1223_2219/val_batch0_pred.jpg)

来源：

- `experiments/runs/baseline_sota_aligned_1223_2219/val_batch0_pred.jpg`

### 图 3. 10ch + SIoU 训练曲线

说明：用于展示 `10ch` 路线中最优实验 `Exp-C` 的整体表现。

![](../../runs_marida/exp_c_10ch_siou3/results.png)

来源：

- `runs_marida/exp_c_10ch_siou3/results.png`

### 图 4. 10ch + SIoU 混淆矩阵

说明：用于展示 `10ch` 最优实验的类别识别情况。

![](../../runs_marida/exp_c_10ch_siou3/confusion_matrix.png)

来源：

- `runs_marida/exp_c_10ch_siou3/confusion_matrix.png`

### 图 5. 10ch + CIoU 基线训练曲线

说明：用于说明为什么仅使用 `10ch + CIoU` 时 Recall 严重不足。

![](../../runs_marida/exp_a_10ch_ciou_baseline/results.png)

来源：

- `runs_marida/exp_a_10ch_ciou_baseline/results.png`

### 图 6. 10ch + SIoU + Heatmap 训练曲线

说明：用于展示在 `10ch` 上继续加入 heatmap 后的性能回退现象。

![](../../runs_marida/exp_d_10ch_siou_heatmap/results.png)

来源：

- `runs_marida/exp_d_10ch_siou_heatmap/results.png`

## 7. 当前存在的问题与边界

为了保证汇报真实可靠，以下问题需要在文档中主动说明，而不是回避。

### 7.1 评估口径尚未完全统一

- 当前仓库中同时存在历史主线结果、`10ch` 消融结果和 `major classes` 后续结果。
- 这些结果并不完全处于同一评估口径下，因此不能简单放在一张表中宣称“统一最优”。

### 7.2 任务定义仍需在论文中进一步解释

- MARIDA 原始研究背景与当前 detection 主线之间仍需要更清晰的任务说明。
- 尤其需要解释为什么当前选择检测任务，以及 knowledge heatmap 为什么是合理的输入级先验，而不是信息泄漏。

### 7.3 外部泛化证据仍不足

- 当前主要结论仍建立在现有数据集及其派生实验之上。
- 尚缺少更强的跨场景、跨数据集或外部验证证据。

### 7.4 7ch 创新贡献仍需更严格隔离

- 目前虽然已有阶段性强结果，但如果要进入毕业论文或投稿阶段，仍建议补做更严格的 `6ch vs 7ch` 对照与多 seed 稳定性实验。

## 8. 建议向导师汇报时的表达方式

建议你在口头汇报时坚持以下三点：

- 第一，明确说“我已经形成一条可写论文的阶段性主线”，而不是说“我已经完成最终最优结果”。
- 第二，所有数字只报已经有日志和图像支撑的内容。
- 第三，主动说明当前短板，包括评估口径统一、外部验证和严格对照实验仍需补充。

一个较稳妥的口头总结可以是：

> 当前项目已经完成了基于多光谱输入和 knowledge heatmap 的阶段性主线构建，`7ch + knowledge heatmap + CIoU` 是目前最成熟、最适合论文叙事的方案；`10ch` 消融表明 SIoU 对 recall 有帮助，但整体尚未超过 7ch 主线。项目已经有真实结果、训练图像和失败分析，下一步重点不应是盲目换更复杂模型，而是统一评估协议、补严格对照和外部验证。

## 9. 下一步建议

如果这份汇总是为“阶段汇报”服务，下一步建议只写三项即可：

1. 统一主结果口径，整理正式结果表。
2. 补做严格的 `6ch vs 7ch` 对照以及稳定性检查。
3. 选择一个外部验证方向，用于支撑论文的泛化论证。

## 10. 附录：本报告使用的主要证据文件

- `docs/context.md`
- `docs/PREPROCESSING_RECONSTRUCTION.md`
- `docs/paper/PROJECT_ARCHITECT_PM_SUMMARY.md`
- `docs/paper/MASTERS_THESIS_PUBLICATION_ROADMAP.md`
- `experiments/runs/baseline_sota_aligned_1223_2219/results.csv`
- `experiments/runs/baseline_sota_aligned_1223_2219/results.png`
- `experiments/runs/baseline_sota_aligned_1223_2219/val_batch0_pred.jpg`
- `runs_marida/FINAL_ABLATION_REPORT.json`
- `runs_marida/README_ABLATION_STUDY.md`
- `runs_marida/exp_a_10ch_ciou_baseline/results.png`
- `runs_marida/exp_c_10ch_siou3/results.png`
- `runs_marida/exp_c_10ch_siou3/confusion_matrix.png`
- `runs_marida/exp_d_10ch_siou_heatmap/results.png`
- `README.md`
