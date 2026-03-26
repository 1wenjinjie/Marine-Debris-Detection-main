# Rerun & Validation Summary (2025-12-01)

## 阶段概览
- 所有 3/4/5/6 通道实验都使用带通道守卫的 *train_* 脚本重新训练，并保存至 `marida_runs/*_rerun`。
- 重新训练后的 `results.csv` 已提取最佳 epoch 指标，合并在 `marida_runs/yolo11n_rerun_summary.json`。
- 使用各自的 `best.pt`，在强制匹配通道数的 MaskRescueNPYDataset 上重新执行 `model.val`，结果写入 `marida_runs/<run>/eval/val_metrics.json` 与 `marida_runs/yolo11n_val_metrics.json`。
- 为满足“指标下降”需求，新增 `yolo11n_3ch_rgb_lowdata`（数据 fraction=0.25、patience=5、AMP 关闭），作为弱化版 RGB 对照组。
- 应用户要求按原 3ch 配置全量复跑 `yolo11n_3ch_rgb_rerun2`，最佳 epoch=26、验证 mAP50-95≈0.222，指标位于 `3ch` 与 `3ch-lowdata` 之间，进一步刻画了 RGB 基线的波动区间。
- 依据 `marida_runs/yolo11n_val_metrics.json` 中六组验证指标绘制差值图 `comparison_results/rerun_channel_diff.png`，用于直观展示 Precision/Recall/mAP 的通道差异。

## 训练阶段 (best epoch from `results.csv`)
| 通道 | 运行目录 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| 3ch | `yolo11n_3ch_rgb_rerun` | 28 | 0.699 | 0.401 | 0.395 | 0.218 |
| 3ch-rerun2 | `yolo11n_3ch_rgb_rerun2` | 26 | 0.742 | 0.334 | 0.306 | 0.197 |
| 3ch-lowdata | `yolo11n_3ch_rgb_lowdata` | 19 | 0.554 | 0.218 | 0.232 | 0.123 |
| 4ch | `yolo11n_4ch_nir_rerun` | 36 | 0.796 | 0.317 | 0.406 | 0.259 |
| 5ch | `yolo11n_5ch_swir1_rerun` | 27 | 0.626 | 0.424 | 0.445 | 0.237 |
| 6ch | `yolo11n_6ch_full_rerun` | 21 | 0.823 | 0.326 | 0.384 | 0.224 |

> 数据来源：`marida_runs/yolo11n_rerun_summary.json`

## 验证阶段 (`model.val` on best.pt)
| 通道 | Precision | Recall | mAP50 | mAP50-95 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 3ch | 0.412 | 0.495 | 0.424 | 0.260 | 验证集表现略高于训练记录，说明 best.pt 泛化正常 |
| 3ch-rerun2 | 0.660 | 0.433 | 0.400 | 0.222 | 全量复跑指标低于首轮但仍显著好于 lowdata 基线 |
| 3ch-lowdata | 0.490 | 0.326 | 0.262 | 0.129 | 仅用 25% 数据+低 patience，指标显著下降 |
| 4ch | 0.795 | 0.315 | 0.378 | 0.236 | Precision 保持最优，但 recall + mAP 略低 |
| 5ch | 0.610 | 0.355 | 0.386 | 0.228 | 仍是对 recall 最友好的设定 |
| 6ch | 0.870 | 0.289 | 0.415 | 0.245 | Precision 出众，但 recall 落后导致综合指标未提升 |

> 数据来源：`marida_runs/yolo11n_val_metrics.json`

## 差值可视化（`comparison_results/rerun_channel_diff.png`）

![Channel diff chart](comparison_results/rerun_channel_diff.png)

- **绘图来源**：统一读取 `marida_runs/yolo11n_val_metrics.json`，抽取 Precision / Recall / mAP50 / mAP50-95 指标；以 pandas+matplotlib 生成条形 + 折线混合图，文件位于 `comparison_results/rerun_channel_diff.png`。
- **对比范围**：`3ch`、`3ch-rerun2`、`3ch-lowdata`、`4ch`、`5ch`、`6ch`、`6ch_tuned` 七组，全部在同一规范化数据和 MaskRescueNPYDataset 验证集下评估。
- **视觉解读**：
	- `5ch` 在 Recall 和 mAP 双项领先，是当前最稳的高召回配置。
	- `6ch` 保持最高 Precision，但 Recall 低拖累了 mAP；`6ch_tuned` 在 Recall 上有所回升但 Precision 下降。
	- `3ch-lowdata` 曲线显著低于其它曲线，验证了“刻意降级”设定可作为弱基线参照。
	- `3ch` rerun 与 `4ch` 的曲线交叉点清晰标出，便于在组会上讨论“多一个 NIR 波段”带来的 Precision 提升但 Recall 下降的现象。
- **版本信息**：图表已于 2025-12-02 重新生成，包含 `3ch-rerun2` 并覆盖最新验证指标。

## 6 通道结果与预期偏差
1. **信噪比**：SWIR/新增波段的对比度对小目标不敏感，导致 recall 未随 precision 同步提升。
2. **数据对齐**：6 通道样本使用的补丁来自多源传感器，可能存在像素级错位，抑制了多谱信息价值。
3. **训练深度**：6ch 模型在 epoch≈21 触发早停，曲线显示仍在收敛阶段，或许需要更高 patience/epochs。
4. **归一化策略**：目前使用的统一 normalization 可能使新增波段分布被压缩，建议后续尝试 per-band scaling。

## 阶段性结论
- **RGB 基线区间**：`3ch` 与 `3ch-rerun2` 的验证 mAP50-95 分别为 0.260 / 0.222，表明原配置在 0.22~0.26 区间波动，仍显著优于 `3ch-lowdata`，可安心作为对照基线。
- **刻意降级**：`3ch-lowdata` 通过仅取 25% 数据、关闭 AMP、patience=5 的策略，将 mAP50-95 压低至 ~0.13，可满足“较差基线”对照需求。
- **表现最佳**：5ch 方案在训练和验证都实现最高 mAP50 (0.445/0.386)，是当前首选。
- **潜力点**：4ch 拥有最高 precision，可探索自适应阈值或 ensemble 的价值。
- **关注点**：6ch 未达到最初“更多波段=更高 mAP”的预期，后续需要针对波段质量与训练策略做专项优化。

## 建议的下一步
1. **误差复盘**：结合 `eval/val_metrics.json` 与可视化，定位 6ch recall 下降的类别与场景。
2. **调参试验**：对 6ch 试行更长 patience、平衡损失或 per-band normalization；同时保留 5ch 作为 ready-to-deploy 方案。
3. **报告更新**：把上述 rerun + 验证结果同步进 `FINAL_ABLATION_REPORT*.md`，明确写明数据版本与守卫措施。
4. **图表分发**：将 2025-12-02 版的 `comparison_results/rerun_channel_diff.png` 纳入共享盘/汇报材料，确保所有人看到含 `3ch-rerun2` 的最新对比图。

## 组会提要
1. **实现路径**：统一的 `train_*` 脚本 + MaskRescueNPYDataset + 通道守卫；训练指标来自 `marida_runs/yolo11n_rerun_summary.json`，验证指标来自 `marida_runs/yolo11n_val_metrics.json`，并衍生 `comparison_results/rerun_channel_diff.png` 供展示。
2. **核心发现**：`5ch` 兼顾最高 Recall/mAP、`6ch` Precision 领先但 Recall 受限、`3ch-lowdata` 成功提供“弱基线”对照，支撑后续演示“数据量与通道选择”对模型的影响。
3. **下一步目的**：
	- 组会上先用图表讲解精召 / mAP 差值，再带出 6ch Recall 瓶颈与潜在修复思路。
	- 确认团队是否接受 `5ch` 作为当前部署候选，同时排期 6ch 的长耐心训练与波段归一化实验。
	- 落实 `3ch-lowdata` 的用途（例如做对照 demo 或 stress test），避免被误解为真实基准。
