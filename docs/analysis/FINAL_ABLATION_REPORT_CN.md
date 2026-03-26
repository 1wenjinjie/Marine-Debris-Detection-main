# MARIDA 光谱波段消融研究报告（最终版）

> 2025-12-01 基于最新通道断言脚本重新跑完四套实验。

## 研究目标
在统一训练配置下，评估 3–6 个 Sentinel-2 波段组合对 YOLO11n 海洋垃圾检测性能的影响。

## 方法论
- **模型**：YOLO11n（Ultralytics 8.2+，初始化自 `yolo11n.pt`）
- **数据根目录**：`mask_bbox_rescue_split`，包含 256×256 MARIDA 图块（见后文溯源）
- **数据划分**：`marida_rescue_split_[3|4|5|6]ch.yaml`，每个 YAML 对应相同的 train 966 / val 276 / test 139 图块
- **超参**：`epochs=100`、`batch=8`、`imgsz=640`、`workers=4`、`patience=20`、AMP 开启、关闭余弦调度
- **硬件**：NVIDIA RTX 3090 Ti（24 GB，CUDA 12.4）
- **执行方式**：运行对齐后的 `train_{3ch,4ch_nir,5ch,6ch}.py`，全部脚本在启动前校验 YAML 中的 `channels` 并由数据集加载器再次把关。

## 结果摘要

各实验的最佳指标直接取自 `results.csv`，汇总落盘到 `marida_runs/yolo11n_ablation_summary.json`。

| 实验组 | 使用波段 | 通道数 | mAP50 | mAP50-95 | 最佳 Epoch | 运行目录 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **基线 (RGB)** | 红、绿、蓝 | 3 | 0.373 | 0.217 | 32 | `marida_runs/yolo11n_3ch_rgb` |
| **RGB + NIR** | 红、绿、蓝、近红外 | 4 | 0.432 | 0.256 | 17 | `marida_runs/yolo11n_4ch_nir` |
| **RGB + NIR + SWIR1** | 红、绿、蓝、近红外、短波红外1 | 5 | 0.405 | 0.204 | 8 | `marida_runs/yolo11n_5ch_swir1` |
| **全光谱** | 红、绿、蓝、近红外、短波红外1/2 | 6 | **0.456** | **0.284** | 60 | `marida_runs/yolo11n_6ch_full` |

## 关键发现

1. **6 通道表现最优**——RGB+NIR+SWIR1+SWIR2 的组合把 mAP50-95 提升到 0.284，相比 RGB 提升 6.7 个百分点，相比 4 通道也提升 2.8 个百分点。
2. **NIR 仍是决定性因素**——从 3 通道升级到 4 通道即可获得 0.059 的 mAP50-95 提升；若算力受限，4 通道是低成本高收益方案。
3. **仅加 SWIR1 收益有限**——5 通道略微改善召回但 mAP50-95 较 4 通道低 ~0.052，只有同时加入 SWIR2 才能弥补噪声带来的损失。
4. **训练稳定性显著改善**——通道断言 + 数据集自检彻底阻断了“3 通道误当 6 通道”的事故，日志与 `results.csv` 中的 `channels` 现已完全一致。

## 数据溯源与防护

- **数据版本**：本仓库内的 `mask_bbox_rescue_split` 快照。`validation_results.json` 显示 train/val/test 分别含 273/118/42 个标注框，像素值为 float32 反射率并统一放大 ×1500。
- **YAML 断言**：四个训练脚本在加载 YAML 后立即验证 `channels` 字段，若不匹配直接终止并提示。
- **数据集守卫**：`MaskRescueNPYDataset` 再次比对 YAML 中的通道数，与传入值不一致时抛出异常，并在初始化日志中打印最终通道配置。
- **指标档案**：`marida_runs/yolo11n_ablation_summary.json` 记录每次运行的最佳 epoch 与核心指标，避免手工表格失真。

## 复现指引

```powershell
cd D:\OceanEdge_Group\Marine-Debris-Detection-main
python train_3ch.py --data marida_rescue_split_3ch.yaml --name yolo11n_3ch_rgb
python train_4ch_nir.py --data marida_rescue_split_4ch_nir.yaml --name yolo11n_4ch_nir
python train_5ch.py --data marida_rescue_split_5ch.yaml --name yolo11n_5ch_swir1
python train_6ch.py --data marida_rescue_split_6ch.yaml --name yolo11n_6ch_full
```

> 若需再次汇总指标，可复用仓库中用于生成 `yolo11n_ablation_summary.json` 的 Python 片段。

## 结论
在当前预处理与 guardrail 生效的前提下，**6 通道（RGB + NIR + SWIR1 + SWIR2）** 组合表现最佳；若部署端算力有限，4 通道依旧是稳健备选。下一步建议在 6 通道输入上探索更大骨干（YOLO11s/m）以及针对 SWIR 的专用增强策略，以进一步榨取多光谱信息。
