# PREPROCESSING 重建说明（preprocess_generate_knowledge_channel.py 重建）

说明：本文件记录对缺失脚本 `preprocess_generate_knowledge_channel.py` 的重建过程、算法细节、验证方法与如何在本仓库中运行重建脚本。

## 概要

原始仓库缺失 `preprocess_generate_knowledge_channel.py`。该脚本用于将 MARIDA 数据集中每个 patch 的 GT segmentation mask 转换为一条连续的知识蒸馏热图（第7通道），并将其与原来的 6 通道多光谱数组拼接保存为 7ch `.npy` 文件。本文件说明了重建实现、关键参数与复现步骤，并提供对审稿人的说明（此脚本为重建实现而非原始作者提交的源脚本）。

## 算法 (简要)

1. 对每个 patch，从 `MARIDA/patches/{scene}/{patch}_cl.tif` 加载 GT mask（单通道，整形标签）。
2. 生成二值掩码：mask > 0 表示物体像素。
3. 对二值掩码执行欧氏距离变换：`cv2.distanceTransform(binary, cv2.DIST_L2, 5)`，得到以最近边界距离为值的浮点图。
4. 将距离图按最大值归一化到 [0, 1]，作为 heatmap（若 mask 全零则返回 0 图）。
5. 将 heatmap 作为第7通道拼接到源 6 通道数组，保存为目标路径。
6. 为每个样本计算统计量：knowledge_mean、knowledge_max、knowledge_coverage（>0.01），并基于这些统计量计算 `confidence_weight = alpha + scale * (mean + coverage)/2`，默认 alpha=0.4，scale=0.1。

## 运行与参数

脚本路径：`scripts/preprocessing/preprocess_generate_knowledge_channel.py`

默认用法：

```powershell
python scripts\preprocessing\preprocess_generate_knowledge_channel.py \
  --source-dataset MARIDA_YOLO_6band_split \
  --target-dataset mask_bbox_rescue_split \
  --mask-root MARIDA/patches \
  --splits train val test \
  --summary mask_bbox_rescue_split/knowledge_generation_summary.json
```

关键参数：
- `--skip-existing` ：跳过已存在的目标 `.npy` 文件
- `--conf-alpha` / `--conf-scale` ：置信度权重计算系数（默认 0.4 / 0.1）

## 验证方法

1. 检查生成的 `.npy` 文件维度为 HW7，dtype 为 float32。
2. 使用 `scripts/visualize_knowledge_overlay.py` 可视化第7通道叠加到 RGB，确认热图在物体中心呈高值，边界处递减。
3. 比较生成的 `knowledge_generation_summary.json` 中统计值（平均值约 ~0.02、覆盖率 ~0.03）与现有元数据一致性。

## 论文中如何描述

推荐在方法中写明：第7通道为基于 GT 掩码的距离变换热图，作为离线预处理生成并拼接到多光谱输入中；若脚本缺失，则将此实现作为补充材料提供并说明这是根据现有数据与统计量逆向重建的实现。

## 文件与位置

- 脚本：`scripts/preprocessing/preprocess_generate_knowledge_channel.py`
- 摘要：`mask_bbox_rescue_split/knowledge_generation_summary.json`

## 免责声明

此脚本为重建实现，依据仓库中已有的 `.npy` 样本分布与 `knowledge_generation_summary.json` 的统计量复现原始步骤。如需严格复现，请使用原始作者提供的脚本或核对原始提交历史（若可用）。

---

作者注：若需要我将此文件翻译为英文版或按照期刊要求格式化（补充材料、脚注、代码链接），我可以继续处理。
