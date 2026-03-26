# 当前 PKD 实验数据集诊断与修正建议

本文档面向当前 `PKD` 实验主线，目的是回答三个问题：

1. 当前实验实际使用的数据集到底是什么。
2. 这套数据是否足以支撑当前结论。
3. 哪些数据问题正在影响 `Precision`、泛化性和论文表述。

本文档只讨论当前检测任务使用的数据子集，不等同于原始完整 `MARIDA` 基准。

## 1. 当前实验实际使用的数据集

当前主实验的数据配置文件为 [marida_rescue_7ch.yaml](d:/OceanEdge_Group/Marine-Debris-Detection-main/configs/datasets/marida_rescue_7ch.yaml)，其 `path` 指向：

`data/processed/mask_bbox_rescue_split`

这意味着当前 `P0/P1/P2` 的检测实验使用的是一个经过重构的 `YOLO bbox` 检测子集，而不是原始论文中完整的像素级 `MARIDA` 数据。

需要明确区分两套概念：

- 原始 `MARIDA`：像素级标注数据，论文描述为 `1381` patches、`63` scenes，并强调同一 `scene/date` 应保留在同一个 split 中。
- 当前检测子集：`347/114/53` 张原始 `NPY` patch，对应 `train/val/test` 三个集合，面向 `7` 类检测任务。

## 2. 原始数据规模与当前检测子集规模

基于当前目录 [data/processed/mask_bbox_rescue_split](d:/OceanEdge_Group/Marine-Debris-Detection-main/data/processed/mask_bbox_rescue_split) 的真实 `images/labels` 重新统计，得到如下结果：

| Split | Images | Objects | Scenes | Objects/Image |
| --- | ---: | ---: | ---: | ---: |
| Train | 347 | 767 | 55 | 2.21 |
| Val | 114 | 235 | 46 | 2.06 |
| Test | 53 | 119 | 26 | 2.25 |
| Total | 514 | 1121 | 58 unique scenes | 2.18 |

这说明当前检测数据并不大。即使不考虑类别稀疏，只看原始检测样本量，它也更接近“小样本遥感检测实验集”，而不是一个可以稳定支撑复杂 `7` 类检测的中大规模 benchmark。

## 3. 当前类别分布的真实情况

重新扫描当前 `labels/*.txt` 后，得到如下总类别分布：

| Class | Objects | Percent |
| --- | ---: | ---: |
| Marine_Debris | 941 | 83.94% |
| Dense_Sargassum | 16 | 1.43% |
| Sparse_Sargassum | 9 | 0.80% |
| Natural_Organic | 0 | 0.00% |
| Ship | 84 | 7.49% |
| Foam | 55 | 4.91% |
| Wakes | 16 | 1.43% |

结论非常明确：

- `Marine_Debris` 是绝对主导类。
- `Dense_Sargassum`、`Sparse_Sargassum`、`Wakes` 都属于极低频类。
- `Natural_Organic` 在当前检测子集中为零样本类，无法训练，也无法评估。

换句话说，虽然配置文件声明了 `7` 类，但当前数据本身并不真正支持“完整 7 类检测”。

## 4. 测试集是否支持 7 类评估

当前测试集类别统计如下：

| Class | Test Objects |
| --- | ---: |
| Marine_Debris | 103 |
| Dense_Sargassum | 0 |
| Sparse_Sargassum | 0 |
| Natural_Organic | 0 |
| Ship | 11 |
| Foam | 4 |
| Wakes | 1 |

这意味着：

- 测试集根本没有 `Dense_Sargassum`、`Sparse_Sargassum` 和 `Natural_Organic`。
- `Wakes` 在测试集中只有 `1` 个框。
- 当前实验的 `mAP` 实际上主要由 `Marine_Debris`、`Ship` 和极少量 `Foam/Wakes` 决定。

因此，当前结果更准确的表述应是：

`PKD` 在一个严重长尾、测试集类别缺失的 `MARIDA` 检测子集上验证了方法可行性。

而不是：

`PKD` 在完整、平衡、可稳定评估的 `7` 类检测基准上得到结论。

## 5. 没有纯背景负样本

当前 `514` 张原始图像全部对应非空标签文件，也就是：

- `labeled_images = 514`
- `empty_or_missing_labels = 0`

这会带来两个直接后果：

1. 模型几乎没有机会学习“整张图没有目标时应输出空结果”。
2. 任何促进“更积极报框”的机制，都更容易提升 `Recall`，同时拉低 `Precision`。

这与当前实验现象高度一致：

- `Response KD` 显著提升 `Recall`
- 但 `Precision` 下降明显

因此，`Precision` 下滑不应只解释为蒸馏策略本身有问题，也要看到数据集没有提供足够的“背景抑制训练”。

## 6. Scene-level split 结构存在明显重叠

按当前文件名中的 `scene` 前缀统计：

| Overlap | Count |
| --- | ---: |
| Train ∩ Val | 43 |
| Train ∩ Test | 26 |
| Val ∩ Test | 25 |

进一步看比例：

- `93.48%` 的 `val scenes` 也出现在 `train`
- `100%` 的 `test scenes` 也出现在 `train`

这说明当前 split 不是严格意义上的跨场景划分，而更接近：

- 同一场景下不同 patch 的再分配
- 或“同场景泛化”而不是“跨场景泛化”

这不会让当前实验失效，但会影响论文结论边界：

- 可以支持方法在当前数据协议下的相对比较
- 不足以强支撑真实部署场景下的跨场景泛化能力

## 7. Manifest 与增强样本存在路径失配

检查 [train.txt](d:/OceanEdge_Group/Marine-Debris-Detection-main/data/processed/mask_bbox_rescue_split/train.txt) 后发现：

| Item | Value |
| --- | ---: |
| Manifest lines | 1216 |
| Resolved entries | 347 |
| Missing entries | 869 |
| `images_aug` entries | 869 |

这说明 `train.txt` 中包含大量增强样本路径，但这些路径在当前 `data/processed/...` 下并不能被解析到真实文件。

含义如下：

- 名义上看，训练集像是在使用“原图 + 大量增强样本”。
- 实际上，当前 manifest 可正确解析的只有 `347` 条原始训练图像。

因此，如果训练过程确实依赖这个 manifest，那么当前实验很可能主要训练在原始样本上，而不是你以为的更大训练集上。

这会进一步放大：

- 小样本问题
- 长尾问题
- seed 波动问题

## 8. 仓库中已有统计文件与当前数据不一致

[dataset_stats_summary.json](d:/OceanEdge_Group/Marine-Debris-Detection-main/results/metrics/dataset_stats_summary.json) 与 [dataset_stats.json](d:/OceanEdge_Group/Marine-Debris-Detection-main/results/metrics/dataset_stats.json) 不能直接作为当前检测子集的依据，原因是对应脚本 [summarize_dataset.py](d:/OceanEdge_Group/Marine-Debris-Detection-main/scripts/utils/summarize_dataset.py) 和 [tmp_stats.py](d:/OceanEdge_Group/Marine-Debris-Detection-main/scripts/utils/tmp_stats.py) 实际指向的是旧目录：

`MARIDA_YOLO_Enriched_Dilated_split`

因此，仓库里存在两套统计口径：

- 一套来自旧数据版本
- 一套来自当前 `mask_bbox_rescue_split`

论文写作时必须明确采用后者，否则会出现“样本数、框尺度、类别分布”互相打架的问题。

## 9. 这是否意味着数据集不适合做 PKD

不意味着“不适合”，但意味着“适合的结论边界更窄”。

当前数据适合支撑的内容：

- `PKD` 是否可行
- `RKD/FKD/RFKD` 谁更有效
- `Recall` 与 `Precision` 的机制差异
- 特权空间先验是否能迁移到纯 `6ch student`

当前数据不适合强支撑的内容：

- 完整 `7` 类检测 benchmark 结论
- 稳健的跨场景泛化结论
- 强调绝对数值优势而忽略数据局限的部署结论

因此，对当前阶段最准确的评价是：

这套数据足以完成 `PKD` 的最小可行验证与消融研究，但不足以直接作为高置信、完整 `7` 类检测论文主 benchmark。

## 10. 对当前实验结果的解释

这套数据会系统性地影响当前指标：

### 10.1 为什么 `RKD` 更容易拉升 Recall

- 当前图像全部为正样本 patch
- 背景抑制训练不足
- `Marine_Debris` 占绝对多数

因此 `Response-level KD` 所传递的“更积极检出”倾向，会在该数据上被放大，从而明显抬升 `Recall`。

### 10.2 为什么 Precision 会掉

- 没有足够的纯背景负样本
- 测试集类别覆盖有限
- 同场景 split 使 student 更容易学到场景模式，而不是更强的空间判别边界

因此 `Precision` 的下降不能简单理解成 `RKD` 不好，而更像是数据协议对“多报框”策略缺少约束。

### 10.3 为什么 FKD/RFKD 不稳定

- 数据总量小
- 长尾严重
- scene overlap 高
- 类别本身存在强谱间混淆和标注不确定性

在这种条件下，feature-level 对齐更容易把 student 拉向 teacher 的偏置，而不一定稳定提升最终检测行为。

## 11. 论文中建议采用的保守表述

建议将当前数据描述为：

`a reconstructed detection subset derived from MARIDA, used for validating PKD under limited-data, long-tail, and partially scene-overlapping conditions.`

中文可写为：

“本文实验基于从 MARIDA 重构得到的检测子集开展。该子集具有样本量有限、类别极度不平衡、测试集部分类别缺失以及 scene-level overlap 较高等特点，因此本文结果主要用于验证 PKD 方法的可行性与相对有效性，而非宣称完整 7 类检测基准上的最终性能上界。”

## 12. 类别重构建议

### 12.1 4 类方案

推荐的 `4-class` 研究型重构如下：

| New Class | Source Classes | Total Objects | Test Objects |
| --- | --- | ---: | ---: |
| Marine_Debris | Marine_Debris | 941 | 103 |
| Sargassum | Dense + Sparse | 25 | 0 |
| Ship | Ship | 84 | 11 |
| Foam_Wake | Foam + Wakes | 71 | 5 |

优点：

- 去掉零样本类 `Natural_Organic`
- 减少类别碎片化
- 保留一定语义解释性

缺点：

- 当前 test 中 `Sargassum` 仍为 `0`
- 不重做 split 的话，`4-class` 仍然不能形成完整可评估 benchmark

### 12.2 3 类方案

推荐的 `3-class` 应用型重构如下：

| New Class | Source Classes | Total Objects | Test Objects |
| --- | --- | ---: | ---: |
| Marine_Debris | Marine_Debris | 941 | 103 |
| Floating_Others | Dense + Sparse + Foam (+ Natural if future) | 80 | 4 |
| Ship_Wake | Ship + Wakes | 100 | 12 |

优点：

- 更接近当前数据真实可学到的主语义结构
- 更适合作为检测应用原型

缺点：

- 当前 test 中 `Floating_Others` 仍然过少
- 只做类别合并而不重做 split，统计稳定性仍然不足

### 12.3 结论

仅靠类别合并不能彻底解决问题。

真正需要的是：

1. 去掉零样本类。
2. 重新做 `scene-level split`。
3. 保证每个保留类在 `train/val/test` 都有最小样本覆盖。

## 13. Scene-level split 修正建议

推荐按照文件名前缀中的 `scene/date` 进行分组，即：

`S2_日期_瓦片号`

同一 `scene` 下的所有 patch 必须进入同一个 split。

### 13.1 修正原则

1. 不允许同一 `scene` 同时出现在 `train/val/test`。
2. 每个保留类至少满足：
   - `train >= 10` objects
   - `val >= 3` objects
   - `test >= 3` objects
3. 额外加入一批纯背景负样本 patch。
4. 所有 manifest 统一改为相对路径，并只引用当前可解析的真实文件。

### 13.2 建议优先级

`P0`

- 修复 `train.txt` 中坏掉的增强路径
- 统一统计口径，只以当前 `data/processed/mask_bbox_rescue_split` 为准

`P1`

- 删除零样本类 `Natural_Organic`
- 先尝试 `4-class` 重构，并检查 test 覆盖是否足够

`P2`

- 若 `4-class` 仍然缺类，则重做 split
- 目标是让每个保留类在 `test` 至少有少量对象

`P3`

- 补纯背景 patch
- 再重新验证 `Precision` 与 `Recall` 的平衡是否改善

## 14. 对当前论文主线的建议

如果当前时间和算力有限，建议这样推进：

1. 先保留当前 `RKD` 结果，完成 `3-seed` 稳定性验证。
2. 在论文中明确数据局限，不夸大“完整 7 类”结论。
3. 将当前数据问题写成 limitation，而不是回避。
4. 若后续还有时间，再做一次“小规模但严格”的 `scene-level` 重构版本复验。

这会比继续在当前数据上反复扫更多蒸馏超参数，更能增强论文可信度。

## 15. 可复现审计工具

为避免后续再次混淆统计口径，新增了审计脚本：

[audit_mask_rescue_dataset.py](d:/OceanEdge_Group/Marine-Debris-Detection-main/scripts/utils/audit_mask_rescue_dataset.py)

用法：

```bash
python scripts/utils/audit_mask_rescue_dataset.py
```

或：

```bash
python scripts/utils/audit_mask_rescue_dataset.py --output runs_phase2/current_dataset_audit.json
```

该脚本会统一输出：

- 每个 split 的图像数、目标数、scene 数
- 当前真实类别分布
- scene overlap 统计
- manifest 可解析性与坏路径数量

后续所有论文图表与汇报中的数据集数字，建议都从该脚本输出中提取。
