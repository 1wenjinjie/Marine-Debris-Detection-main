# MARIDA v2 实验矩阵（导师任务对齐版）

**Version:** 2026-04-27
**目的:** 锁定每个实验的训练设置、seeds 策略、产出文件位置，形成单一事实来源（Single Source of Truth）。
**原则:** Seed=0 闸门策略 + 拆表防混淆 + 工程基线与结构消融分离。

---

## 闸门策略（Gate Strategy）

不一上来全量 3-seed。先做 seed=0 闸门验证，通过后再决定是否补 seed=1/2。

| 闸门类型 | 触发条件 | seed 投入 |
|---|---|---|
| **必跑 3-seed** | baseline、最终模型（Full）、工程优化基线（focal+1024） | 3 seeds |
| **闸门通过补 3-seed** | 单模块 +CBAM / +BiFPN / +EIoU 任一在 seed=0 显著优于 baseline | 3 seeds |
| **只跑 seed=0** | 闸门未通过的单模块（写入消融表，但不报告 std） | 1 seed |

闸门指标：mAP@0.5 提升 ≥ 0.01，或 Marine_Debris debris-sweep TP @ conf=0.025 提升 ≥ 1。

---

## 实验列表（导师任务核心）

### 表 1：结构消融（统一 imgsz=512、统一 BCE 损失，唯一变量是结构改动）

**所有行均 3-seed 评估**（路线 C 决策，2026-04-30）。导师任务要求消融表完整，所以即使每个变体不稳，也照实呈现 mean ± std，论文叙事走"消融研究表明结构改动均未稳定提升"。

| Run name | Model YAML | imgsz | Loss | Seeds 策略 | 状态 |
|---|---|---|---|---|---|
| `baseline_6ch_s{0,1,2}` | `yolov11n_v2_6ch.yaml` | 512 | BCE | 3 seeds | ✅ |
| `cbam_v2_s{0,1,2}` | `yolov11n_v2_6ch_cbam.yaml` | 512 | BCE | 3 seeds | ✅ |
| `bifpn_v2_s{0,1,2}` | `yolov11n_v2_6ch_bifpn.yaml` | 512 | BCE | 3 seeds | 🚀 s0 ✅ / s1 s2 待跑 |
| `eiou_v2_s{0,1,2}` | `yolov11n_v2_6ch.yaml` | 512 | BCE+EIoU | 3 seeds | 🚀 s0 ✅ / s1 s2 待跑 |
| `full_v2_s{0,1,2}` | `yolov11n_v2_6ch_full.yaml` | 512 | BCE+EIoU | 3 seeds | 🚀 s0 ✅ / s1 s2 待跑 |

### 关键发现（2026-04-30）：CBAM 不能作为最终模型

CBAM 3-seed 完成后暴露 seed-instability：

| Seed | mAP50 | M.Debris AP50 | TP@0.025 | Gate |
|---|---:|---:|---:|---:|
| s0 | 0.180 | 0.066 | 4 | ✅ |
| s1 | 0.192 | 0.024 | 1 | ❌ |
| s2 | 0.164 | 0.020 | 1 | ❌ |
| **3-seed** | **0.178 ± 0.014** | **0.037 ± 0.025** | **2.0 ± 1.7** | – |

Marine_Debris AP50 std (0.025) 大于 mean (0.037) → **统计不显著**优于 baseline。s0 的 4 TPs 是 outlier。

**结论：** seed=0 闸门策略漏掉了 seed-stability 维度。CBAM 不是稳定的改进。

### 论文叙事调整

- **表 1（结构消融）**：诚实呈现 negative result —— 所有结构改动（CBAM/BiFPN/EIoU/Full）在 3-seed 上均未稳定提升 baseline。这是有用的科学发现。
- **表 2（工程优化）**：保留正向贡献 —— `imgsz=1024 + focal(γ=2.0) + Platt 校准` 已 3-seed 稳定改善 Marine_Debris 实用检测。
- **论文最终模型** = `baseline + focal(γ=2.0) + imgsz=1024 + post-hoc Platt`（在表 2 中报告，已完成）。
- 表 1 的所有结构改动作为"已尝试但未成功"的消融研究存在，符合论文严谨性。

### 表 2：工程优化对比（与表 1 解耦，统一 baseline 结构、唯一变量是损失/分辨率/校准）

| Run name | Model YAML | imgsz | Loss | Calibration | Seeds 策略 | 状态 |
|---|---|---|---|---|---|---|
| `baseline_6ch_s{0,1,2}` | `yolov11n_v2_6ch.yaml` | 512 | BCE | – | 3 seeds（已完成） | ✅ |
| `imgsz1024_s{0,1,2}` | `yolov11n_v2_6ch.yaml` | 1024 | BCE | – | 3 seeds（已完成） | ✅ |
| `focal_g2.0_i1024_s{0,1,2}` | `yolov11n_v2_6ch.yaml` | 1024 | focal γ=2.0 | – | 3 seeds（已完成） | ✅ |
| `focal_g2.0_i1024_s{0,1,2}` + Platt | – | – | – | post-hoc Platt | 3 seeds（已完成） | ✅ |

### 拆表理由（重要）

**不要把表 1 和表 2 合并。** 否则审稿/导师会问："提升到底来自结构（CBAM/BiFPN/EIoU），还是来自 imgsz=1024 和 focal loss？"

- 表 1 回答："结构改动有没有用？"（在统一 512+BCE 的条件下）
- 表 2 回答："工程优化有没有用？"（在统一 baseline 结构的条件下）
- 若需要，再做表 3 = full × focal+1024 的组合（顶配模型），但只跑 1 seed 闸门后再决定。

---

## 扩展实验（已完成的工作，作为加分项）

| Run name | 用途 | 状态 |
|---|---|---|
| 6ch vs 3ch 消融（focal+1024） | 多光谱必要性证据 | ✅ |
| Platt scaling 校准（3 seeds） | 后处理增强 | ✅ |
| PKD（teacher-student） | 失败的尝试，写在 Discussion | ✅（不再续做） |
| Oversampling | 失败的尝试，写在 Discussion | ✅（不再续做） |

---

## 跨域 / 应用层（修正版）

### Roboflow Ocean Waste（**降级处理**）

**问题:** Roboflow Ocean Waste 是 RGB UAV 图像，不是 Sentinel-2 六波段。**不能**直接用 6ch best 模型 fine-tune。

**正确说法:** 使用 RGB 外部数据集进行**跨域验证**，而非 6ch 跨域微调。

**操作:**
1. 下载 Roboflow `khrissabdelaadim/ocean-waste-detection` 数据
2. 训练一个**单独的** 3ch RGB 模型在 Roboflow 上（与 v2 主线无关，独立分支）
3. 写为 "外部数据集泛化能力评估"，不与 6ch 主线混合

**优先级:** 中。可放到第 4 周以后。

### CMEMS 海流叠加（**修正版**）

**问题:** MARIDA 是稀疏 scene（不到 100 个 tile，跨多年），不是连续时间序列。**不能**写"污染相关性显著"。

**正确说法:** "环境因子描述性统计"。

**可做的:**
- 检测点经纬度反投影（图 9）
- 叠加邻近日期海流矢量场（图 11）
- 案例级漂移方向分析（≤ 5 个示例）

**不要写:**
- 风速/流速与垃圾密度的"相关性显著"统计
- 时间序列变化（图 10 改为"按月统计 detection 计数"，不要画趋势线）

### Sentinel-2 GeoTIFF 反投影前置条件

`data/processed/marida_v2/*.npy` 已无地理坐标。需写一个**索引脚本**：
- 输入：tile 文件名（如 `S2_20-4-18_30VWH_4`）
- 查找：原始 MARIDA `.tif` 的 transform / CRS
- 输出：tile 中心经纬度 + 像素分辨率

旧仓库脚本 `scripts/data/` 下若有 rasterio transform 痕迹可借鉴。**优先级 1 验证：原始 .tif 是否还在。**

---

## 训练超参数（统一）

所有结构消融实验共用 `MASK_RESCUE_TRAIN_DEFAULTS`（`scripts/training/mask_rescue_hyperparams.py`），仅以下字段允许各 run 单独覆盖：

| 允许覆盖字段 | 用途 |
|---|---|
| `imgsz` | 表 2 的工程优化（512 / 1024） |
| `batch` | imgsz=1024 时降到 4，避免显存溢出 |
| `device`, `seed`, `name`, `project`, `exist_ok` | run 元信息 |
| `workers` | DataLoader 稳定性（必要时设 0） |
| 损失函数 patch | 通过 trainer 子类（FocalLossTrainer / EIoUTrainer）实现 |

**禁止**直接修改 `mosaic / mixup / copy_paste / hsv_*` —— 这些已在 hyperparams 锁定，避免静默漂移。

---

## 评估流程（统一）

每个 run 训练完成后必须立即执行：

1. `scripts/v2_evaluate.py` 写入 `runs_v2/<run>/eval_test.json`
2. `scripts/v2_sweep_thresholds.py` 写入 `runs_v2/sweep_<run>/sweep_results.json`
3. FPS 测试（待补脚本 `scripts/v2_fps_benchmark.py`）写入 `runs_v2/<run>/fps_bench.json`

**不允许**跳过 sweep 直接报告 mAP——Marine_Debris 的实际能力需要 sweep 反映。

---

## 输出文件命名规范

```
runs_v2/<run_name>/
  weights/best.pt          # 最佳权重
  eval_test.json           # 标准 eval（mAP, P, R, AP50/95 per class）
  fps_bench.json           # FPS 测试（待补）
  results.csv              # 训练曲线
  results.png              # 训练曲线图
  BoxPR_curve.png          # PR 曲线
  BoxF1_curve.png          # F1 曲线
runs_v2/sweep_<run_name>/
  sweep_results.json       # debris 阈值 sweep
```

汇总文件：

```
runs_v2/baseline_6ch_summary.json          # 已有
runs_v2/focal_g2.0_i1024_summary.json      # 已有
runs_v2/structure_ablation_summary.json    # 待生成（表 1）
runs_v2/engineering_ablation_summary.json  # 待生成（表 2）
```

---

## 时间表（6 周）

| 周次 | 主任务 | 子任务 |
|---|---|---|
| W1 (04-27 ~ 05-03) | 实验基础 | 实验矩阵、CBAM smoke test、FPS 脚本、原始 .tif 索引验证 |
| W2 (05-04 ~ 05-10) | 三件套 seed=0 闸门 | CBAM s0、BiFPN s0、EIoU s0、Full s0 |
| W3 (05-11 ~ 05-17) | 最终模型 3-seed | Full s1, s2；闸门通过的单模块补 seed |
| W4 (05-18 ~ 05-24) | 空间分析数据 | 经纬度反投影、CMEMS 数据下载、检测点地图 |
| W5 (05-25 ~ 05-31) | 图表生产 | 图 1-8 主线模型图，图 9 / 11 空间图，4 张表 |
| W6 (06-01 ~ 06-07) | 整合与写作 | Grad-CAM（图 7）、图 10 时间统计（不画趋势线）、Roboflow 轻量验证、文档定稿 |

**风险预算:** 若 W2 三件套全部跑通无障碍，W4 提前 1 周。若 CBAM/BiFPN smoke test 卡住，FPS 脚本和 .tif 索引可以并行做，不浪费时间。

---

## 优先级与风险

### 高优先级（不做就翻车）
1. **CBAM/BiFPN/EIoU seed=0 闸门** —— 导师任务核心
2. **Full 模型 3-seed** —— 表 1 终点
3. **图 1（总体框架）+ 图 5（消融柱状图）+ 表 1** —— 答辩展示主体

### 中优先级（直接影响答辩观感）
4. **图 9（经纬度反投影热力图）** —— 遥感专业特色
5. **图 11（CMEMS 海流叠加，案例级）** —— 遥感专业特色
6. **FPS 测试** —— 表 1 必填列

### 低优先级（做不完不致命）
7. Grad-CAM（图 7）
8. 图 10（时间统计，不画趋势）
9. Roboflow 跨域（独立分支，单独训练 RGB 模型即可）

### 风险点
- ❗ Bi-FPN 实现复杂度高于 CBAM，预留半周
- ❗ 原始 MARIDA .tif 若已被清理，图 9/11 整体降级
- ❗ Full（CBAM+BiFPN+EIoU）若 seed=0 表现劣于 baseline，需快速诊断（很可能是模块组合冲突）

---

## 当前位置（2026-04-30）

- ✅ 工程基础层：MARIDA v2 + 6ch baseline + focal+1024 + Platt 校准 + 3ch 消融
- ✅ 核心方法层（s0 闸门）：CBAM/BiFPN/EIoU/Full 全部 s0 完成
- ✅ 核心方法层（CBAM 3-seed）：CBAM s0/s1/s2 完成 → 揭示 seed-instability，不再作为最终模型
- ⏳ 核心方法层（路线 C）：BiFPN/EIoU/Full 各补 s1/s2，让表 1 全部 3-seed
- ✅ 应用价值层基础设施：tile_geo_index.json（1381/1381，0 缺失）已就绪
- ✅ 应用价值层脚本：matplotlib 图 9 已生成；cartopy 版图 9 脚本已写好待环境就绪
- ✅ 工具链：FPS benchmark 脚本、表 1 汇总脚本、图 1 框架图已就绪
- ⏳ Roadmap：BiFPN/EIoU/Full 6 个新 seed 训练 → cartopy 安装 → 图 9 升级 → FPS 跑全 → 表 1 最终版

**下一步动作（本周）:**
1. ✅ 本文件落地（实验矩阵）
2. ✅ 新建 `configs/models/yolov11n_v2_6ch_cbam.yaml`（纯 CBAM 配置）
3. ✅ 新建 `scripts/v2_cbam_smoke_test.py`（验证可训练）
4. ✅ 运行 smoke test 闭环
5. ✅ 新建 `scripts/v2_cbam_train.py`（正式训练 + eval + debris sweep）
6. ✅ 启动并完成 `cbam_v2_s0` 完整训练（50 epoch）
7. ✅ 新建 `ultralytics_plugins/eiou_trainer.py`（训练时将 CIoU box loss patch 为 EIoU）
8. ✅ 新建 `scripts/v2_eiou_smoke_test.py`（1-epoch 闭环验证）
9. ✅ 新建 `scripts/v2_eiou_train.py`（正式训练 + eval + debris sweep）
10. ✅ 启动并完成 `eiou_v2_s0` 完整训练（50 epoch；gate 未通过，不补 seed）
11. ✅ 新建 `ultralytics_plugins/bifpn_modules.py`（WeightedConcat + parse patch）
12. ✅ 新建 `configs/models/yolov11n_v2_6ch_bifpn.yaml`（BiFPN-lite 配置）
13. ✅ 新建 `scripts/v2_bifpn_smoke_test.py` 与 `scripts/v2_bifpn_train.py`
14. ✅ 完成 BiFPN-lite YAML 解析检查（未训练）
15. ✅ 运行 `smoke_bifpn_v2` 闭环验证
16. ✅ 启动并完成 `bifpn_v2_s0` 完整训练（50 epoch；gate 未通过，不补 seed）
17. ✅ 新建 `configs/models/yolov11n_v2_6ch_full.yaml`（CBAM + BiFPN-lite）
18. ✅ 新建 `scripts/v2_full_smoke_test.py` 与 `scripts/v2_full_train.py`
19. ✅ 完成 Full YAML 解析检查（未训练）
20. ✅ 运行 `smoke_full_v2` 闭环验证
21. ✅ 修改 `scripts/v2_evaluate.py` 默认 `workers=0`，规避 Windows 多进程权限问题
22. 🚀 启动 `full_v2_s0` 完整训练（50 epoch，`--workers 0`）
