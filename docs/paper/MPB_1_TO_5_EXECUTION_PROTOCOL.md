# Marine Pollution Bulletin 投稿实验协议：建议 1-5

版本：`marida_mpb_v1`  
状态：数据审计、协议锁、盲态双评审闭环和五路径 `smoke_v10` 已通过工程验证；真实人工评审与 39 次正式多种子训练尚未开始。  
目标期刊：[Marine Pollution Bulletin](https://www.sciencedirect.com/journal/marine-pollution-bulletin)

## 结论先行

现有历史结果不能直接作为确认性投稿结果。旧流程反复使用测试集比较架构、输入尺寸和阈值，`Best F1` 与 `tau=0.25` 也来自测试集扫描。因此：

1. 旧结果仅保留为探索性结果或泄漏史料。
2. 新实验使用新的 `marida_mpb_v1` 协议，训练和阈值选择只访问 train/val。
3. MARIDA 已被历史工作反复查看，即使新协议重新划分，也只能称为回顾性内部验证，不能重新称为 untouched test。
4. 第 6 条独立海域验证仍是最终确认性证据，与 1-5 条并不冲突。

MPB 明确接受利用 AI 评估海洋环境的研究，但不鼓励只有模型开发、缺少海洋污染解释与质量控制的稿件。因此主文必须同时报告完整数据总体、误报警负担、场景级不确定性、适用边界和管理含义，而不能只报告 mAP 涨点。

## 统一研究边界

- 数据源：MARIDA 的 63 个 Sentinel-2 acquisition scenes、1,381 个 256 x 256 patches。
- 光谱输入：B4、B3、B2、B8、B11、B12。
- 产品表述：原始数据为 MARIDA 发布的 ACOLITE Rayleigh-corrected Sentinel-2 影像，正式稿不得笼统改写为 L2A surface reflectance。
- 检测类别：Marine_Debris、Algae_Organic、Ship。
- bbox 转换：4-connectivity，原始网格 bbox 外扩 2 pixels。
- 主 estimand：原始 mask 中面积 `>=4` pixels 的候选连通域实例。
- 置信度政策：保留 MARIDA 发布的全部标注置信度，同时逐 split 报告 high/moderate/low 像素构成；不把低置信度像素隐去后再称作确定背景。
- 统计单位：acquisition scene，而不是 patch、像素或 seed。
- 随机种子：`0, 1, 2`，所有比较使用配对 seeds。
- 主终点：Marine_Debris AP50；同时报告 TP/FP/FN、P/R/F1、每 100 patches 报警数、每 scene 误报警和 scene-cluster bootstrap 95% CI。

### 预注册主张层级

- 唯一 primary contrast：全量 scene-disjoint 数据上的 `NDWI7 - 6ch`。
- 必要机制证据：`NDWI7 - shuffled7`；只有该配对 scene-bootstrap 95% CI 也位于 0 以上，才允许解释为物理表示带来的增益。
- 2 x 2 泄漏归因与固定 `MIN=4` estimand 敏感性为 secondary effect-estimation analyses。
- matched MIN 只作 target-definition 描述，不参与最优 MIN 选择。
- primary 支持规则：`NDWI7 - 6ch` 的配对 scene-bootstrap 95% CI 须位于 0 以上，同时逐一报告 3 个配对 seed 效应；若多数 seed 方向相反，不作增益主张。
- 内部 MARIDA 结果不得单独支持管理可接受性或真实海域有效性主张；在外部场景与使用方成本未定义前，误报警负担只作描述。

## 1. 统一 2 x 2 泄漏归因

### 为什么不直接复用旧四格

旧数字来自不同数据版本、不同测试集、不同初始化和不同训练阶段；其中一个格还是“训练后推理置零”，不是独立重训。它们不能用于估计划分主效应、先验主效应或交互效应。

### 固定审计面板

2 x 2 只承担泄漏归因，不承担部署性能结论。它使用历史 v1 的 514-patch 目标富集总体，以便保持旧标签定义，但统一测试面板：

- scene-clean 原测试来源：73 patches、9 scenes。
- 1 个只有单 patch 的 scene 不具备 sibling exposure，只作 clean-only 描述。
- 其余 8 scenes 内，patch stem 按 SHA-256 排序形成固定 audit test 与 sibling 候选；该测试面板选择不查看标签。
- 每个 scene 的 70% patch 进入固定 audit test，共 51 patches。
- 剩余 21 patches 只用于 shared-scene sibling exposure，audit patch 本身绝不进入训练。
- clean 与 shared 都使用同一 70-patch、9-scene validation 和同一 51-patch、8-scene test。
- shared train 加入 21 个 sibling patches，同时用确定性 constrained beam matching 移除 8 个 clean-train scenes、恰好 21 patches。匹配只使用训练设计阶段可见的 7 类框计数/出现、MD 框尺度和输入通道统计，不使用任何模型结果。
- 两个 exposure 水平最终均为 371 train patches、40 train scenes，无精确 patch 重复。
- 四格统一使用 clean/shared 训练集精确交集中的 350 个 patches 计算前六通道 p01/p99；双方各自独有的 21 个训练 patches、validation 和 audit test 均不参与。两类 prior 固定到其天然 `[0,1]` 范围，避免归一化成为第三个因子。

四格为：

| Scene exposure | 第 7 通道 | 性质 |
|---|---|---|
| scene-disjoint | NDWI-inverse | 可部署审计格 |
| scene-disjoint | GT distance-transform | oracle 审计格 |
| shared-scene siblings | NDWI-inverse | scene exposure 审计格 |
| shared-scene siblings | GT distance-transform | 双重风险审计格 |

所有格使用同一 YOLO11n、`imgsz=512`、batch 8、50 epochs、AdamW、相同 7ch 初始化哈希和 seeds `0/1/2`，共 12 次训练。GT 格必须明确标注为 oracle，不能进入部署结论。

交互量定义为：

```text
I = (shared, GT - clean, GT) - (shared, NDWI - clean, NDWI)
```

匹配后 clean/shared 训练框数中，Marine_Debris 为 `655/644`，Ship 为 `62/61`，Foam 为 `41/39`，Wakes 为 `8/13`。受“整 scene 替换且恰好 8 scenes/21 patches”约束，监督组成不能完全相同。因此该实验正式名称为 **pre-specified conditional 2 x 2 leakage-attribution audit**；主效应和交互是该固定面板下的条件诊断量，不能写成普适因果效应。

局限：该面板来自目标富集 v1 子集，不代表真实海域 prevalence，禁止用它报告总体 false-alarm rate 或 operational precision。历史 v1 标签转换与新组件账本不完全同构，2 x 2 不报告 size-stratified recall。

## 2. MIN_PIXELS 敏感性

### 数据事实

在 4-connectivity、全部置信度的 Marine_Debris 原始组件中：

| MIN_PIXELS | 组件数 | 含组件 scenes | 含组件 patches |
|---:|---:|---:|---:|
| 1 | 1,425 | 55 | 373 |
| 2 | 814 | 52 | 266 |
| 4 | 212 | 31 | 80 |
| 8 | 40 | 9 | 21 |

`MIN=4` 相对 `MIN=1` 排除了 1,213/1,425，即 85.1% 的 Marine_Debris 连通域。因此 MIN 不是普通超参数，而是在改变研究总体。

### 主敏感性设计：固定 MIN=4 estimand

- 使用全量 1,381 patches 和同一 scene-disjoint split。
- 只改变 Marine_Debris 的训练标签阈值：`MIN={1,2,4,8}`。
- Algae_Organic 和 Ship 始终使用 `MIN=4`。
- connectivity、padding、置信度政策、输入、增强、优化器和采样方式完全固定。
- 四组模型都在同一 `MIN=4` validation 上选择 `best.pt` 和阈值，并在同一 `MIN=4` test 上比较。
- 每档运行 seeds `0/1/2`；`MIN=4` 直接复用六通道主基线，不重复训练。

该主分析公平回答：“训练时纳入或排除小组件，是否改变同一个 MIN=4 检测任务的表现？”固定 test 中四组均为 212 patches、10 scenes、32 个 MD 框。

### 补充敏感性设计：matched target definition

- 另建 `MIN={1,2,4,8}` 各自一致的 train/val/test 标签，并在各自 validation 上选择 checkpoint 和阈值。
- 四组使用同一批 212 test patches 和 10 scenes，但 MD 真值分别为 `206/117/32/7`。
- 该分析只展示“目标定义改变后的样本支持与绝对性能”，不得按 AP 大小选择所谓最佳 MIN。
- `MIN=8` test 仅 7 个 MD 实例，只作描述性结果并报告宽区间。
- matched `MIN=4` 仍复用六通道主基线，不重复训练。

组件账本已记录 component ID、scene、patch、原始类别、面积、bbox 和置信度像素构成。它用于完整说明数据筛选、负样本构成和 `MIN_PIXELS` 改变时的组件数，不把未经完成的人工复核写成研究证据。自动化组件审计覆盖全部数据；若后续有真实专家复核，只能作为补充材料，不能替代正式的敏感性实验。

## 3. 六通道与 NDWI 七通道

该实验使用全量数据，不使用 514-patch 全阳性子集。

| 模型 | 输入 | 目的 |
|---|---|---|
| C6 | B4/B3/B2/B8/B11/B12 | 六通道基线 |
| C7-physical | C6 + `(1-NDWI)/2` | 显式物理指数表示 |
| C7-shuffled | C6 + patch 内固定空间置乱的 NDWI | 控制额外通道和优化效应 |

公平性约束：

- 相同 1,381 patches、scene split、`MIN=4` 标签、负样本和增强。
- 相同 YOLO11n COCO checkpoint 扩展初始化。
- 6ch 和 7ch 的前六通道卷积权重逐元素一致，第七通道以 RGB filter mean 初始化。
- 初始化合同逐张量验证：前三通道与 COCO 源权重相同，新增通道为 RGB mean 经 Ultralytics FP16 序列化，6ch/7ch 前六通道逐元素相同，其余层与源模型逐元素相同。
- 每个初始化文件、数据 YAML、split manifest、输入数组、标签、统计文件和 checkpoint 都记录或聚合校验 SHA-256。
- 各通道 p01/p99 仅由对应训练 manifest 计算。
- C6 与 C7 前六通道的训练统计必须逐元素一致。
- C7-shuffled 必须独立重训，不能用推理置零、置乱或加噪替代。
- 每个模型运行 seeds `0/1/2`，共 9 次训练。

NDWI 完全由 B3/B8 决定，不增加新的传感器信息。因此若 C7-physical 有增益，只能解释为显式物理表示或 inductive bias，不能写成“新增光谱信息”。

## 4. 数据筛选与负样本

### 历史子集重建

- 原始：1,381 patches、63 scenes。
- 历史 v1：514 patches、58 scenes。
- 排除：867 patches。
- 其中 655 patches 没有任何旧七类目标像素。
- 另有 212 patches 含旧七类相关像素，但未通过旧类别特异面积、形态学或覆盖率筛选。
- v1 的 514 个标签文件全部非空，因此它是目标富集子集，不是 MARIDA 的随机抽样。

旧 v1 使用的面积阈值实际为 3/8/8/6/18/3/64 pixels，并混合 dilation、closing、8-connectivity 和覆盖率阈值。它不能在方法中简写为统一 `MIN_PIXELS=4`。

### 新 scene-disjoint 全量划分

| Split | Patches | Scenes | 空检测标签 (`MIN=4`) | MD instances |
|---|---:|---:|---:|---:|
| train | 965 | 44 | 701 | 145 |
| val | 204 | 9 | 152 | 35 |
| test | 212 | 10 | 161 | 32 |

scene ID 交集为 0。随机 patch 审计对照保持相同 965/204/212 样本量，但 train-val、train-test、val-test 分别共享 50、53、43 个 scenes。

`MIN=4` scene-disjoint 数据的互斥 patch 构成为：

| 类别 | Train | Val | Test |
|---|---:|---:|---:|
| Marine_Debris positive | 57 | 11 | 12 |
| Subthreshold Marine_Debris | 200 | 48 | 45 |
| Algae/ship hard confounder | 193 | 38 | 37 |
| Foam/waves/wakes visual confounder | 67 | 17 | 14 |
| Environmental water background | 393 | 71 | 92 |
| Cloud or shadow | 43 | 17 | 10 |
| Unannotated or other | 12 | 2 | 2 |

这些类别是报告层级，不是把所有 patch 认证为真实负样本。正式术语使用 `no retained target boxes`，不使用 `pure background`。MARIDA 的 0 值包含未标注含义，含亚阈值 Marine_Debris 的 patch 也不得称为 verified negative。

val/test 保留所有无框 patch，不进行负样本下采样。正式结果需报告：

- FP/100 all patches；
- FP/100 label-negative patches；
- FP/scene；
- 每 scene 报警数；
- 主要混淆类别上的 FP 构成；
- 完整场景或原始 prevalence 条件下的失败案例。

## 5. 模型、阈值与测试隔离

### 历史结论

不能确认旧模型和阈值未使用测试集；代码证据表明它们确实使用了测试集：

- v2 训练脚本训练后自动评估 test；
- 是否扩展 CBAM/BiFPN/EIoU/Full seeds 曾由 test 指标门控；
- v1 `Best F1` 在 test confidence grid 上取最大值；
- BiFPN@1024 与 `tau=0.25` 是查看 test 后选出的。

所以旧 `F1=0.604 @ tau=0.25`、旧 Best F1、旧 test-best FDI/NDWI 阈值和“最终最优模型”均降级为探索性上限。

### 新运行器的强制流程

```powershell
# 1. 查看预注册运行矩阵
python scripts/mpb/run_experiments.py --stage status --suite all

# 2. 仅训练并访问 validation
python scripts/mpb/run_experiments.py --stage train --suite all --seed 0 1 2

# 3. 所有 run 完成后冻结 checkpoint 与 validation 阈值，并记录打印的 SHA-256
python scripts/mpb/run_experiments.py --stage lock --suite all --seed 0 1 2

# 4. 人工复核 decision_lock 后，输入上一步打印的完整 SHA-256，才允许一次内部 test
python scripts/mpb/run_experiments.py --stage test --suite all --seed 0 1 2 --confirm-final-test --confirm-decision-lock-sha256 LOCK_SHA256_FROM_STAGE_4

# 仅当同一 decision lock 的 test 被硬件/进程中断时，显式续跑未完成单元
python scripts/mpb/run_experiments.py --stage test --suite all --seed 0 1 2 --confirm-final-test --confirm-decision-lock-sha256 LOCK_SHA256_FROM_STAGE_4 --resume-final-test

# 5. 仅从已保存预测账本重算配对统计，不再次推理 test
python scripts/mpb/run_experiments.py --stage summarize --suite all
```

阈值只在 validation 上按严格一对一 IoU `>=0.5` 匹配选择，候选网格在查看正式 validation 结果前冻结为 `0.001-0.99`：

- balanced：最大 F1，平局选择报警更少的阈值；
- high recall：在 `R>=0.80` 的候选中最大化 precision；
- high precision：在 `P>=0.80` 的候选中最大化 recall；
- 若约束不可达到，记录 `no feasible operating point`，不放宽条件追求漂亮数字。

训练清单记录完整数据内容指纹，并锁定实际执行的本地实现文件，包括包初始化副作用与模块注册文件。协议同时保存确定性源码/配置 ZIP、逐文件 SHA-256 清单及完整 Python 包、CUDA、cuDNN 和 GPU 环境清单，因此即使工作区为 dirty，也能恢复本次相关执行版本。validation 明确使用 `rect=False` 的固定 512 x 512 几何；真实 6ch/7ch/oracle patch 与独立 operating-point 预处理已逐元素核对为完全一致。训练结束立即保存 val prediction ledger，并在完成标记中锁定 checkpoint、`eval_val`、阈值文件和 ledger 的 hash。decision lock 会核对预注册超参数、正式 run 身份与代码合同，确认 ledger stems 与 val manifest 完全一致，再从 ledger 重算整条阈值曲线；禁止验证后替换 `best.pt` 或修改验证证据。

test 阶段要求人工输入完整 decision-lock SHA-256，随后先完成全部只读预检，再写入全局 one-shot marker。预检重新核对正式 run 路径与 manifest、YAML、全部输入数组、标签、train-only stats、初始化合同、checkpoint 和 validation 证据，并从 val ledger 再次重算整条阈值曲线，逐项比较 balanced/high-recall/high-precision 阈值。中断后只能显式在同一 SHA 和同一 decision lock 下续跑 marker 中尚未完成的单元，已完成单元不得再次推理。全部完成后生成全局 test output manifest，锁定逐 run 结果、预测账本、总结果和配对统计。

test 完成后保存逐 patch 锁定预测账本，并从账本计算配对 scene-cluster bootstrap：2 x 2 的 scene-exposure 主效应、oracle-prior 主效应与交互，以及 NDWI7-vs-6ch、shuffled7-vs-6ch 和固定-MIN 差值。`summarize` 阶段可重复计算统计量，但不会再次运行 test 推理；重算前必须验证当前代码合同、output manifest 的文件集合、合同哈希和每个 result/ledger 哈希，重算结果也必须与冻结 summary 哈希完全一致。

### 预注册计算量

| 实验 | Cells | Seeds | Runs |
|---|---:|---:|---:|
| v1 固定面板 2 x 2 | 4 | 3 | 12 |
| 全量 6ch/NDWI-7ch/shuffled-7ch | 3 | 3 | 9 |
| MIN train 1/2/8，固定 eval MIN=4（MIN4 复用 6ch） | 3 additional | 3 | 9 additional |
| matched MIN 1/2/8（MIN4 复用 6ch） | 3 additional | 3 | 9 additional |
| 合计 | 13 unique | 3 | 39 |

## MPB 投稿前仍需完成

1. 完成 39 次正式训练，并独立复核日志、hash、validation ledger 和 decision lock。
2. test 只在 decision lock 后执行一个可审计的测试阶段；硬件中断仅允许在同一 lock 下续跑未完成单元，结果明确写成 retrospective internal validation。
3. 使用独立海域/新增 Sentinel-2 acquisition scenes 完成最终确认；目标域标签不能用于模型或阈值选择。
4. 外部场景最好有 UAV、甚高分影像或现场证据；证据不足时表述为 marine debris/floating matter 或 suspected plastic，不能直接宣称 confirmed plastic。
5. 提供 Data Availability Statement、split manifest、组件账本、环境、配置、权重、代码 commit/归档版本和数据集正式引用。

相关一手先例包括 [MARIDA](https://doi.org/10.1371/journal.pone.0262247)、[MADOS](https://doi.org/10.1016/j.isprsjprs.2024.02.017) 和完整场景误报警研究 [iScience 2023](https://doi.org/10.1016/j.isci.2023.108402)。

## 已生成产物

- 数据审计：`data/processed/marida_mpb_v1/audit/data_audit.json`
- patch 账本：`data/processed/marida_mpb_v1/audit/patch_inventory.csv`
- 组件账本：`data/processed/marida_mpb_v1/audit/marine_debris_components.csv`
- 固定 2 x 2 面板：`data/processed/marida_mpb_v1/audit/v1_factorial_panel.json`
- 协议锁：`data/processed/marida_mpb_v1/protocol_lock.json`
- 可恢复源码与环境：`data/processed/marida_mpb_v1/execution_source_archive.zip`、`execution_source_manifest.json`、`execution_environment.json`
- 数据配置：`configs/datasets/mpb/`
- 数据准备脚本：`scripts/mpb/prepare_protocol.py`
- 数据内容指纹：`scripts/mpb/protocol_integrity.py`
- 分阶段运行器：`scripts/mpb/run_experiments.py`
- 已通过五路径 smoke：`runs_mpb/smoke/*__smoke_v12/`（6ch、full NDWI7、v1 shared-oracle 7-class、matched MIN1、matched MIN8）；这些仅证明执行链可运行，不作为性能结果。
