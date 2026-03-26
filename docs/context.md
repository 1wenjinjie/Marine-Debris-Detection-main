# Project Context - MARIDA 多光谱海洋碎片检测

##  项目目标
- **核心目标**: 发表Remote Sensing/IEEE TGRS论文
- **技术目标**: 
  -  7ch基线超越SOTA (目标F10.675, 已达成F1=0.6774)
  -  10ch光谱增强 (期望F1+2~5%)
  -  EIoU Loss + Bi-FPN实现

##  重大突破 (2025-12-23)

###  成功超越SOTA基线
**训练目录**: experiments/runs/baseline_sota_aligned_1223_2219

| 指标 | SOTA基线 | 本次训练 | 提升 | 状态 |
|------|---------|---------|------|------|
| **F1** | 0.6753 | **0.6774** | +0.3% |  超越 |
| **Recall** | 0.6875 | **0.7351** | +6.9% |  显著提升 |
| **Precision** | 0.6635 | 0.6281 | -5.3% |  略低 |
| **mAP@50** | 0.6345 | **0.6673** | +5.2% |  提升 |
| **最佳Epoch** | 30 | 38 | - | - |

**最佳权重**: experiments/runs/baseline_sota_aligned_1223_2219/weights/best.pt

###  成功关键因素

#### 1. 发现致命问题: auto_augment=randaugment
**问题**: 随机增强破坏了第7通道知识热力图的空间先验
- 知识热力图生成流程: GT Mask  形态学处理  距离变换  [0,1]归一化
- randaugment会破坏这个精心设计的空间一致性
- **关闭后Recall暴涨45.5%** (0.5054  0.7351)

#### 2. 训练策略对齐SOTA
| 参数 | 失败配置 | 成功配置 | 影响 |
|------|---------|---------|------|
| auto_augment | randaugment | **none** | Recall+45.5% |
| patience | 10 | **50** | 避免Epoch 25过早停止 |
| lrf | 0.01 | **0.1** | 温和学习率衰减 |
| translate | 0.1 | **0.05** | 保护小目标空间特征 |
| scale | 0.5 | **0.2** | 减少尺度扰动 |
| erasing | 0.4 | **0.2** | 减少随机擦除 |

#### 3. 训练轨迹分析
- Epoch 1-10: 快速学习 (F1: 0.003  0.496)
- Epoch 11-30: 稳定提升 (F1: 0.420  0.530)
- **Epoch 38: 最佳性能** (F1=0.6774, P=0.628, R=0.735)
- Epoch 39-50: 性能波动但保持高位

##  技术路线回顾

### 阶段1: 问题诊断 (12月23日 15:00-21:00)
`mermaid
flowchart LR
    A[首次训练<br/>F1=0.5701] --> B{性能差距<br/>vs SOTA -15.6%}
    B --> C[诊断问题]
    C --> D[发现问题1:<br/>patience=10过早停止]
    C --> E[发现问题2:<br/>lrf=0.01衰减过快]
    C --> F[发现问题3:<br/>数据增强过激进]
    C --> G[ 发现核心问题:<br/>auto_augment=randaugment<br/>破坏知识热力图]
### 阶段2: SOTA对齐训练 (12月23日 22:00-22:30)
`mermaid
flowchart LR
    A[调整超参数] --> B[启动训练<br/>50 epochs]
    B --> C[Epoch 38<br/>达最佳]
    C --> D[ F1=0.6774<br/>超越SOTA]
    D --> E[保存权重<br/>best.pt]
"`

**关键调整**:
-  auto_augment: randaugment  **none**
-  patience: 10  **50**
-  lrf: 0.01  **0.1**
-  epochs: 30  **50**
-  温和数据增强: translate=0.05, scale=0.2, erasing=0.2

### 阶段3: 路线验证 

**验证问题**: 我们的技术路线是否正确？

**答案**:  **完全正确！**

**论据**:
1. **问题定位准确**: 精准识别出auto_augment破坏知识热力图
2. **解决方案有效**: 关闭后Recall提升45.5%
3. **超越SOTA**: F1=0.6774 > 0.6753
4. **可复现**: 训练稳定，有完整记录

##  未来规划框架

### 短期目标 (1-2周)

#### 1. 10ch光谱增强实验 [P0]
**目标**: 验证额外3个光谱通道的价值

`bash
python scripts/training/train_7ch_main.py \
  --data-config configs/datasets/marida_mask_rescue_10ch.yaml \
  --epochs 50 --batch 8 --imgsz 512 \
  --lr0 0.0005 --lrf 0.1 --warmup-epochs 3.0 \
  --patience 50 --close-mosaic 0 \
  --auto-augment none --translate 0.05 --scale 0.2 --erasing 0.2 \
  --disable-mosaic --device 0
"`

**期望结果**:
- F1: 0.6774  0.70+ (+2~5%)
- 额外通道: FDI(水体指数) + NDVI + RedEdge

#### 2. EIoU Loss实现 [P1]
**Alpha规划**: 先输出结构 + Mermaid图
"`mermaid
flowchart TD
    A[src/core/losses/eiou_loss.py] --> B[EIoULoss类]
    B --> C[实现公式:<br/>L_EIoU = 1-IoU + ρ/c + v/c]
    C --> D[集成到MaskRescueTrainer]
    D --> E[实验验证]
"`

**期望提升**:
- 小目标Recall: +5~8%
- 边界框质量改善

#### 3. Bi-FPN Neck实现 [P2]
**Alpha规划**: 
`mermaid
flowchart TD
    A[src/modules/necks/bifpn.py] --> B[Bi-FPN模块]
    B --> C[替换PANet]
    C --> D[双向特征金字塔]
    D --> E[实验对比]
`

**期望提升**:
- 小目标mAP: +5~8%

### 中期目标 (2-4周)

#### 4. 知识蒸馏实验 [P3]
**方案**:
- Teacher: YOLOv11x-10ch (期望F1>0.70)
- Student: YOLOv11n-10ch with KD loss
- 期望: Student F1+0.02~0.04，保持轻量

#### 5. 完整消融实验
| 实验组 | 配置 | 期望F1 | 状态 |
|--------|------|--------|------|
| Baseline | 7ch基线 | 0.6774 |  完成 |
| +10ch | 添加3光谱通道 | 0.70 |  待做 |
| +10ch+EIoU | 改进Loss | 0.72 |  待做 |
| +10ch+EIoU+BiFPN | 改进Neck | 0.74 |  待做 |
| +10ch+EIoU+BiFPN+KD | 知识蒸馏 | 0.75 |  待做 |

### 长期目标 (1-2月)

#### 6. 论文撰写
**目标期刊**: Remote Sensing / IEEE TGRS

**论文结构**:
1. Introduction
   - 海洋塑料污染问题
   - 多光谱遥感优势
2. Related Work
   - 海洋碎片检测方法
   - 多光谱目标检测
3. **Methodology (核心创新)**
   - 知识热力图第7通道
   - 10通道光谱增强
   - EIoU Loss
   - Bi-FPN架构
4. Experiments
   - 消融实验表
   - 与SOTA对比
   - 可视化分析
5. Conclusion

##  项目配置验证

### 环境
- Python 3.13.5, PyTorch 2.6.0+cu124, Ultralytics 8.3.231
- GPU: RTX 3090 Ti 24GB
- conda env: yolo_test

### 数据集状态
#### 7ch 
- 样本: Train 347, Val 114, Test 53
- Shape: (256, 256, 7)
- 通道: S2六波段 + 知识热力图
- YAML: configs/datasets/marida_mask_rescue_7ch.yaml

#### 10ch 
- 样本: Train 347, Val 114, Test 53
- Shape: (256, 256, 10)
- 通道: B1-B12(0-6) + FDI(7) + NDVI(8) + RedEdge(9)
- YAML: configs/datasets/marida_mask_rescue_10ch.yaml
- 归一化: 所有通道[0,1] (NDVI/RedEdge失去负值物理意义)

##  经验教训

###  做对的事
1. **保护知识热力图**: 识别出auto_augment的破坏性
2. **严格对齐SOTA**: 逐参数对比找出差异
3. **耐心训练**: patience=50避免过早停止
4. **温和增强**: 小目标需要保护空间特征

###  避免的坑
1.  **盲目使用数据增强**: randaugment不适合知识通道
2.  **过早停止训练**: patience=10会错过最佳性能
3.  **学习率衰减过快**: lrf=0.01限制后期优化
4.  **激进的几何增强**: translate/scale过大破坏小目标

##  目录规范 (Alpha工作流)
1. 新代码放 src/, scripts/, configs/
2. **Alpha规划**: 先输出目录结构 + Mermaid图
3. 再编码实现
4. pathlib约束

##  关键文件路径
- **最佳7ch权重**: experiments/runs/baseline_sota_aligned_1223_2219/weights/best.pt
- **训练配置**: configs/datasets/marida_mask_rescue_7ch.yaml
- **自定义Dataset**: ultralytics_plugins/mask_rescue_dataset.py
- **自定义Trainer**: ultralytics_plugins/mask_rescue_trainer.py
- **训练脚本**: scripts/training/train_7ch_main.py

---
**最后更新**: 2025-12-25
**状态**:  7ch基线已超越SOTA, 准备启动10ch实验
