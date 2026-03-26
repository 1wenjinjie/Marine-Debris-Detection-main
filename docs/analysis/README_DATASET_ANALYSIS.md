# MARIDA数据集分析报告

## 📋 数据集概览

### 基本信息
- **数据源**: Sentinel-2卫星多光谱影像
- **场景数量**: 63个
- **图像总数**: ~1400+张
- **图像尺寸**: 256×256像素
- **波段数**: 11个光谱波段
- **数据类型**: float32 (反射率值)
- **坐标系统**: EPSG:32748 (UTM Zone 48S)

### 文件结构
```
MARIDA/
├── patches/
│   ├── S2_*/ (63个场景文件夹)
│   │   ├── *.tif (原始11波段影像)
│   │   ├── *_cl.tif (分类标签)
│   │   └── *_conf.tif (置信度)
├── shapefiles/ (矢量标注)
├── splits/ (训练/测试划分)
└── labels_mapping.txt (类别映射)
```

---

## ⚠️ 关键挑战

### 1. 极度类别不平衡 ⚠️⚠️⚠️
```
背景(海洋): 99.95%
目标区域:   0.05%
```
**影响**: 模型容易学成"全预测为背景"
**解决方案**: 
- Focal Loss
- 类别加权
- 过采样

### 2. 小目标检测难度 ⚠️⚠️
- 海洋垃圾目标小（几个像素到几十个像素）
- 需要高分辨率特征图
- 建议使用YOLOv11的小目标增强

### 3. 标签格式不匹配 ⚠️
- 当前: 语义分割格式（像素级）
- 需要: YOLO目标检测格式（bbox）
- **必须转换**: 从分割mask提取边界框

### 4. 数据量偏少 ⚠️
- 1400+张对深度学习较少
- 建议数据增强 + 预训练模型

---

## 🎯 适用模型推荐

### 目标检测（推荐YOLOv11）
| 模型 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| YOLOv11-n | 速度快，适合边缘部署 | 小目标检测弱 | ⭐⭐⭐ |
| YOLOv11-s | 速度与精度平衡 | 需要GPU | ⭐⭐⭐⭐⭐ |
| YOLOv11-m | 精度高 | 较慢 | ⭐⭐⭐⭐ |

### 语义分割
| 模型 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| U-Net | 标签直接可用 | 速度慢 | ⭐⭐⭐⭐ |
| DeepLabv3+ | 精度高 | 计算量大 | ⭐⭐⭐ |

---

## 📝 预处理需求与实施方案

### ⚠️ 重要调整建议

#### 1. 图像尺寸：512×512 vs 256×256
**用户方案**: 裁剪为512×512  
**当前数据**: 256×256

**分析**:
- ✅ **MARIDA原始分辨率是10m/像素**, 256×256覆盖2.56km×2.56km区域
- ✅ **上采样到512×512**: 可提供更多特征图层级,改善小目标检测
- ⚠️ **注意**: 不会增加实际空间分辨率,但能让模型有更多计算空间
- ✅ **建议**: 使用双线性插值上采样,保持数据一致性

**实施方案**:
```python
import cv2
# 影像和标签都需要调整
image_512 = cv2.resize(image_256, (512, 512), interpolation=cv2.INTER_LINEAR)
label_512 = cv2.resize(label_256, (512, 512), interpolation=cv2.INTER_NEAREST)
```

---

#### 2. 波段选择：RGB+NIR (4通道)
**用户方案**: 提取B2(蓝)、B3(绿)、B4(红)、B8(近红外)  
**当前数据**: 11个波段

**分析**:
- ✅ **RGB可见光**: 人眼可识别,便于可视化
- ✅ **NIR近红外**: 水体/植被敏感,增强检测能力
- ✅ **减少计算量**: 4通道比11通道轻量化
- ⚠️ **权衡**: 丢失短波红外(SWIR)等信息,可能影响精度

**波段映射**:
```
Sentinel-2波段 → 输出通道
B2 (Blue, ~490nm)  → Channel 0
B3 (Green, ~560nm) → Channel 1  
B4 (Red, ~665nm)   → Channel 2
B8 (NIR, ~842nm)   → Channel 3
```

**实施方案**:
```python
# 假设bands = [B1, B2, B3, ..., B11]
rgb_nir = np.stack([
    bands[1],  # B2 Blue
    bands[2],  # B3 Green
    bands[3],  # B4 Red
    bands[7]   # B8 NIR
], axis=-1)
```

---

#### 3. 数据增强策略
**用户方案**: 旋转、翻转、亮度/对比度调整、色彩抖动、仿射变换

**推荐增强配置**:
```python
# Albumentations增强pipeline
import albumentations as A

train_transforms = A.Compose([
    A.RandomRotate90(p=0.5),                    # 90度旋转
    A.Flip(p=0.5),                               # 随机翻转
    A.RandomBrightnessContrast(p=0.3),           # 亮度/对比度
    A.HueSaturationValue(p=0.3),                 # 色彩抖动
    A.ShiftScaleRotate(                          # 仿射变换
        shift_limit=0.1,
        scale_limit=0.1, 
        rotate_limit=15,
        p=0.5
    ),
    A.GaussNoise(p=0.2),                         # 高斯噪声
], bbox_params=A.BboxParams(format='yolo'))
```

**额外建议**:
- ✅ **Mosaic增强**: YOLO特有,将4张图拼接
- ✅ **MixUp**: 混合两张图像,增加泛化
- ⚠️ **避免过度增强**: 保持海洋场景真实性

---

#### 4. 标签格式转换
**用户方案**: 分割mask → YOLO bbox格式  
**当前数据**: `*_cl.tif` 像素级分类

**转换流程**:
```python
# 1. 读取分割mask (256×256)
mask = read_tif('S2_xxx_cl.tif')

# 2. 连通域分析
from scipy import ndimage
labeled, num_objects = ndimage.label(mask > 0)

# 3. 提取每个目标的bbox
for obj_id in range(1, num_objects+1):
    coords = np.where(labeled == obj_id)
    ymin, ymax = coords[0].min(), coords[0].max()
    xmin, xmax = coords[1].min(), coords[1].max()
    
    # 4. 转换为YOLO格式 (归一化到[0,1])
    x_center = (xmin + xmax) / 2 / 512
    y_center = (ymin + ymax) / 2 / 512
    width = (xmax - xmin) / 512
    height = (ymax - ymin) / 512
    
    # 5. 保存: class_id x_center y_center width height
    with open('S2_xxx.txt', 'a') as f:
        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
```

**注意事项**:
- ⚠️ **过滤小目标**: bbox面积 < 4像素的可能是噪声
- ⚠️ **边界处理**: 确保坐标在[0,1]范围内
- ✅ **类别映射**: Marine Debris → 0, Ship → 1, etc.

---

#### 5. 数据集划分
**用户方案**: train 70% / val 20% / test 10%  
**当前数据**: 63个场景,~1400张图像

**划分策略**:
```python
# 按场景划分,避免数据泄露
scenes = sorted(os.listdir('MARIDA/patches'))
np.random.seed(42)
np.random.shuffle(scenes)

n = len(scenes)
train_scenes = scenes[:int(n*0.7)]      # ~44个场景
val_scenes = scenes[int(n*0.7):int(n*0.9)]  # ~13个场景
test_scenes = scenes[int(n*0.9):]       # ~6个场景
```

**目录结构**:
```
processed_data/
├── train/
│   ├── images/
│   │   ├── S2_xxx.png (512×512×4)
│   └── labels/
│       └── S2_xxx.txt (YOLO format)
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

### 📋 预处理任务清单

- [ ] **步骤1**: 编写影像上采样脚本 (256→512)
- [ ] **步骤2**: 实现RGB+NIR提取与保存
- [ ] **步骤3**: 分割mask → YOLO bbox转换
- [ ] **步骤4**: 数据集按场景划分
- [ ] **步骤5**: 配置增强pipeline (Albumentations)
- [ ] **步骤6**: 生成YOLOv11配置文件 (data.yaml)
- [ ] **步骤7**: 验证数据完整性 (可视化检查)

---

## **

---

## 🚀 训练策略建议

### 1. 预训练模型
```python
# 使用COCO预训练的YOLOv11
model = YOLO('yolov11s.pt')  # 已在COCO上训练
```

### 2. 损失函数
```python
# Focal Loss处理不平衡
# Class Weighted Loss
# 小目标额外权重
```

### 3. 训练配置
```yaml
epochs: 200-300
batch_size: 16 (根据GPU调整)
optimizer: AdamW
lr: 0.001 (cosine decay)
patience: 50 (early stopping)
```

---

## 📊 预期效果

### 乐观估计
- **mAP@0.5**: 60-70%
- **召回率**: 50-60%
- **精确率**: 55-65%

### 悲观估计（不处理类别不平衡）
- **mAP@0.5**: 30-40%
- 模型倾向于预测"无目标"

---

## ✅ 总体评价

**结论**: 该数据集**可以使用**，但需要：

1. ✅ 认真处理类别不平衡
2. ✅ 进行标签格式转换
3. ✅ 充分的数据增强
4. ✅ 合理选择模型和超参数
5. ✅ 可能需要2-3轮实验迭代

**推荐方案**: YOLOv11-s + Focal Loss + 数据增强

---

## 📞 后续步骤

1. [ ] 编写标签转换脚本（分割→YOLO格式）
2. [ ] 实现数据增强pipeline
3. [ ] 设置训练配置文件
4. [ ] 开始baseline实验
5. [ ] 根据结果迭代优化

**预计开发时间**: 1-2周
**预计训练时间**: 2-3天（100-200 epochs）

---

## 📞 后续步骤（更新版）

### 🎯 完整预处理流程已就绪！

所有预处理脚本已创建完成。现在可以一键运行完整pipeline：

```bash
python run_preprocessing_pipeline.py
```

### 📁 已创建的预处理脚本

| 脚本 | 功能 | 输出 |
|------|------|------|
| `preprocess_resize.py` | 图像尺寸调整 (256→512) | `MARIDA_resized/` |
| `preprocess_extract_bands.py` | 提取RGB+NIR波段 | `MARIDA_4band/` |
| `preprocess_mask_to_yolo.py` | 分割mask→YOLO bbox | `MARIDA_YOLO/` |
| `preprocess_split_dataset.py` | 数据集划分 (70/20/10) | `MARIDA_YOLO_split/` |
| `preprocess_augmentation.py` | 数据增强 | `MARIDA_YOLO_augmented/` |
| `run_preprocessing_pipeline.py` | **一键运行全流程** | 所有上述输出 |

---

### 🚀 快速开始指南

#### 第1步：安装依赖
```bash
# 核心依赖
pip install rasterio numpy scipy tqdm

# 数据增强（可选，用于augmentation步骤）
pip install albumentations opencv-python

# YOLOv11训练（稍后）
pip install ultralytics
```

#### 第2步：运行预处理
```bash
# 方式1: 一键运行全流程（推荐）
python run_preprocessing_pipeline.py

# 方式2: 分步运行
python preprocess_resize.py              # 步骤1
python preprocess_extract_bands.py       # 步骤2
python preprocess_mask_to_yolo.py        # 步骤3
python preprocess_split_dataset.py       # 步骤4
python preprocess_augmentation.py        # 步骤5（可选）
```

#### 第3步：验证数据
```bash
# 查看数据集统计
cat MARIDA_YOLO_split/split_info.json

# 查看类别信息
cat MARIDA_YOLO_split/classes.txt

# 可视化检查（使用display_marida_images.py）
python display_marida_images.py
```

#### 第4步：开始训练
```bash
# 安装YOLOv11
pip install ultralytics

# 基础训练
yolo detect train data=MARIDA_YOLO_split/marida.yaml model=yolov11s.pt epochs=100 imgsz=512

# 高级训练（推荐）
yolo detect train \
  data=MARIDA_YOLO_split/marida.yaml \
  model=yolov11s.pt \
  epochs=200 \
  imgsz=512 \
  batch=16 \
  patience=50 \
  device=0 \
  project=marida_training \
  name=yolov11s_baseline \
  exist_ok=True
```

---

### 📋 详细时间表

#### 🟢 已完成
- [x] 数据集探索与分析
- [x] 编写所有预处理脚本
- [x] 创建一键运行pipeline
- [x] 文档完善

#### 🟡 下一步（1-2天）
1. [ ] 运行完整预处理pipeline
2. [ ] 验证数据完整性
3. [ ] 可视化检查标注质量
4. [ ] 配置训练环境（GPU/云服务器）

#### 🔵 训练阶段（3-7天）
1. [ ] Baseline实验（YOLOv11-n/s/m对比）
2. [ ] 超参数调优
3. [ ] 处理类别不平衡（Focal Loss）
4. [ ] 评估验证集性能
5. [ ] 测试集最终评估

#### 🟣 优化阶段（可选，1-2周）
1. [ ] 数据增强策略优化
2. [ ] 模型集成
3. [ ] 后处理优化（NMS阈值等）
4. [ ] 部署准备

---

### ⚠️ 预处理注意事项

#### 1. 磁盘空间需求
```
原始数据:         ~5-10 GB
调整尺寸后:       ~20-40 GB (512x512)
提取波段后:       ~10-20 GB (4 channels)
YOLO格式:         ~10-20 GB
增强后:           ~40-80 GB (3x augmentation)
──────────────────────────────
总计:             ~85-170 GB
```
**建议**: 至少准备200GB可用空间

#### 2. 内存需求
- 预处理阶段: 8GB+ RAM
- 训练阶段: 16GB+ RAM + GPU (8GB+ VRAM)

#### 3. 处理时间估算
| 步骤 | 预计时间 | 依赖GPU |
|------|---------|---------|
| 调整尺寸 | 30-60分钟 | ❌ |
| 提取波段 | 20-40分钟 | ❌ |
| YOLO转换 | 10-20分钟 | ❌ |
| 数据集划分 | 5-10分钟 | ❌ |
| 数据增强 | 1-2小时 | ❌ |
| **总计** | **2-3.5小时** | - |

---

### 🎓 训练建议配置

#### 推荐配置1：轻量级快速实验
```yaml
model: yolov11n.pt  # Nano model
epochs: 100
batch: 32
imgsz: 512
device: 0
```
**适用**: 快速验证数据集，GPU内存 ≤ 6GB

#### 推荐配置2：标准训练（推荐）
```yaml
model: yolov11s.pt  # Small model
epochs: 200
batch: 16
imgsz: 512
device: 0
optimizer: AdamW
lr0: 0.001
patience: 50
```
**适用**: 平衡精度与速度，GPU内存 ≥ 8GB

#### 推荐配置3：高精度训练
```yaml
model: yolov11m.pt  # Medium model
epochs: 300
batch: 8
imgsz: 640
device: 0
optimizer: AdamW
mosaic: 1.0
mixup: 0.1
```
**适用**: 追求最高精度，GPU内存 ≥ 16GB

---

### 📊 预期性能基准

基于MARIDA数据集的特点和类别不平衡：

#### 保守估计（Baseline）
```
mAP@0.5:    40-50%
mAP@0.5:0.95: 25-35%
Precision:  45-55%
Recall:     35-45%
```

#### 优化后（经过调优）
```
mAP@0.5:    60-70%
mAP@0.5:0.95: 40-50%
Precision:  60-70%
Recall:     50-60%
```

#### 成功标准
- ✅ **可接受**: mAP@0.5 > 50%
- ✅ **良好**: mAP@0.5 > 60%
- ✅ **优秀**: mAP@0.5 > 70%

---

### 🐛 常见问题解决

#### Q1: `ModuleNotFoundError: No module named 'rasterio'`
```bash
pip install rasterio
# 或者使用conda
conda install -c conda-forge rasterio
```

#### Q2: 预处理过程中内存不足
```python
# 修改批处理大小
# 在各脚本中添加批处理逻辑
# 或分批处理场景
```

#### Q3: YOLO训练显存不足
```bash
# 减小batch size
yolo detect train ... batch=4

# 使用更小的模型
yolo detect train ... model=yolov11n.pt

# 减小图像尺寸
yolo detect train ... imgsz=416
```

#### Q4: 转换后bbox数量为0
```python
# 检查最小面积阈值
# 在preprocess_mask_to_yolo.py中调整:
min_area=5  # 降低阈值
```

---

### 📚 参考资源

- **YOLOv11文档**: https://docs.ultralytics.com/
- **MARIDA论文**: https://doi.org/10.1371/journal.pone.0262247
- **Sentinel-2波段**: https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2
- **数据增强**: https://albumentations.ai/docs/

---

### 💡 关键成功因素

1. ✅ **处理类别不平衡** - 使用Focal Loss和过采样
2. ✅ **小目标优化** - 多尺度训练，调整anchor
3. ✅ **充分增强** - 旋转、翻转、Mosaic
4. ✅ **耐心训练** - 至少200 epochs
5. ✅ **持续监控** - 观察loss曲线，及时调整

---

### 🎉 预计开发时间线

| 阶段 | 时间 | 任务 |
|------|------|------|
| **第1天** | 2-3小时 | 运行预处理，验证数据 |
| **第2-3天** | 1-2天 | Baseline训练，初步评估 |
| **第4-7天** | 3-5天 | 调优，对比实验 |
| **第8-10天** | 2-3天 | 最终评估，撰写报告 |
| **总计** | **10-14天** | **从预处理到完成** |

---

## ✅ 总结

### 你的预处理方案非常专业！

结合你的建议，我们的pipeline具备：

1. ✅ **图像尺寸统一**: 512×512（上采样改善小目标）
2. ✅ **通道优化**: RGB+NIR（平衡性能与效果）
3. ✅ **标签转换**: 分割→YOLO（自动化提取）
4. ✅ **合理划分**: 70/20/10（防止过拟合）
5. ✅ **充分增强**: 旋转、色彩、仿射变换

### 现在就可以开始！

```bash
# 一键启动预处理
python run_preprocessing_pipeline.py
```

**预计2-3小时后，你就可以开始训练YOLOv11了！** 🚀

有任何问题或需要调整，随时告诉我！
