# MARIDA类别策略分析

## 原始15个类别

| ID | 类别名称 | 用途建议 | 理由 |
|----|---------|---------|------|
| 1 | Marine Debris | ✅ **核心目标** | 主要检测对象 |
| 2 | Dense Sargassum | ✅ **核心目标** | 浮游藻类，可能与垃圾混淆 |
| 3 | Sparse Sargassum | ✅ **核心目标** | 稀疏藻类 |
| 4 | Natural Organic Material | ✅ **核心目标** | 自然有机物 |
| 5 | Ship | ✅ **核心目标** | 船只，重要目标 |
| 6 | Clouds | ⚠️ **背景** | 云层会遮挡，可作为背景或单独类 |
| 7 | Marine Water | ❌ **背景** | 纯海水背景 |
| 8 | Sediment-Laden Water | ⚠️ **可选** | 含沉积物水体 |
| 9 | Foam | ✅ **核心目标** | 泡沫，可能与垃圾混淆 |
| 10 | Turbid Water | ❌ **背景** | 浑浊水体 |
| 11 | Shallow Water | ❌ **背景** | 浅水 |
| 12 | Waves | ❌ **背景** | 波浪 |
| 13 | Cloud Shadows | ❌ **背景** | 云影 |
| 14 | Wakes | ✅ **核心目标** | 船迹，可能指示船只 |
| 15 | Mixed Water | ❌ **背景** | 混合水体 |

---

## 📊 推荐的类别策略

### **方案1：细粒度7类检测（推荐用于研究）**
```yaml
classes:
  0: Marine_Debris      # 海洋垃圾
  1: Dense_Sargassum    # 密集马尾藻
  2: Sparse_Sargassum   # 稀疏马尾藻
  3: Natural_Organic    # 自然有机物
  4: Ship               # 船只
  5: Foam               # 泡沫
  6: Wake               # 船迹
```
**优点**: 保留详细信息，便于分析混淆情况  
**缺点**: 类别多，不平衡严重

---

### **方案2：粗粒度3类检测（推荐用于应用）**
```yaml
classes:
  0: Marine_Debris      # 海洋垃圾（原类别1）
  1: Floating_Object    # 浮游物体（合并2,3,4,9）
  2: Ship_Wake          # 船只及船迹（合并5,14）
```
**优点**: 
- 减少类别不平衡
- 更容易训练
- 实际应用更有意义

**类别合并逻辑**:
```python
def merge_classes(original_class_id):
    """将15类映射到3类"""
    if original_class_id == 1:
        return 0  # Marine Debris
    elif original_class_id in [2, 3, 4, 9]:
        return 1  # Floating Objects (Sargassum, organic, foam)
    elif original_class_id in [5, 14]:
        return 2  # Ship and Wake
    else:
        return None  # Background
```

---

### **方案3：二分类检测（推荐用于快速部署）**
```yaml
classes:
  0: Target    # 所有目标（1,2,3,4,5,9,14）
  1: Ship      # 单独保留船只（5）
```
**优点**: 
- 最简单，训练最快
- 适合"有/无"目标检测

---

## 🎯 不同场景的推荐方案

| 应用场景 | 推荐方案 | 理由 |
|---------|---------|------|
| **科研分析** | 方案1（7类） | 需要详细分类信息 |
| **实时监测** | 方案2（3类） | 平衡精度与速度 |
| **边缘部署** | 方案3（2类） | 最快速度 |
| **初步验证** | 方案2（3类） | 最佳起点 |

---

## ⚠️ 当前代码的问题

您的代码中：
```python
MARINE_DEBRIS_CLASSES = [1, 2, 3, 4, 5, 9, 14]
bboxes = mask_to_bboxes(class_mask, class_id - 1, min_area)  # class_id - 1
```

**问题**:
1. ❌ `class_id - 1` 会导致类别ID变为 0,1,2,3,4,8,13（不连续）
2. ❌ YOLO需要连续的类别ID: 0,1,2,3,4,5,6

**修复方案**:
```python
# 创建类别映射字典
ORIGINAL_TO_YOLO = {
    1: 0,  # Marine Debris
    2: 1,  # Dense Sargassum
    3: 2,  # Sparse Sargassum
    4: 3,  # Natural Organic
    5: 4,  # Ship
    9: 5,  # Foam
    14: 6  # Wake
}

# 在转换时使用
yolo_class_id = ORIGINAL_TO_YOLO.get(class_id)
if yolo_class_id is not None:
    bboxes = mask_to_bboxes(class_mask, yolo_class_id, min_area)
```

---

## 💡 我的最终建议

**第一阶段：使用方案2（3类）**
- 更容易训练成功
- 类别有实际意义
- 便于后续优化

**如果效果好，再尝试方案1（7类）**
- 提供更细粒度的分类
- 用于深入分析

**代码实现**:
```python
# 方案2的实现
CLASS_MERGE_MAP = {
    1: 0,   # Marine Debris -> Marine_Debris
    2: 1,   # Dense Sargassum -> Floating_Object
    3: 1,   # Sparse Sargassum -> Floating_Object
    4: 1,   # Natural Organic -> Floating_Object
    5: 2,   # Ship -> Ship_Wake
    9: 1,   # Foam -> Floating_Object
    14: 2   # Wake -> Ship_Wake
}

YOLO_CLASS_NAMES = [
    'Marine_Debris',
    'Floating_Object', 
    'Ship_Wake'
]
```

这样可以：
✅ 减轻类别不平衡  
✅ 提高训练稳定性  
✅ 保留核心检测能力  
✅ 便于实际应用
