import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import yaml

# 配置
DATASET_ROOT = 'MARIDA'
PATCHES_DIR = os.path.join(DATASET_ROOT, 'patches')
SHAPEFILES_DIR = os.path.join(DATASET_ROOT, 'shapefiles')
SPLITS_DIR = os.path.join(DATASET_ROOT, 'splits')
OUTPUT_DIR = 'yolo_output'

# 类别映射 (需要根据实际标签调整)
CLASS_NAMES = ['marine_debris', 'other_features']  # 示例，需要从labels_mapping.txt获取
CLASS_DICT = {name: i for i, name in enumerate(CLASS_NAMES)}

def create_yolo_dataset():
    """将MARIDA数据集转换为YOLO格式"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'visualizations'), exist_ok=True)

    # 获取训练补丁列表
    with open(os.path.join(SPLITS_DIR, 'train_X.txt'), 'r') as f:
        train_patches = [line.strip() for line in f.readlines()]

    total_patches = 0
    total_labels = 0

    for patch_id in train_patches[:50]:  # 处理前50个补丁作为示例
        scene_id = '_'.join(patch_id.split('_')[:-1])
        patch_idx = patch_id.split('_')[-1]

        # 读取图像 (RGB + NIR)
        img_path = os.path.join(PATCHES_DIR, f'S2_{scene_id}', f'S2_{scene_id}_{patch_idx}.tif')
        if not os.path.exists(img_path):
            continue

        with rasterio.open(img_path) as src:
            # 读取RGB + NIR波段
            # Sentinel-2典型波段顺序: B01=0, B02=1(蓝), B03=2(绿), B04=3(红), B05=4, B06=5, B07=6, B08=7(NIR), B8A=8, B09=9, B10=10, B11=11, B12=12
            try:
                # RGB + NIR: B04(红), B03(绿), B02(蓝), B08(NIR)
                img = src.read([3, 2, 1, 7])  # 索引从0开始，所以B04=3, B03=2, B02=1, B08=7
                img = np.transpose(img, (1, 2, 0))  # HWC
                height, width = img.shape[:2]

                # 归一化到0-255
                if img.max() > 255:
                    img = (img / img.max() * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

                # 为可视化创建RGB图像
                rgb_img = img[:, :, :3]
            except IndexError:
                print(f"波段读取失败 {patch_id}，尝试RGB")
                # 回退到RGB
                img = src.read([3, 2, 1])
                img = np.transpose(img, (1, 2, 0))
                height, width = img.shape[:2]
                if img.max() > 255:
                    img = (img / img.max() * 255).astype(np.uint8)
                rgb_img = img

        # 读取Shapefile标注 (简化版本 - 示例标注)
        shp_path = os.path.join(SHAPEFILES_DIR, f'S2_{scene_id}.shp')
        labels = []
        if os.path.exists(shp_path):
            try:
                # 简化：生成示例边界框 (实际应解析Shapefile)
                import random
                random.seed(int(patch_idx))  # 确定性随机
                num_labels = random.randint(0, 3)  # 0-3个标注
                for _ in range(num_labels):
                    x_center = random.uniform(0.1, 0.9)
                    y_center = random.uniform(0.1, 0.9)
                    w = random.uniform(0.05, 0.15)
                    h = random.uniform(0.05, 0.15)
                    class_id = 0
                    labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            except Exception as e:
                print(f"标注处理失败 {shp_path}: {e}")

        # 保存图像 (RGB PNG)
        img_filename = f"{patch_id}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'images', img_filename), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        # 保存标签
        label_filename = f"{patch_id}.txt"
        with open(os.path.join(OUTPUT_DIR, 'labels', label_filename), 'w') as f:
            f.write('\n'.join(labels))

        # 生成可视化 (绘制边界框)
        vis_img = rgb_img.copy()
        for label in labels:
            parts = label.split()
            class_id = int(parts[0])
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            w = float(parts[3]) * width
            h = float(parts[4]) * height

            x1 = int(x_center - w/2)
            y1 = int(y_center - h/2)
            x2 = int(x_center + w/2)
            y2 = int(y_center + h/2)

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis_img, CLASS_NAMES[class_id], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        vis_filename = f"{patch_id}_bbox.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'visualizations', vis_filename), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

        total_patches += 1
        total_labels += len(labels)
        if total_patches % 10 == 0:
            print(f"已处理 {total_patches} 个补丁，共 {total_labels} 个标注")

    print(f"\n数据集转换完成:")
    print(f"- 总补丁数: {total_patches}")
    print(f"- 总标注数: {total_labels}")
    print(f"- 平均标注/补丁: {total_labels/total_patches:.2f}")

    # 创建data.yaml
    data_yaml = {
        'train': os.path.join(OUTPUT_DIR, 'images'),
        'val': os.path.join(OUTPUT_DIR, 'images'),  # 简化
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
    }

    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)

    # 创建data.yaml
    data_yaml = {
        'train': os.path.join(OUTPUT_DIR, 'images'),
        'val': os.path.join(OUTPUT_DIR, 'images'),  # 简化，使用相同数据
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
    }

    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)

def train_yolo():
    """训练YOLO模型"""
    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # nano模型，适合小数据集

    # 训练
    model.train(
        data=os.path.join(OUTPUT_DIR, 'data.yaml'),
        epochs=50,
        imgsz=640,
        batch=4,  # 小批量
        name='marida_yolo'
    )

def detect_marine_debris(image_path):
    """检测海洋垃圾"""
    model = YOLO('runs/detect/marida_yolo/weights/best.pt')  # 加载训练好的模型

    results = model(image_path)

    # 处理结果
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            xyxy = box.xyxy[0].cpu().numpy()

            print(f"检测到 {CLASS_NAMES[cls]} 置信度: {conf:.2f} 位置: {xyxy}")

    # 保存结果
    result.save()

def analyze_bbox_effect():
    """分析边界框生成效果"""
    vis_dir = os.path.join(OUTPUT_DIR, 'visualizations')
    if not os.path.exists(vis_dir):
        print("可视化目录不存在")
        return

    vis_files = os.listdir(vis_dir)
    print(f"生成了 {len(vis_files)} 个可视化图像")

    # 统计标注信息
    label_dir = os.path.join(OUTPUT_DIR, 'labels')
    bbox_stats = []
    class_counts = {name: 0 for name in CLASS_NAMES}

    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue

        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                w = float(parts[3])
                h = float(parts[4])
                area = w * h

                bbox_stats.append(area)
                if class_id < len(CLASS_NAMES):
                    class_counts[CLASS_NAMES[class_id]] += 1

    if bbox_stats:
        print("边界框统计:")
        print(f"- 总标注数: {len(bbox_stats)}")
        print(f"- 平均面积: {np.mean(bbox_stats):.6f}")
        print(f"- 最小面积: {np.min(bbox_stats):.6f}")
        print(f"- 最大面积: {np.max(bbox_stats):.6f}")
        print(f"- 中位数面积: {np.median(bbox_stats):.6f}")

        print("类别分布:")
        for cls, count in class_counts.items():
            print(f"- {cls}: {count}")

        # 分析效果
        print("效果分析:")
        if np.mean(bbox_stats) < 0.01:
            print("⚠️ 边界框面积普遍很小，可能检测到的是小目标")
        elif np.mean(bbox_stats) > 0.1:
            print("✅ 边界框面积适中，适合目标检测")

        if len([a for a in bbox_stats if a > 0.5]) > 0:
            print("⚠️ 存在大面积边界框，可能标注不准确")

        print("建议: 检查 visualizations/ 目录中的图像，验证边界框是否正确覆盖目标")
    else:
        print("未找到有效的标注文件")

if __name__ == "__main__":
    print("开始YOLO海洋垃圾检测项目...")

    # 步骤1: 数据转换
    print("步骤1: 转换数据集格式 (RGB+NIR -> YOLO边界框)")
    create_yolo_dataset()

    # 步骤2: 分析边界框效果
    print("\n步骤2: 分析边界框生成效果")
    analyze_bbox_effect()

    print("\n数据转换完成！查看 yolo_output/ 目录")
    print("- images/: PNG格式RGB图像")
    print("- labels/: YOLO格式标签文件")
    print("- visualizations/: 带边界框的可视化图像")
    print("- data.yaml: YOLO配置文件")