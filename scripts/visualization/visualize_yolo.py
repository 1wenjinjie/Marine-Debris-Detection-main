import cv2
import numpy as np
import os
import glob

def visualize_sample(image_dir, label_dir, class_names, num_samples=5):
    image_files = glob.glob(os.path.join(image_dir, '*.tif'))
    for i, img_path in enumerate(image_files[:num_samples]):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f'{base_name}.txt')
        
        # 读取6通道影像，取RGB显示
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 6:
            img_rgb = img[:, :, :3]
        else:
            img_rgb = img
        
        height, width = img_rgb.shape[:2]
        
        # 读取YOLO标签并绘制边界框
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    w = float(parts[3]) * width
                    h = float(parts[4]) * height
                    
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)
                    
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_rgb, class_names[class_id], (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow(f'Sample {i+1}: {base_name}', img_rgb)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

# 示例用法（替换为实际路径）
image_dir = 'MARIDA_YOLO_split/train/images'
label_dir = 'MARIDA_YOLO_split/train/labels'
class_names = ['Plastic_debris', 'Ship', 'Marine_water', 'Dense_sargassum', 'Sparse_sargassum', 'Natural_organic_material', 'Ship_wake', 'Mixed_pixel', 'Foam', 'Marine_litter_macro', 'Marine_litter_micro']  # 从 classes.txt 读取
visualize_sample(image_dir, label_dir, class_names)
