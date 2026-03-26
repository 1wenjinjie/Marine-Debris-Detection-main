"""
Comparative Training Experiments: YOLOv8 vs YOLOv11 × RGB vs 6-band
4组实验：
1. YOLOv8-3ch (RGB)
2. YOLOv8-6ch (RGB+NIR+SWIR)
3. YOLOv11-3ch (RGB)
4. YOLOv11-6ch (RGB+NIR+SWIR)
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
import time

# 实验配置
EXPERIMENTS = {
    'yolov8_3ch': {
        'name': 'YOLOv8-RGB',
        'data': 'marida_rgb_v8.yaml',
        'model': 'yolov8n.pt',
        'description': 'YOLOv8n with 3-channel RGB input'
    },
    'yolov8_6ch': {
        'name': 'YOLOv8-6band',
        'data': 'marida_6band_v8.yaml',
        'model': 'yolov8n_6ch.yaml',
        'description': 'YOLOv8n with 6-channel (RGB+NIR+SWIR) input'
    },
    'yolov11_3ch': {
        'name': 'YOLOv11-RGB',
        'data': 'marida_rgb_v11.yaml',
        'model': 'yolo11n.pt',
        'description': 'YOLOv11n with 3-channel RGB input'
    },
    'yolov11_6ch': {
        'name': 'YOLOv11-6band',
        'data': 'marida_6band_v11.yaml',
        'model': 'yolo11n_6ch_init.pt',
        'description': 'YOLOv11n with 6-channel (RGB+NIR+SWIR) input and expanded weights'
    }
}

# 训练参数
TRAIN_PARAMS = {
    'epochs': 50,
    'imgsz': 256,
    'batch': 8,
    'patience': 10,
    'device': 0,  # GPU
    'workers': 4,
    'save_period': 10
}

def check_prerequisites():
    """检查前置条件"""
    print("=" * 80)
    print("检查前置条件...")
    print("=" * 80)
    
    # 检查数据文件
    required_files = [
        'marida_rgb_v8.yaml',
        'marida_6band_v8.yaml',
        'marida_rgb_v11.yaml',
        'marida_6band_v11.yaml',
        'yolov8n_6ch.yaml',
        'yolov11n_6ch.yaml',
        'yolo11n.pt',
        'yolo11n_6ch_init.pt'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ 缺少以下配置文件:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    # 检查预训练权重
    if not Path('yolov8n.pt').exists():
        print("⚠️  未找到 yolov8n.pt，将自动下载...")
    
    if not Path('yolov11n.pt').exists():
        print("⚠️  未找到 yolov11n.pt，将自动下载...")
    
    # 检查数据目录
    required_dirs = [
        'MARIDA_YOLO_RGB_split',
        'MARIDA_YOLO_6band_split'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("❌ 缺少以下数据目录:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        return False
    
    print("✅ 所有前置条件满足！")
    return True

def train_experiment(exp_name, config):
    """执行单个实验的训练"""
    print("\n" + "=" * 80)
    print(f"开始实验: {config['name']}")
    print(f"描述: {config['description']}")
    print("=" * 80)
    
    # 构建训练命令
    cmd = [
        'yolo', 'detect', 'train',
        f"data={config['data']}",
        f"model={config['model']}",
        f"epochs={TRAIN_PARAMS['epochs']}",
        f"imgsz={TRAIN_PARAMS['imgsz']}",
        f"batch={TRAIN_PARAMS['batch']}",
        f"patience={TRAIN_PARAMS['patience']}",
        f"device={TRAIN_PARAMS['device']}",
        f"workers={TRAIN_PARAMS['workers']}",
        f"save_period={TRAIN_PARAMS['save_period']}",
        f"project=runs/comparative_experiments",
        f"name={exp_name}",
        'exist_ok=False',
        'verbose=True'
    ]
    
    print(f"\n执行命令: {' '.join(cmd)}\n")
    
    start_time = time.time()
    
    try:
        # 执行训练
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ 实验 {config['name']} 完成!")
        print(f"⏱️  训练耗时: {elapsed_time/3600:.2f} 小时")
        
        return {
            'status': 'success',
            'elapsed_time': elapsed_time,
            'output_dir': f"runs/comparative_experiments/{exp_name}"
        }
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ 实验 {config['name']} 失败!")
        print(f"错误信息: {e}")
        
        return {
            'status': 'failed',
            'elapsed_time': elapsed_time,
            'error': str(e)
        }

def validate_experiment(exp_name, config, train_result):
    """验证单个实验"""
    if train_result['status'] != 'success':
        print(f"\n⚠️  跳过验证 {config['name']}（训练失败）")
        return None
    
    print("\n" + "-" * 80)
    print(f"验证实验: {config['name']}")
    print("-" * 80)
    
    # 构建验证命令
    weights_path = f"runs/comparative_experiments/{exp_name}/weights/best.pt"
    
    if not Path(weights_path).exists():
        print(f"❌ 未找到权重文件: {weights_path}")
        return None
    
    cmd = [
        'yolo', 'detect', 'val',
        f"model={weights_path}",
        f"data={config['data']}",
        f"imgsz={TRAIN_PARAMS['imgsz']}",
        f"batch={TRAIN_PARAMS['batch']}",
        f"device={TRAIN_PARAMS['device']}",
        f"project=runs/comparative_experiments",
        f"name={exp_name}_val",
        'save_json=True',
        'save_hybrid=True'
    ]
    
    print(f"执行命令: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✅ 验证 {config['name']} 完成!")
        
        return {
            'status': 'success',
            'output_dir': f"runs/comparative_experiments/{exp_name}_val"
        }
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 验证 {config['name']} 失败!")
        print(f"错误信息: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }

def collect_results():
    """收集所有实验的结果"""
    print("\n" + "=" * 80)
    print("收集实验结果...")
    print("=" * 80)
    
    results = {}
    
    for exp_name, config in EXPERIMENTS.items():
        result_file = f"runs/comparative_experiments/{exp_name}/results.csv"
        
        if not Path(result_file).exists():
            print(f"⚠️  未找到结果文件: {result_file}")
            continue
        
        try:
            # 读取results.csv的最后一行（最终结果）
            with open(result_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    header = lines[0].strip().split(',')
                    last_line = lines[-1].strip().split(',')
                    
                    # 提取关键指标
                    metrics = {}
                    for i, col in enumerate(header):
                        col = col.strip()
                        if i < len(last_line):
                            try:
                                metrics[col] = float(last_line[i])
                            except ValueError:
                                metrics[col] = last_line[i]
                    
                    results[exp_name] = {
                        'name': config['name'],
                        'description': config['description'],
                        'metrics': metrics
                    }
                    
                    print(f"✅ 收集了 {config['name']} 的结果")
        
        except Exception as e:
            print(f"❌ 读取 {config['name']} 结果时出错: {e}")
    
    return results

def save_summary(results, experiment_log):
    """保存实验总结"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'train_params': TRAIN_PARAMS,
        'experiments': {},
        'comparison': {}
    }
    
    # 添加各实验结果
    for exp_name, result in results.items():
        summary['experiments'][exp_name] = result
    
    # 添加实验日志
    summary['experiment_log'] = experiment_log
    
    # 比较分析
    if len(results) >= 2:
        print("\n" + "=" * 80)
        print("性能对比分析")
        print("=" * 80)
        
        # 提取mAP50指标进行比较
        comparisons = []
        for exp_name, result in results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                map50 = metrics.get('metrics/mAP50(B)', metrics.get('mAP50', 0))
                precision = metrics.get('metrics/precision(B)', metrics.get('precision', 0))
                recall = metrics.get('metrics/recall(B)', metrics.get('recall', 0))
                
                comparisons.append({
                    'experiment': result['name'],
                    'mAP50': map50,
                    'precision': precision,
                    'recall': recall
                })
        
        # 按mAP50排序
        comparisons.sort(key=lambda x: x.get('mAP50', 0), reverse=True)
        
        print("\n排名（按 mAP@0.5）:")
        for i, comp in enumerate(comparisons, 1):
            print(f"{i}. {comp['experiment']:<20} mAP50: {comp['mAP50']:.4f}  "
                  f"Precision: {comp['precision']:.4f}  Recall: {comp['recall']:.4f}")
        
        summary['comparison']['ranking'] = comparisons
    
    # 保存到JSON文件
    output_file = 'runs/comparative_experiments/experiment_summary.json'
    os.makedirs('runs/comparative_experiments', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 实验总结已保存到: {output_file}")
    
    return summary

def main():
    """主函数"""
    print("=" * 80)
    print("YOLO 对比实验：YOLOv8 vs YOLOv11 × RGB vs 6-band")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查前置条件
    if not check_prerequisites():
        print("\n❌ 前置条件检查失败，请先准备好配置文件和数据集！")
        return
    
    # 实验日志
    experiment_log = {}
    
    # 训练所有实验
    print("\n" + "=" * 80)
    print("阶段 1: 训练所有模型")
    print("=" * 80)
    
    for exp_name, config in EXPERIMENTS.items():
        train_result = train_experiment(exp_name, config)
        experiment_log[exp_name] = {
            'train': train_result
        }
        
        # 训练后立即验证
        val_result = validate_experiment(exp_name, config, train_result)
        experiment_log[exp_name]['validation'] = val_result
        
        print("\n" + "~" * 80 + "\n")
    
    # 收集结果
    results = collect_results()
    
    # 保存总结
    summary = save_summary(results, experiment_log)
    
    print("\n" + "=" * 80)
    print("所有实验完成！")
    print("=" * 80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n结果目录: runs/comparative_experiments/")
    print(f"总结文件: runs/comparative_experiments/experiment_summary.json")

if __name__ == '__main__':
    main()
