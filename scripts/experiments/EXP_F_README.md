# ========================================
# Exp-F (7ch + SIoU) 实验准备完成
# ========================================

## 实验配置
- **输入通道**: 7ch (RGB + NIR + RedEdge)
- **损失函数**: SIoU (Shape-aware IoU)  
- **知识蒸馏**: GT掩膜形态学距离变换热力图
- **模型**: YOLOv11n
- **训练参数**: 50 epochs, batch=8, imgsz=512

## 预期结果
- **保守估计**: F1  0.745 (+10% from baseline)
- **目标**: F1 > 0.70 (突破顶会发表门槛)
- **理论基础**: 
  - 7ch baseline F1=0.6774 (优于10ch)
  - SIoU在10ch上提升+35.9%
  - 两者结合预期产生协同效应

## 启动方式

### 方式1: 使用批处理脚本
.\scripts\experiments\run_exp_f.bat

### 方式2: 直接运行Python
python scripts\experiments\train_exp_f_7ch_siou.py

### 方式3: 后台训练(推荐)
Start-Process python -ArgumentList 'scripts\experiments\train_exp_f_7ch_siou.py' -NoNewWindow

## 训练监控
- **输出目录**: runs_marida\exp_f_7ch_siou\
- **实时日志**: 终端输出
- **训练时长**: 约1-2小时 (取决于GPU)

## 结果分析
训练完成后运行:
python scripts\evaluation\analyze_exp_f_results.py

========================================
准备就绪! 输入启动命令开始训练
========================================
