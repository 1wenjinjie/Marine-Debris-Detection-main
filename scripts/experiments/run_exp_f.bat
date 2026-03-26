@echo off
echo ========================================
echo Exp-F: 7ch + SIoU Training
echo ========================================
echo Target: Break F1=0.70 threshold
echo Config: 7ch (RGB+NIR+RedEdge) + SIoU
echo ========================================
echo.
python scripts\experiments\train_exp_f_7ch_siou.py
echo.
echo ========================================
echo Training complete!
echo Results: runs_marida\exp_f_7ch_siou\
echo ========================================
pause
