@echo off
REM Temporary set PATH for Python and conda
REM Adjust paths as needed based on your installation

REM Add Python to PATH (example path, replace with actual)
set PATH=%PATH%;C:\Users\18453\AppData\Local\Programs\Python\Python39

REM Add conda to PATH (example path, replace with actual, e.g., C:\ProgramData\Anaconda3\Scripts or C:\Users\18453\miniconda3\Scripts)
set PATH=%PATH%;C:\ProgramData\Anaconda3\Scripts

REM Activate conda environment
call conda activate yolo_test

REM Run the pipeline
python run_preprocessing_pipeline.py --yes

pause
