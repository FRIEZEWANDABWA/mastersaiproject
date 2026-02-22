@echo off
cd /d C:\websites\mastersaiproject
set PYTHONUTF8=1
title MSc AI — Maize Disease Classifier

:menu
cls
echo ============================================================
echo   MSc AI — Maize Disease Classifier
echo   Candidate : WANDABWA Frieze  (ST62/55175/2025)
echo   Supervisor : Dr. Richard Rimiru
echo   Institution: Open University of Kenya (OUK)
echo ============================================================
echo.
echo   [1] Project Status   (FRIENDS framework compliance check)
echo   [2] Train Model      (ResNet50 two-phase training)
echo   [3] Evaluate Model   (confusion matrix + LIME XAI)
echo   [4] Grad-CAM XAI     (visual explanation for one image)
echo   [5] Jupyter Notebook (data exploration)
echo   [6] Exit
echo.
set /p choice="   Enter choice [1-6]: "

if "%choice%"=="1" goto status
if "%choice%"=="2" goto train
if "%choice%"=="3" goto evaluate
if "%choice%"=="4" goto gradcam
if "%choice%"=="5" goto jupyter
if "%choice%"=="6" goto end
goto menu

:status
cls
echo Running project status check...
python src\project_status.py
pause
goto menu

:train
cls
echo Starting model training (ResNet50, two-phase)...
echo WARNING: This requires at least 50 images per class for good results.
echo Current dataset stats shown in status check (option 1).
echo.
python src\train.py
pause
goto menu

:evaluate
cls
echo Running model evaluation...
python src\evaluate.py
pause
goto menu

:gradcam
cls
set /p imgpath="   Enter path to maize image: "
python src\explainability.py --image "%imgpath%"
pause
goto menu

:jupyter
cls
echo Launching Jupyter Notebook...
start jupyter notebook notebooks\
goto menu

:end
echo Goodbye!
exit /b 0
