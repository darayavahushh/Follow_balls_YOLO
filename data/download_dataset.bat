@echo off

set DATA_DIR=data
set ZIP_FILE=%DATA_DIR%\V1.zip
set EXTRACT_DIR=%DATA_DIR%\V1

echo Creating data directory...
if not exist %DATA_DIR% mkdir %DATA_DIR%

echo Installing huggingface_hub...
python -m pip install huggingface_hub

echo Downloading dataset...
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Adit-jain/Soccana_player_ball_detection_v1', local_dir='data', repo_type='dataset')"

echo Extracting V1.zip...
if not exist %EXTRACT_DIR% mkdir %EXTRACT_DIR%

powershell -Command "Expand-Archive -Force '%ZIP_FILE%' '%EXTRACT_DIR%'"

echo Done. Dataset available in %EXTRACT_DIR%
pause