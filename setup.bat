@echo off
setlocal
cd /d %~dp0
if "%PYTHON_BIN%"=="" set PYTHON_BIN=python
if "%VENV_DIR%"=="" set VENV_DIR=.venv
%PYTHON_BIN% -m venv %VENV_DIR%
call %VENV_DIR%\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo Environment ready: %CD%\%VENV_DIR%
echo Activate with: %VENV_DIR%\Scripts\activate.bat
echo Run builder with: python run.py --output-root .\out
endlocal
