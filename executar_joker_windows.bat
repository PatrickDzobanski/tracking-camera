@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%PYTHON%" (
    "%PYTHON%" "%~dp0Interface.py" %*
) else (
    python "%~dp0Interface.py" %*
)
if errorlevel 1 (
    echo.
    echo Falha ao executar interface. Verifique a instalacao de Python, OpenCV e NumPy.
    pause
)
endlocal
