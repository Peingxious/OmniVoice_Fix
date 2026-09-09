@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem 与 start_demo.bat 保持一致，让脚本能找到本地模型缓存
set "HF_HOME=%~dp0.cache\huggingface"
set "PYTHONIOENCODING=utf-8"

echo 正在检查 OmniVoice 可移植环境...
echo.

rem 优先用本仓库的 .venv，其次用系统 python
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" check_portable.py
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [错误] 未找到 Python。请先安装 Python 3.10~3.12 或运行 uv sync 重建环境。
        echo 下载: https://www.python.org/downloads/
        echo.
        pause
        exit /b 1
    ) else (
        python check_portable.py
    )
)

echo.
pause
endlocal
