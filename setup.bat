@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem ======================================================================
rem  OmniVoice 一键环境引导（不依赖系统 Python）
rem   - 用 uv 在 .venv 里安装指定版本的 Python（uv 会自动下载该版本）
rem   - 然后 uv sync 安装全部依赖（含 CUDA 版 torch）
rem  被 start_demo.bat / start_studio_web.bat 自动调用；也可单独双击运行
rem ======================================================================

rem ----- 在这里定义要使用的 Python 版本 -----
set "PYTHON_VERSION=3.12"

echo ============================================
echo   OmniVoice 环境准备 (Python %PYTHON_VERSION%)
echo ============================================
echo.

rem ----- 1) 确保 uv 可用（没有就联网装一个）-----
where uv >nul 2>&1
if errorlevel 1 (
    echo [步骤1/3] 未检测到 uv，正在下载安装（需联网）...
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo [错误] uv 安装失败，请手动安装后重试：
        echo         pip install uv   或   见 https://docs.astral.sh/uv/getting-started/installation/
        pause
        exit /b 1
    )
    rem 让本会话能找到刚装好的 uv
    set "PATH=%USERPROFILE%\.local\bin;%LOCALAPPDATA%\uv\bin;%PATH%"
) else (
    echo [步骤1/3] uv 已存在，跳过安装。
)

rem ----- 2) 创建虚拟环境（uv 自动下载指定版本 Python，不碰系统 Python）-----
if not exist ".venv\Scripts\python.exe" (
    echo [步骤2/3] 正在创建虚拟环境 .venv，Python 版本 %PYTHON_VERSION%...
    uv venv --python %PYTHON_VERSION%
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败，请检查网络（需下载 Python %PYTHON_VERSION%）后重试。
        pause
        exit /b 1
    )
) else (
    echo [步骤2/3] .venv 已存在，跳过创建。
)

rem ----- 3) 安装/同步依赖（含 CUDA 版 torch，联网）-----
echo [步骤3/3] 正在安装依赖 (uv sync)...
uv sync
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络或代理后重试。
    pause
    exit /b 1
)

echo.
echo [完成] 环境已就绪：.venv (Python %PYTHON_VERSION%)
echo          可直接双击 start_demo.bat 或 start_studio_web.bat 启动。
echo.

endlocal
