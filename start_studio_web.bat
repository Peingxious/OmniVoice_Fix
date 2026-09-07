@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem ===== 所有缓存/临时文件都锁在本文件夹内，不占用 C 盘 =====
set "HF_HOME=%~dp0.cache\huggingface"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "GRADIO_TEMP_DIR=%~dp0.cache\gradio"
set "TMP=%~dp0.cache\tmp"
set "TEMP=%~dp0.cache\tmp"
set "PYTHONIOENCODING=utf-8"

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到 .venv 虚拟环境
    pause
    exit /b 1
)

if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%TMP%" mkdir "%TMP%"

rem 自动清理可能残留占用的 8002 端口，防止端口冲突
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8002 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" >nul 2>&1
timeout /t 1 /nobreak >nul

rem 检测 Tailscale 状态并获取 IPv4 地址
set "TS_IP="
if exist "C:\Program Files\Tailscale\tailscale.exe" (
    for /f %%i in ('""C:\Program Files\Tailscale\tailscale.exe" ip -4"') do (
        set "TS_IP=%%i"
    )
)

echo =======================================================
echo   OmniVoice 分句配音工作室 (原生 Web 前端 + FastAPI)
echo.
echo   模型缓存 : %HF_HUB_CACHE%
echo   本机访问 : http://127.0.0.1:8002
if defined TS_IP (
echo   Tailscale: http://%TS_IP%:8002
)
echo.
echo   首次加载模型约需 20-30 秒，请稍候
echo   服务就绪后将自动打开浏览器
echo   关闭本窗口即可停止服务
echo =======================================================
echo.

rem 后台监听 8002 端口，就绪后自动打开浏览器
start "" powershell -WindowStyle Hidden -Command "for ($i=0; $i -lt 90; $i++) { if (Get-NetTCPConnection -LocalPort 8002 -State Listen -ErrorAction SilentlyContinue) { Start-Process 'http://127.0.0.1:8002'; break }; Start-Sleep -Seconds 2 }"

".venv\Scripts\python.exe" -m uvicorn omnivoice.server.app:app --host 0.0.0.0 --port 8002

echo.
echo 服务已停止。
pause
endlocal
