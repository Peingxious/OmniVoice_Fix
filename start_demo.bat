@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem ===== 所有缓存/临时文件都锁在本文件夹内，不占用 C 盘 =====
set "HF_HOME=%~dp0.cache\huggingface"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

rem 国内镜像加速：根目录存在 .hf_mirror 文件时，启用 HF 镜像（首次下载模型更快）
if exist "%~dp0.hf_mirror" (
    set "HF_ENDPOINT=https://hf-mirror.com"
)
set "GRADIO_TEMP_DIR=%~dp0.cache\gradio"
set "TMP=%~dp0.cache\tmp"
set "TEMP=%~dp0.cache\tmp"
set "PYTHONIOENCODING=utf-8"

if not exist ".venv\Scripts\python.exe" (
    echo [提示] 未检测到 .venv，将自动创建虚拟环境并安装依赖...
    echo.
    call "%~dp0setup.bat"
    if not exist ".venv\Scripts\python.exe" (
        echo [错误] 虚拟环境创建失败，请查看上方报错或手动运行 setup.bat
        pause
        exit /b 1
    )
)

if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%GRADIO_TEMP_DIR%" mkdir "%GRADIO_TEMP_DIR%"
if not exist "%TMP%" mkdir "%TMP%"

rem 自动清理可能残留占用的 8001 端口，防止端口冲突
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" >nul 2>&1
timeout /t 1 /nobreak >nul

rem 检测 Tailscale 状态并获取 IPv4 地址
set "TS_IP="
if exist "C:\Program Files\Tailscale\tailscale.exe" (
    for /f %%i in ('""C:\Program Files\Tailscale\tailscale.exe" ip -4"') do (
        set "TS_IP=%%i"
    )
)

echo ============================================
echo   OmniVoice 正在启动 (支持局域网与 Tailscale)
echo.
echo   模型缓存 : %HF_HUB_CACHE%
echo   本机访问 : http://127.0.0.1:8001
if defined TS_IP (
echo   Tailscale: http://%TS_IP%:8001
)
echo.
echo   首次加载模型约需 30-60 秒，请耐心等待
echo   浏览器会在服务就绪后自动打开
echo   关闭本窗口即可停止服务
echo ============================================
echo.

rem 后台监听 8001 端口，就绪后自动打开浏览器
start "" powershell -WindowStyle Hidden -Command "for ($i=0; $i -lt 90; $i++) { if (Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue) { Start-Process 'http://127.0.0.1:8001'; break }; Start-Sleep -Seconds 2 }"

".venv\Scripts\python.exe" -u -m omnivoice.cli.demo --ip 0.0.0.0 --port 8001

echo.
echo 服务已停止。
pause
endlocal
