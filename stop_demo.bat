@echo off
chcp 65001 >nul

echo 正在停止 OmniVoice 服务 (端口 8001) ...
set FOUND=0
for /f "tokens=5" %%a in ('netstat -ano ^| findstr LISTENING ^| findstr :8001') do (
    echo   结束进程 PID %%a
    taskkill /F /PID %%a >nul 2>&1
    set FOUND=1
)

if "%FOUND%"=="0" (
    echo 未发现运行中的服务。
) else (
    echo 已停止。
)
echo.
pause
