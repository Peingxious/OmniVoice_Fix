# CLI-004: Windows 启动与缓存管理脚本

## 用途与可观察行为
针对 Windows 环境的自动化部署、启动与磁盘安全守护方案。由 `start_demo.bat` 与 `stop_demo.bat` 组成。其最核心的架构决策在于：**严格隔离所有临时目录与模型权重缓存，全数重定向到仓库内部的 `.cache/` 目录，杜绝污染与撑爆 Windows 系统盘（C盘）**。同时实现服务探活并在 8001 端口就绪后自动唤起默认浏览器。

## 关键脚本与行为分析

### 1. `start_demo.bat`
- **环境变量重定向**：
  ```bat
  set "HF_HOME=%~dp0.cache\huggingface"
  set "HF_HUB_CACHE=%HF_HOME%\hub"
  set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
  set "GRADIO_TEMP_DIR=%~dp0.cache\gradio"
  set "TMP=%~dp0.cache\tmp"
  set "TEMP=%~dp0.cache\tmp"
  ```
- **依赖检查**：自动探测 `.venv\Scripts\python.exe` 是否存在，若不存在则提示用户运行 `uv sync`。
- **目录自建**：自动确保 `%HF_HOME%`, `%GRADIO_TEMP_DIR%`, `%TMP%` 存在。
- **后台端口探针**：启动非阻塞 PowerShell 后台任务，轮询 `http://127.0.0.1:8001` 是否开始监听，一旦就绪立即调用系统默认浏览器打开界面。
- **Tailscale 外部访问与全网卡监听**：
  - 自动调用 `tailscale ip -4` 检测 Tailscale 虚拟网卡 IP，并在启动控制台输出本机访问（`http://127.0.0.1:8001`）与 Tailscale 远程访问（如 `http://100.91.115.73:8001`）双地址。
  - 调用 `.venv\Scripts\python.exe -u -m omnivoice.cli.demo --ip 0.0.0.0 --port 8001` 监听所有网卡，允许局域网与 Tailscale 异地设备（手机、平板、笔记本）即开即用。

### 2. `stop_demo.bat`
- 查询监听 8001 端口的 PID 并通过 `taskkill /F /PID` 强制关闭，实现干净停机。

## 实现入口与依赖
- **启动脚本**：[start_demo.bat](file:///d:/BaiduNetdiskWorkspace/OmniVoice/start_demo.bat)
- **停止脚本**：[stop_demo.bat](file:///d:/BaiduNetdiskWorkspace/OmniVoice/stop_demo.bat)
- **依赖**：Windows CMD, PowerShell, 本地 `.venv`

## 当前状态
- **状态**：`verified` (在当前 Windows 机器上已实际执行，并成功隔离模型缓存)
