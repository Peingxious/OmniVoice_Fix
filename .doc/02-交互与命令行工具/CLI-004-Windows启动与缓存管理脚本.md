# CLI-004: Windows 启动、缓存管理与一键环境引导脚本

## 用途与可观察行为
针对 Windows 环境的自动化部署、启动、磁盘安全守护与**一键环境引导**方案。其最核心的架构决策在于：**严格隔离所有临时目录与模型权重缓存，全数重定向到仓库内部的 `.cache/` 目录，杜绝污染与撑爆 Windows 系统盘（C盘）**；并进一步提供 `setup.bat` 让新机器**不依赖系统 Python** 即可自动装好运行环境，配合 `.hf_mirror` 开关实现国内模型下载加速。

## 关键脚本与行为分析

### 1. `start_demo.bat`（Gradio 综合控制台，端口 8001）
- **环境变量重定向**：
  ```bat
  set "HF_HOME=%~dp0.cache\huggingface"
  set "HF_HUB_CACHE=%HF_HOME%\hub"
  set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
  set "GRADIO_TEMP_DIR=%~dp0.cache\gradio"
  set "TMP=%~dp0.cache\tmp"
  set "TEMP=%~dp0.cache\tmp"
  ```
- **国内镜像加速**：根目录存在空文件 `.hf_mirror` 时自动 `set HF_ENDPOINT=https://hf-mirror.com`，首次下载模型走国内镜像（详见下方脚本清单）。
- **依赖检查与自动引导**：自动探测 `.venv\Scripts\python.exe`，若不存在则**自动 `call setup.bat`** 装好环境再继续（不再仅提示手动 `uv sync`）。
- **目录自建**：自动确保 `%HF_HOME%`, `%GRADIO_TEMP_DIR%`, `%TMP%` 存在。
- **后台端口探针**：启动非阻塞 PowerShell 后台任务，轮询 `http://127.0.0.1:8001` 是否开始监听，一旦就绪立即调用系统默认浏览器打开界面。
- **Tailscale 外部访问与全网卡监听**：
  - 自动调用 `tailscale ip -4` 检测 Tailscale 虚拟网卡 IP，并在启动控制台输出本机访问（`http://127.0.0.1:8001`）与 Tailscale 远程访问（如 `http://100.91.115.73:8001`）双地址。
  - 调用 `.venv\Scripts\python.exe -u -m omnivoice.cli.demo --ip 0.0.0.0 --port 8001` 监听所有网卡，允许局域网与 Tailscale 异地设备（手机、平板、笔记本）即开即用。

### 2. `start_studio_web.bat`（原生 Web 工作室，端口 8002）
- 与 `start_demo.bat` 同构的缓存隔离、`.hf_mirror` 镜像、`.venv` 自动引导、端口探针与 Tailscale 检测逻辑，端口改为 **8002**。
- 最终调用 `.venv\Scripts\python.exe -m uvicorn omnivoice.server.app:app --host 0.0.0.0 --port 8002` 启动 FastAPI 原生 Web 工作室（详见 [CLI-005](CLI-005-分句配音工作室原生Web服务.md)）。

### 3. `stop_demo.bat`
- 查询监听 8001 端口的 PID 并通过 `taskkill /F /PID` 强制关闭，实现干净停机。

### 4. `setup.bat`（一键环境引导，不依赖系统 Python）
把"新机器跑起来"收敛为单条命令，三步完成：
1. **确保 `uv`**：`where uv` 找不到则联网执行 `irm https://astral.sh/uv/install.ps1` 安装，并把其安装目录临时加入 `PATH`。
2. **创建 `.venv`**：`uv venv --python 3.12` —— `uv` 自动下载**指定版本 Python 3.12** 装入 `.venv`，完全不碰系统 Python（版本在脚本顶部 `PYTHON_VERSION=3.12` 一处定义）。
3. **同步依赖**：`uv sync` 安装全部依赖（含 CUDA 12.8 版 `torch`）。
- 被 `start_demo.bat` / `start_studio_web.bat` 在 `.venv` 缺失时自动 `call`，也可单独双击运行。

### 5. `check_portable.bat` / `check_portable.py`（可移植性自检）
- `check_portable.bat` 自动选用 `.venv` 或系统 `python` 运行 `check_portable.py`。
- `check_portable.py` 仅用 Python 标准库，逐项检查：
  - `.venv` 是否存在；
  - 主模型 `k2-fsa/OmniVoice`（model.safetensors / audio_tokenizer / config.json）是否缓存完整；
  - ASR 模型 `openai/whisper-large-v3-turbo` 是否缓存；
  - 若已装 `torch`，进一步检测 CUDA 是否可用。
- **缺什么当场给出下载地址/命令**（含 `uv sync`、HF 镜像、模型 `huggingface-cli download`），结论为 `[通过] / [失败] / [警告]`。

### 6. `.hf_mirror`（镜像加速开关，空文件）
- 仓库根目录的空文件；存在时 `start_demo.bat` / `start_studio_web.bat` 自动启用 HF 国内镜像。已被 `.gitignore` 忽略，可随整合包分发而不进版本库。无此文件则直连官方 HF。

### 7. `INSTALL.md`（独立部署说明）
- 系统要求、双击即用流程、国内镜像开启、手动分步安装、校验脚本用法、可移植/打包方式、常见问题与文件清单，完整说明整合包的使用。

## 实现入口与依赖
- **启动脚本**：[start_demo.bat](file:///d:/BaiduNetdiskWorkspace/OmniVoice/start_demo.bat)、[start_studio_web.bat](file:///d:/BaiduNetdiskWorkspace/OmniVoice/start_studio_web.bat)
- **停止脚本**：[stop_demo.bat](file:///d:/BaiduNetdiskWorkspace/OmniVoice/stop_demo.bat)
- **环境引导**：[setup.bat](file:///d:/BaiduNetdiskWorkspace/OmniVoice/setup.bat)
- **可移植性自检**：[check_portable.bat](file:///d:/BaiduNetdiskWorkspace/OmniVoice/check_portable.bat)、[check_portable.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/check_portable.py)
- **部署说明**：[INSTALL.md](file:///d:/BaiduNetdiskWorkspace/OmniVoice/INSTALL.md)
- **依赖**：Windows CMD, PowerShell, `uv`（自动安装）, 本地 `.venv`

## 当前状态
- **状态**：`verified`（`start_demo.bat` 实跑通过；`setup.bat` 本机 `uv sync` 实跑通过；`check_portable.bat` 全项 `[通过]`；模型缓存与缓存隔离策略已生效）

