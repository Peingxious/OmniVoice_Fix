# 04 - 可移植整合包与一键环境引导 (Portable Packaging & Auto-Bootstrap)

**What to build:**
让 OmniVoice 仓库成为可直接搬运、双击即用的"整合包"，降低新机器部署门槛：
1. **一键环境引导 `setup.bat`**：
   - 用 `uv` 在 `.venv` 内安装**指定版本 Python（3.12）**，完全不依赖系统 Python（`uv venv --python 3.12` 自动下载该版本）；
   - 缺失 `uv` 时自动联网安装（`irm https://astral.sh/uv/install.ps1`）；
   - 随后 `uv sync` 安装全部依赖（含 CUDA 12.8 版 torch）。
2. **启动脚本自动接环境**：`start_demo.bat` / `start_studio_web.bat` 在 `.venv` 缺失时自动 `call setup.bat` 装好再启动，不再报"未找到 venv"。
3. **国内镜像加速**：根目录存在空文件 `.hf_mirror` 时，两个启动脚本自动 `set HF_ENDPOINT=https://hf-mirror.com`，首次下载模型走国内镜像。
4. **可移植性自检 `check_portable.bat`（`check_portable.py`）**：纯标准库，逐项检查 `.venv`/主模型/Whisper/CUDA，缺什么当场给出下载地址与命令，结论为 `[通过]/[失败]/[警告]`。
5. **独立安装说明 `INSTALL.md`**：系统要求、双击即用流程、国内镜像、手动分步、校验脚本、可移植/打包方式、常见问题、文件清单。

**Blocked by:** 无（独立于 Sentence Studio 功能，属于发行/部署层）

**Status:** complete

**Use check:**
在一台**仅拷入代码 + `.cache`**（无 `.venv`、无系统 Python 管理权）的新 Windows 机器上：
1. 双击 `start_demo.bat`；
2. 脚本自动检测 `uv` → 装 `uv` → `uv venv --python 3.12` → `uv sync`；
3. 首次加载模型时经 HF 镜像自动下载 `k2-fsa/OmniVoice` 与 `whisper-large-v3-turbo` 并缓存到 `.cache/huggingface`；
4. 浏览器自动打开 8001 界面，可正常克隆/设计/生成；
5. 另起一台机双击 `check_portable.bat`，报告 `[通过]` 或明确列出缺失项与下载方式。

- [x] `setup.bat` 用 uv 自管 Python 3.12 建 `.venv`，不碰系统 Python
- [x] `start_*.bat` 在 `.venv` 缺失时自动引导安装
- [x] `.hf_mirror` 开关启用 HF 国内镜像加速模型下载
- [x] `check_portable.bat` / `.py` 自检并给出缺失项下载指引
- [x] `INSTALL.md` 编写完整部署与可移植说明
- [x] `setup.bat` 在本机实跑通过（`uv sync` 增量同步成功）
