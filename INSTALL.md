# OmniVoice 安装与部署指南（Windows）

本指南说明如何在一台**新机器**上把 OmniVoice 跑起来。设计目标：**双击即用、不依赖系统 Python、首次运行自动下载模型**。

---

## 0. 系统要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 10/11（64 位） |
| 显卡 | NVIDIA 显卡（本整合包内置 CUDA 12.8 版 torch，需匹配的驱动；建议驱动版本 ≥ 527） |
| 内存 | ≥ 8 GB（克隆/设计模式） |
| 磁盘 | 代码 + 环境约 5~8 GB；模型权重约 4~5 GB（首次自动下载） |
| 网络 | 首次运行需联网（装环境 + 下模型）；可离线使用后续启动 |

> 无独显的机器也能跑，但会退化为 CPU，速度极慢且可能显存不足。如需在无 GPU 机器上运行，请手动修改 `pyproject.toml` 的 torch 来源后再 `uv sync`。

---

## 1. 最快方式：双击启动（自动装环境 + 自动下模型）

直接双击下面任意一个脚本，**第一次会自动完成一切**：

- `start_demo.bat` —— Gradio 网页界面（语音克隆 / 音色设计），端口 8001
- `start_studio_web.bat` —— 分句配音工作室（原生 Web + FastAPI），端口 8002

脚本会自动：
1. 检查 `.venv`，没有就用 `uv` 装一个 **Python 3.12** 虚拟环境（不碰系统 Python）；
2. `uv sync` 安装全部依赖（含 CUDA 版 torch）；
3. 启动服务，首次加载模型时**自动从 HuggingFace 下载**权重（约 4~5 GB）并缓存到 `.cache/huggingface`；
4. 服务就绪后自动打开浏览器。

> ⏱ 首次运行因为要下载环境依赖 + 模型，可能要等几分钟；之后启动只需加载模型（约 20~60 秒）。

---

## 2. 国内下载加速（重要）

HuggingFace 官方源在国内可能很慢或超时。开启镜像的方法（任选其一）：

**方法 A（推荐，零改动）**：在仓库根目录新建一个**空文件**取名 `.hf_mirror`，
两个 `start_*.bat` 检测到它就会自动走 `https://hf-mirror.com` 镜像。

**方法 B（手动）**：启动前在命令行执行
```bat
set HF_ENDPOINT=https://hf-mirror.com
```

> 注：`.hf_mirror` 只是一个"开关文件"，内容为空即可；它已被 `.gitignore` 忽略，不会进版本库。

---

## 3. 手动分步安装（排查问题时用）

如果自动启动失败，可手动跑 `setup.bat`（它等价于第 1 步的自动流程）：

```bat
setup.bat
```

里面做的事：
1. 检测 `uv`；没有则联网安装（`irm https://astral.sh/uv/install.ps1`）。
2. `uv venv --python 3.12` —— 在 `.venv` 创建指定版本 Python（uv 自动下载，不依赖系统 Python）。
3. `uv sync` —— 安装依赖。

模型仍由启动脚本在**首次运行时**自动下载。若想**提前**把模型下好（离线机器预置）：

```bat
set HF_ENDPOINT=https://hf-mirror.com
.venv\Scripts\python.exe -m huggingface_hub.commands.huggingface_cli download k2-fsa/OmniVoice
.venv\Scripts\python.exe -m huggingface_hub.commands.huggingface_cli download openai/whisper-large-v3-turbo
```

模型会落到 `.cache/huggingface/hub`，之后离线也能用。

---

## 4. 校验当前机器是否就绪

双击 `check_portable.bat`，它会逐项检查 `.venv`、主模型、Whisper ASR、CUDA，
缺什么就**当场给出下载地址/命令**，最后给出 `[通过] / [失败] / [警告]` 结论。

```
check_portable.bat
```

---

## 5. 可移植 / 打包说明

仓库代码本身**不含**模型与 `.venv`（均被 `.gitignore` 忽略）。常见两种搬运方式：

| 方式 | 包含 | 目标机操作 |
|---|---|---|
| 整包拷贝 | 代码 + `.cache` + `.venv` | 同配置 Windows 机直接双击 `start_*.bat` 离线可用 |
| 最稳妥 | 代码 + `.cache` | 目标机双击 `start_*.bat`，自动 `uv sync` 重建 `.venv`（需联网一次） |

可省略不带（临时文件）：`.cache/gradio`、`.cache/tmp`。

---

## 6. 常见问题

- **首次启动卡在下载 / 超时**：见第 2 节开启 HF 镜像；或确认网络/代理可用。
- **换到无 GPU 的机器推理报错**：本包 torch 是 CUDA 版；无独显需改 `pyproject.toml` 用 CPU 版 torch 后重跑 `setup.bat`。
- **提示找不到 Python**：正常——脚本用 `uv` 自管的 Python，不会找系统 Python；若 `uv` 也没装上，按第 3 节手动安装。
- **端口被占用**：8001/8002 被占时脚本会自动尝试释放；仍失败请手动改 `.bat` 里的端口号。

---

## 7. 文件清单（本整合包）

| 文件 | 作用 |
|---|---|
| `start_demo.bat` | 启动 Gradio 网页界面（8001） |
| `start_studio_web.bat` | 启动分句配音工作室（8002） |
| `setup.bat` | 一键安装 Python 3.12 虚拟环境 + 依赖 |
| `check_portable.bat` | 可移植性自检（缺什么给下载地址） |
| `.hf_mirror` | （可选，自建空文件）开启 HF 国内镜像 |
| `.cache/huggingface` | 模型权重缓存（首次运行自动生成） |
| `.venv/` | Python 虚拟环境（首次运行自动生成） |
