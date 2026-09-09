# CURRENT: 分句配音工作室（Sentence Studio）可移植蓝图与端到端系统建设

## 目标与完成条件
- **目标**：
  1. 落地规格说明 `.project/spec.md` 与分步探路票据（Tickets 01 ~ 03）。
  2. 实现高内聚、零 UI 强依赖的核心文本清洗分句器与规范化音频打包器（Ticket 01）。
  3. 实现参考音频全局复用、批量顺序生成与单句原地热重算的调度引擎（Ticket 02）。
  4. 在 WebUI 中集成“分句配音工作室”独立 Tab，支持逐行层级展示、独立播放、单句修改重算与一键全部打包下载（Ticket 03）。
  5. 保持原有全部功能 100% 独立兼容，并将架构提炼为可整体克隆移植的对照蓝图。

- **完成条件**：
  1. [omnivoice/sentence_studio/text_processor.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/sentence_studio/text_processor.py)：
     - 实现 `clean_and_split_text`，覆盖换行优先、空行剥离、无换行终结标点兜底。
  2. [omnivoice/sentence_studio/packager.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/sentence_studio/packager.py)：
     - 实现 `pack_sentence_audio_zip`，按 `01_xxx.wav`, `02_xxx.wav` 严格有序打包，附带 `manifest.json` 与 `list.txt`。
  3. [omnivoice/sentence_studio/engine.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/sentence_studio/engine.py)：
     - 实现 `SentenceStudioEngine`，单次提取声纹特征全局复用，支持 `regenerate_sentence` 局部更新。
  4. [omnivoice/sentence_studio/ui.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/sentence_studio/ui.py) & [omnivoice/cli/demo.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/cli/demo.py)：
     - 成功挂载“✂️ 分句配音工作室”Tab，完整集成“选择已训练角色 (PT 文件)”下拉列表（含 `杨总.pt` 等已有角色）、一键刷新与声纹信息卡片展示；双语界面完全支持，无缝流式输出。
  5. [tests/test_sentence_studio.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/tests/test_sentence_studio.py)：
     - 包含 10 个独立单元测试，涵盖分句、极端格式、打大包、引擎重算与 `.pt` 角色加载复用，全部测试通过。
  6. [.doc/02-交互与命令行工具/CLI-001-交互式网页演示服务.md](file:///d:/BaiduNetdiskWorkspace/OmniVoice/.doc/02-交互与命令行工具/CLI-001-交互式网页演示服务.md)：
     - 功能地图同步更新。

## 当前状态与精确接续点
- **状态**：`verified` (所有票据 Tickets 01~03 均已完成，全套单元测试通过；新增前后端分离原生 Web 架构 `omnivoice/server/` 与 `web/index.html`，通过 API 与界面测试)。
- **工作产物**：
  - `omnivoice/sentence_studio/` 核心分句推理引擎与打包器
  - `omnivoice/server/` 解耦的 FastAPI 独立 REST/SSE 后端
  - `web/index.html` Vue 3 + Tailwind CSS 原生单页前端（免构建、单行40px、原生直下）
  - `start_studio_web.bat` 原生 Web 工作室启动脚本（端口 8002）
  - `tests/test_sentence_studio.py` & `tests/test_server_api.py` 自动化测试
  - `.project/spec.md`
- **可移植整合包（Ticket 04，已 complete）**：
  - `setup.bat`：用 `uv` 自管 Python 3.12 建 `.venv`，不依赖系统 Python；缺失 `uv` 自动安装，`uv sync` 装依赖
  - `start_demo.bat` / `start_studio_web.bat`：`.venv` 缺失时自动 `call setup.bat` 引导
  - `.hf_mirror`（空文件开关）：启用 HF 国内镜像加速首次模型下载
  - `check_portable.bat` / `check_portable.py`：可移植性自检，缺什么给下载指引
  - `INSTALL.md`：系统要求、双击即用、国内镜像、手动分步、可移植/打包、常见问题
- **精确接续点**：
  1. 运行 `start_studio_web.bat` 体验全新的 **原生 Vue 3 单行极简 Web 工作室**（端口 8002，无 Canvas 负担，一键秒下载）。
  2. 运行 `start_demo.bat` 继续使用全功能 Gradio 多 Tab 综合控制台（端口 8001）。


