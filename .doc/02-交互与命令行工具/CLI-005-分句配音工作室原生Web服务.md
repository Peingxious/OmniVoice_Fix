# CLI-005: 分句配音工作室原生 Web 服务 (Native Web Studio)

## 用途与可观察行为
与 CLI-001 的 Gradio 综合控制台并列的**轻量原生 Web 工作室**，端口 **8002**。后端为解耦的 FastAPI REST/SSE 服务，前端为免构建的 Vue 3 + Tailwind CSS 单页应用（`web/index.html`）。专注"分句配音工作室"场景，提供极简、低开销、原生直下的逐行配音体验，无 Gradio Canvas 负担，单行 40px 紧凑布局，一键秒下载 ZIP。

核心能力：
1. **分句配音工作室 (Sentence Studio)**：粘贴长文 → 行优先清洗切分 → 批量按序生成独立编号音频 → 逐行试听/单句重跑 → 一键打包下载 ZIP（`01_xxx.wav` + `manifest.json` + `list.txt`）。
2. **声音克隆生成 (Clone Generate)**：上传参考音频 + 文本，或上传 `.pt` 声纹直接生成。
3. **音色设计生成 (Design Generate)**：按属性标签（性别/年龄/音调/口音/方言）生成。
4. **角色库管理**：列出 `voices/` 下已保存 `.pt`、上传/创建声纹、ASR 自动转写参考文本。

---

## 架构与触发方式

### 前端（免构建单页）
- 文件：[web/index.html](file:///d:/BaiduNetdiskWorkspace/OmniVoice/web/index.html)
- 技术：Vue 3（CDN 引入）+ Tailwind CSS，单文件、无需 `npm`/`build`，双击经 `start_studio_web.bat` 由 FastAPI 的 `StaticFiles` 直接托管。
- 通过 `fetch` 调用后端 `/api/*` 接口，长任务走 SSE 实时进度流。

### 后端（FastAPI）
- 入口模块：[omnivoice/server/app.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/server/app.py)，应用对象 `omnivoice.server.app:app`
- 全局懒加载单一 `OmniVoice` 模型与 `SentenceStudioEngine` 实例，进程内复用，避免重复加载显存。

### 启动方式
- 批处理一键启动：运行 [start_studio_web.bat](file:///d:/BaiduNetdiskWorkspace/OmniVoice/start_studio_web.bat)
  - 作用：设好本地缓存环境变量（`HF_HOME` 等）、自动释放 8002 占用、探测 Tailscale 并输出双端地址、后台探针就绪后自动打开浏览器、调用 `uvicorn omnivoice.server.app:app --host 0.0.0.0 --port 8002`。
  - `.venv` 缺失时同 `start_demo.bat` 一样自动引导安装（见 [CLI-004](CLI-004-Windows启动与缓存管理脚本.md)）。
- 命令行直接启动：
  ```bash
  python -m uvicorn omnivoice.server.app:app --host 0.0.0.0 --port 8002
  ```
- 访问地址：
  - 本机：`http://127.0.0.1:8002`
  - Tailscale 跨设备：`http://<Tailscale-IP>:8002`

---

## REST / SSE 接口清单

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/api/voices` | 列出 `voices/` 下所有已保存 `.pt` 声纹 |
| POST | `/api/split` | 行优先清洗切分文本（保留逗号连句），返回句子序列 |
| GET | `/api/audio/{task_id}/{filename}` | 取回单句生成的 WAV 文件 |
| GET | `/api/download_zip/{task_id}` | 原生直下该任务的 ZIP 压缩包流 |
| POST | `/api/synthesize_batch` | 逐行批量合成，SSE 实时推送进度 |
| POST | `/api/synthesize_single` | 单句原地重算，热替换音频并更新 ZIP |
| POST | `/api/upload_voice` | 上传 `.pt` 声纹至 `voices/` |
| POST | `/api/create_voice` | 由参考音频 + 文本创建声纹 `.pt` |
| GET | `/api/design_categories` | 音色设计属性分组 |
| GET | `/api/languages` | 支持语言列表（含 `Auto`） |
| POST | `/api/asr_transcribe` | Whisper 自动转写参考音频台词 |
| POST | `/api/clone_generate` | 声音克隆模式生成 |
| POST | `/api/design_generate` | 音色设计模式生成 |

---

## 与 CLI-001（Gradio 综合控制台）的差异

| 维度 | CLI-001 Gradio 控制台 (8001) | CLI-005 原生 Web 工作室 (8002) |
| :--- | :--- | :--- |
| 前端技术 | Gradio 组件化 | Vue 3 单页（免构建） |
| 定位 | 全能四页（克隆/工作室/角色/设计） | 聚焦分句配音 + 克隆/设计生成 |
| 渲染开销 | Gradio Canvas / WebSocket | 原生直下、单行 40px、无 Canvas |
| 后端形态 | Gradio 内部事件 | 解耦 FastAPI REST/SSE，可独立部署 |
| 打包下载 | ZIP（同契约） | ZIP（同契约，原生直下流） |

两者共用同一底层 `omnivoice/sentence_studio` 引擎与 `pack_sentence_audio_zip` 打包器，导出格式完全一致。

---

## 环境变量
- `OMNIVOICE_CHECKPOINT`：模型仓库名（默认 `k2-fsa/OmniVoice`）。
- `OMNIVOICE_DEVICE`：推理设备（默认 `get_best_device()` 自动选最优 GPU/CPU）。
- `HF_HOME` / `HF_HUB_CACHE`：由启动脚本锁定到仓库内 `.cache/huggingface`，避免污染系统盘。

## 实现入口与依赖
- **后端**：[omnivoice/server/app.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/server/app.py)、[omnivoice/server/__init__.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/server/__init__.py)
- **前端**：[web/index.html](file:///d:/BaiduNetdiskWorkspace/OmniVoice/web/index.html)
- **启动脚本**：[start_studio_web.bat](file:///d:/BaiduNetdiskWorkspace/OmniVoice/start_studio_web.bat)
- **复用引擎**：`omnivoice/sentence_studio/`（text_processor / engine / packager / models）
- **依赖**：`fastapi`, `uvicorn`, `sse-starlette`（SSE）, `torch`, `torchaudio`, `soundfile`

## 当前状态
- **状态**：`verified`（后端 API 与前端交互均经 `tests/test_server_api.py` 自动化测试通过；CLI-004 所述启动脚本实跑可用）
