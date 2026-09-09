# NEXT: 用户实机运行验证与跨仓库移植标准沉淀

## 目标
1. 启动 `start_demo.bat` 进行实机交互验证，确认长文本输入、逐行输出、单句重跑与 ZIP 打包全流程顺畅；
2. 在一台**干净 Windows 机器**上验证"可移植整合包（Ticket 04）"：仅拷代码 + `.cache`，双击 `start_demo.bat` 自动装环境、自动下模型、自动开网页；
3. 沉淀可移植标准，为其他语音项目（ChatTTS / CosyVoice / F5-TTS）输出统一移植指南。

## 启动条件
- Ticket 01、02、03 代码与测试均已完成并通过（状态已为 `verified`）；
- Ticket 04 可移植整合包已完成（`setup.bat` / `INSTALL.md` / `check_portable.bat` 就绪，本机 `uv sync` 实跑通过）。

## 所需上下文
- 运行脚本：`start_demo.bat`、`start_studio_web.bat`
- 浏览器访问地址：`http://127.0.0.1:8001`（Gradio）、`http://127.0.0.1:8002`（原生 Web 工作室）
- 模块路径：`omnivoice/sentence_studio/`
- 部署与校验：`INSTALL.md`、`check_portable.bat`、`setup.bat`

## 预期产出与验证
- 用户在 Web 界面上传音频、输入长文本，验证逐行呈现、单句重跑与一键打包下载；
- 干净机器双击 `start_demo.bat`：自动建 `.venv`（Python 3.12）→ `uv sync` → 首次经 HF 镜像下载模型 → 浏览器自动打开可用；跑 `check_portable.bat` 显示 `[通过]`。


