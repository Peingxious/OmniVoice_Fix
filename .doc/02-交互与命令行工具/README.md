# 02-交互与命令行工具

## 边界与职责
本功能块统管 OmniVoice 的用户交互界面、命令行调用端与操作系统级运行部署脚本，支撑从本地单机体验、批量工业化推理到服务常驻的全场景需求。

- **代码前缀**：`CLI`
- **主要受众**：应用工程师、系统运维人员与最终使用用户

## 功能文件索引
- [CLI-001-交互式网页演示服务.md](CLI-001-交互式网页演示服务.md)：Gradio Web 界面、双语 i18n 切换、实时录音、音色持久化与在线试听。
- [CLI-002-单条合成命令行工具.md](CLI-002-单条合成命令行工具.md)：`omnivoice-infer` 单次推理命令行，支持完整声学参数传入与文件输出。
- [CLI-003-多卡分布式批量推理.md](CLI-003-多卡分布式批量推理.md)：`omnivoice-infer-batch` 基于 JSONL 任务清单的多卡并行与吞吐基准测速。
- [CLI-004-Windows启动与缓存管理脚本.md](CLI-004-Windows启动与缓存管理脚本.md)：`start_demo.bat`/`start_studio_web.bat`/`stop_demo.bat` 启动与缓存隔离脚本，外加 `setup.bat` 一键环境引导（uv 自管 Python 3.12）、`check_portable.bat` 自检与 `.hf_mirror` 国内镜像开关，保障系统盘（C盘）不被大模型权重挤爆。
- [CLI-005-分句配音工作室原生Web服务.md](CLI-005-分句配音工作室原生Web服务.md)：端口 8002 的 FastAPI + Vue 3 免构建原生 Web 工作室，聚焦分句配音、克隆/设计生成与 REST/SSE 接口。
