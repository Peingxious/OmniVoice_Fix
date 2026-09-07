# OmniVoice Durable Memory

本文件记录本仓库稳定、有验证证据且无法通过单纯代码分析廉价推导的关键经验。

## 运行环境与磁盘管理
- **虚拟环境**：本地已配置 `.venv` 虚拟环境，主 Python 入口为 `.venv\Scripts\python.exe`。
- **缓存隔离约束**：
  - Windows 环境下系统盘空间极为宝贵。`start_demo.bat` 已强制设定：
    - `HF_HOME=%~dp0.cache\huggingface`
    - `HF_HUB_CACHE=%HF_HOME%\hub`
    - `GRADIO_TEMP_DIR=%~dp0.cache\gradio`
    - `TMP=%~dp0.cache\tmp`
  - 启动 Python 脚本或 CLI 推理时，若脱离 bat 脚本独立执行，必须自觉设置上述环境变量，防止默认模型权重下载占用 `C:\Users\Administrator\.cache`。

## 模型与算法特性
- **采样率标准**：音频标记器 HiggsAudioV2 输出采样率为 24kHz，`OmniVoice.generate()` 返回格式为单声道、24kHz 的 numpy 数组列表。
- **参考音频长度建议**：Voice Cloning 推荐使用 3-10 秒清晰参考音频。过短影响声纹特征提取，过长会拖慢推理速度且可能稀释音色特征。
- **Prompt 固化机制**：使用 `model.create_voice_clone_prompt()` 提取并保存为 `.pt` 文件（格式版本 `_VOICE_CLONE_PROMPT_FORMAT_VERSION = 1`），保存在 `voices/` 目录供后续直接加载，跳过 ASR 识别耗时。
- **注意力后端分级**：
  - 首选：FlashInfer（需匹配 CUDA 版本的特定 wheel，加速 2.0x-2.6x）。
  - 备选：PyTorch SDPA（默认且跨平台稳定）。
  - 训练：flex_attention，已内置 `_autocast_flex_attention` 解决 fp32 精度回退与性能损失。
- **多语言与音色设计限制**：
  - 语音克隆覆盖 600+ 语言。
  - 音色设计（Voice Design）主要在中英文数据上训练，英文支持属性较丰富，中文支持各地方言（如四川话、陕西话、粤语等），低资源语言使用音色设计可能存在稳定性差异。
