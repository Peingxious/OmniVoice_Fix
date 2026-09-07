# EVL-001: 词错误率与多模型 ASR 基准评测

## 用途与可观察行为
衡量生成语音的可懂度（Intelligibility）与内容准确性。将合成音频通过业界主流的预训练 ASR 模型（如 FunASR SenseVoice、OpenAI Whisper、HuBERT、MiniMax API 等）识别为文本，经过专用的多语言文本归一化与标点剥离（Text Normalization），与 Ground Truth 文本比对并计算字/词错误率（WER/CER）。

## 触发方式与示例
```bash
# 评测脚本运行
bash examples/run_eval.sh
```
- 或针对特定基准调用评测模块：
  ```python
  from omnivoice.eval.wer.fleurs import evaluate_fleurs
  from omnivoice.eval.wer.seedtts import evaluate_seedtts
  ```

## 核心技术点
1. **多基准支持**：针对 SEED-TTS 基准、FLEURS 多语言基准等均有独立适配与评测入口。
2. **多语言文本规整适配**：
   - [omnivoice/eval/wer/text_norm_omni.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/eval/wer/text_norm_omni.py) 支持统一的 Unicode 规范化、繁简体转换（`zhconv`）与通用标点过滤（`punctuations.lst`）。
3. **主流 ASR 引擎封装**：
   - SenseVoice 极速高精中文/英文评测。
   - Whisper 多语言长音频评测。

## 实现入口与依赖
- **评测引擎模块**：[omnivoice/eval/wer/](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/eval/wer/) (`sensevoice.py`, `seedtts.py`, `fleurs.py`, `text_norm_omni.py`)
- **依赖**：`jiwer`, `funasr`, `zhconv`, `zhon`, `unidecode`
- **文档指南**：[docs/evaluation.md](file:///d:/BaiduNetdiskWorkspace/OmniVoice/docs/evaluation.md)

## 当前状态
- **状态**：`verified` (包含完整的基准评测脚本与规范化正则表)
