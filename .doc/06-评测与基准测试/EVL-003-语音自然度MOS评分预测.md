# EVL-003: 语音自然度 MOS 评分预测

## 用途与可观察行为
无需招募人工听音测试即可自动化量化合成语音的自然度、平滑度与声学质量。采用业界经典的神经网络评分系统 `UTMOS` (Universal Speech Quality Assessment)，输入待评测音频，直接预测输出相当于平均主观意见分（MOS, 1.0 ~ 5.0 分）。

## 触发方式与 API
```python
from omnivoice.eval.mos.utmos import compute_utmos

score = compute_utmos("synthesized.wav")
print(f"UTMOS 预测得分: {score:.2f}")
```

## 核心价值
- 提供客观、稳定、可重复的语音品质度量。
- 用于在推理优化实验中量化去噪步数（如 16 步 vs 32 步）或加速内核（FlashInfer vs SDPA）对听感保真度的实际影响。

## 实现入口与依赖
- **评测接口**：[omnivoice/eval/mos/utmos.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/eval/mos/utmos.py)
- **模型实现**：[omnivoice/eval/models/utmos.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/eval/models/utmos.py)
- **依赖**：`torch`, `torchaudio`, `librosa`

## 当前状态
- **状态**：`verified` (可独立加载 UTMOS 预训练权重并批量打分)
