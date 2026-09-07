# TRN-002: 检查点保存与 EMA 恢复

## 用途与可观察行为
在长时间预训练与微调过程中保障容灾与平滑收敛。训练器异步维护模型权重的指数移动平均（EMA, Exponential Moving Average, 衰减率通常为 0.9999），在保存 Checkpoint 时同时记录原始权重与 EMA 权重。在推理阶段或评测前，可一键切换或恢复 EMA 状态，显著提升合成语音的音色平滑度与抗杂音能力。

## 关键流程与功能
1. **EMA 权重追踪**：
   - 每个优化步更新：$W_{\text{EMA}} = \beta \cdot W_{\text{EMA}} + (1 - \beta) \cdot W_{\text{model}}$
2. **检查点组织**：
   - 包含当前步数、优化器状态、LR 调度器状态、RNG 随机种子及模型权重。
   - 保留最近 $K$ 个 Checkpoint，自动清理更早的冗余检查点。
3. **断点恢复 (Resume)**：
   - 传入 `--resume_from_checkpoint` 自动恢复完整训练上下文，保障数据流式索引与优化步数无缝衔接。

## 实现入口与依赖
- **检查点管理器**：[omnivoice/training/checkpoint.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/training/checkpoint.py)
- **配置抽象**：[omnivoice/training/config.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/training/config.py)
- **依赖**：`torch.save`, `torch.load`

## 当前状态
- **状态**：`verified` (包含完整的 EMA 状态字典映射与保存逻辑)
