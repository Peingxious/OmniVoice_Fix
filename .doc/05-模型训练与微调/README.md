# 05-模型训练与微调

## 边界与职责
本功能块覆盖 OmniVoice 的预训练（Pre-training）、全量微调（Full Fine-tuning）以及基于 PEFT 的轻量级 LoRA 微调，涉及多码本加权 Loss 计算、EMA 权重追踪、DeepSpeed ZeRO-2 分布式协同与 LoRA 权重一键合并。

- **代码前缀**：`TRN`
- **主要受众**：深度学习算法研究员、模型调优工程师

## 功能文件索引
- [TRN-001-扩散语言模型预训练与训练器.md](TRN-001-扩散语言模型预训练与训练器.md)：多码本加权交叉熵损失、扩散去噪目标训练与 Trainer 主循环。
- [TRN-002-检查点保存与EMA恢复.md](TRN-002-检查点保存与EMA恢复.md)：指数移动平均 (EMA) 跟踪、定期检查点存储与热启动恢复。
- [TRN-003-分布式训练与DeepSpeed集成.md](TRN-003-分布式训练与DeepSpeed集成.md)：ZeRO-2 显存优化、Accelerate 分布式封装与多卡 DDP 通信钩子。
- [TRN-004-LoRA参数高效微调与权重合并.md](TRN-004-LoRA参数高效微调与权重合并.md)：PEFT LoRA 适配器训练、轻量化音色微调与 `omnivoice-merge-lora` 权重熔接。
