# 03-性能加速与算子优化

## 边界与职责
本功能块聚焦 OmniVoice 的底层核心计算算子、注意力机制调度及硬件加速特性，保障在多平台（NVIDIA GPU, Intel Arc XPU, Apple Silicon MPS）下的高吞吐与低时延表现。

- **代码前缀**：`ACC`
- **主要受众**：高性能计算工程师、底层算子调优人员与推理服务架构师

## 功能文件索引
- [ACC-001-FlashInfer推理加速与算子融合.md](ACC-001-FlashInfer推理加速与算子融合.md)：针对 CFG 条件对的序列打包、融合 RMSNorm/RoPE 算子与 Batch=1 CUDA Graph 捕获加速。
- [ACC-002-注意力实现调度与精度保护.md](ACC-002-注意力实现调度与精度保护.md)：SDPA、FlashAttention-2 与 flex_attention 自动转换，以及混合精度下防止 fp32 精度回退的 Autocast 桥接。
