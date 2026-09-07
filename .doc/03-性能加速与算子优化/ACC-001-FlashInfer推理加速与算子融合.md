# ACC-001: FlashInfer 推理加速与算子融合

## 用途与可观察行为
通过接入 `FlashInfer` 高性能内核库，将 OmniVoice 扩散推理速度提升 2.0x ~ 2.6x，同时输出结果与未加速基线保持 ASR 验证无损一致。核心优化包含针对 CFG 的正负条件对序列打包（Sequence Packing）、Fused RMSNorm、Fused RoPE，以及针对 Batch=1 低延迟流式场景的 CUDA Graphs 捕获执行。

## 触发方式与启用
- Python API：
  ```python
  from omnivoice.models.omnivoice_flashinfer import apply_flashinfer

  model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda", dtype=torch.float16)
  
  # 批处理高吞吐模式 (Batch >= 2)
  apply_flashinfer(model)

  # 单流低延迟模式 (Batch = 1, 消除内核下发开销)
  apply_flashinfer(model, enable_cuda_graph=True)
  ```
- 批量推理 CLI：在 `omnivoice-infer-batch` 传入 `--enable_flashinfer true`。

## 加速效果对比 (基于 H100, fp16, 32步)
| Batch Size | 基线 RTF | FlashInfer RTF | 加速比 (Speedup) |
| :--- | :--- | :--- | :--- |
| **1** | 0.0899 | 0.0430 | **2.1x** |
| **1 + CUDA Graph** | — | **0.0367** | **2.4x** |
| **2** | 0.0480 | 0.0245 | **2.0x** |
| **4** | 0.0331 | 0.0152 | **2.2x** |
| **8** | 0.0298 | **0.0115** | **2.6x** |

## 实现入口与依赖
- **实现文件**：[omnivoice/models/omnivoice_flashinfer.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/models/omnivoice_flashinfer.py)
- **外部依赖**：`flashinfer-python`, `flashinfer-jit-cache`

## 当前状态
- **状态**：`verified` (代码库已完整集成 `apply_flashinfer` 包装与 CUDA Graph 捕获逻辑)
