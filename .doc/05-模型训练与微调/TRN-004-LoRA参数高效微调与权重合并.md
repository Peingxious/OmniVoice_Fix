# TRN-004: LoRA 参数高效微调与权重合并

## 用途与可观察行为
针对特定说话人专属音色、特殊垂直领域发音或小语种快速适配场景。通过微软 PEFT 库冻结模型绝大部分骨干参数，仅在注意力线性层（如 `q_proj`, `v_proj`, `k_proj`, `o_proj`）注入低秩自适应矩阵（LoRA, Low-Rank Adaptation）。大幅缩减显存需求与训练产物大小（微调权重仅数十 MB），并提供 CLI 工具将训练完成的 LoRA 权重一键熔接（Merge）回基础模型，实现零额外推理开销。

## 触发方式与完整管线

### 1. 启动 LoRA 微调
```bash
python -m omnivoice.cli.train \
    --train_config examples/config/train_config_finetune_lora.json \
    --data_config examples/config/data_config_finetune.json
```
- 配置包含 LoRA 超参数：`lora_r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `target_modules=["q_proj", "v_proj"]`。

### 2. 权重合并 (Merge LoRA)
```bash
omnivoice-merge-lora \
    --base_model k2-fsa/OmniVoice \
    --lora_model /path/to/lora_checkpoint \
    --output_dir /path/to/merged_omnivoice
```
合并后的模型即可作为常规基础模型被 `OmniVoice.from_pretrained` 或 `omnivoice-infer` 直接加载。

## 实现入口与依赖
- **LoRA 工具函数**：[omnivoice/utils/lora.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/utils/lora.py)
- **合并 CLI**：[omnivoice/cli/merge_lora.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/cli/merge_lora.py)
- **自动化测试**：[tests/test_lora.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/tests/test_lora.py)
- **文档参考**：[docs/lora_finetuning.md](file:///d:/BaiduNetdiskWorkspace/OmniVoice/docs/lora_finetuning.md)
- **依赖**：`peft>=0.20.0`

## 当前状态
- **状态**：`verified` (在单元测试 `tests/test_lora.py` 中全流程打通，可成功执行注入与 merge)
