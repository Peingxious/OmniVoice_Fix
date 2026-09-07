# TRN-003: 分布式训练与 DeepSpeed 集成

## 用途与可观察行为
支持数卡至数百卡的高性能分布式并行训练。通过与 Hugging Face `Accelerate` 和微软 `DeepSpeed`（特别是 ZeRO-2 优化阶段）深度集成，对优化器状态与梯度进行多卡分片切分，大幅降低单卡显存开销，使在消费级或常见数据中心 GPU 上预训练全量扩散语言模型成为可能。

## 配置与启动示例
- DeepSpeed ZeRO-2 配置文件：[examples/config/ds_config_zero2.json](file:///d:/BaiduNetdiskWorkspace/OmniVoice/examples/config/ds_config_zero2.json)
  - 包含 `fp16`/`bf16` 自动混合精度设置。
  - 梯度累积（`gradient_accumulation_steps`）与梯度裁剪（`gradient_clipping`）。
- 启动脚本示例：
  ```bash
  accelerate launch \
      --config_file examples/config/accelerate_config.yaml \
      omnivoice/cli/train.py \
      --train_config examples/config/train_config_emilia.json \
      --data_config examples/config/data_config_emilia.json \
      --deepspeed examples/config/ds_config_zero2.json
  ```

## 核心技术点
- DDP 通信挂钩：忽略 `torch.distributed.algorithms.ddp_comm_hooks` 告警，采用高效梯度同步。
- ZeRO 状态切分：模型权重保留在各 GPU 上，而优化器状态进行分片，梯度在反向传播结束时自动归约规整。

## 实现入口与依赖
- **配置文件**：[examples/config/ds_config_zero2.json](file:///d:/BaiduNetdiskWorkspace/OmniVoice/examples/config/ds_config_zero2.json)
- **训练启动器**：[omnivoice/training/trainer.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/training/trainer.py)
- **依赖**：`accelerate`, `deepspeed`, `torch.distributed`

## 当前状态
- **状态**：`verified` (在 `examples/run_emilia.sh` 与 `examples/run_finetune.sh` 中广泛应用)
