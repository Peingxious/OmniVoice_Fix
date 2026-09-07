# DAT-003: WebDataset 分片打包与动态批处理

## 用途与可观察行为
面向超大规模（数万至数百万小时）多语言训练语料的流式加载与批处理组织。将携带 Token 的 JSONL 转换为标准 WebDataset TAR 归档分片，避免海量小文件对文件系统 inode 的冲击；并在训练运行时基于设定的最大 Token 预算（Max Tokens）执行动态 Batching，最大化 GPU 显存利用率并减少 Pad Token 浪费。

## 打包工具与使用
```bash
python -m omnivoice.scripts.jsonl_to_webdataset \
    --input_jsonl data/train_tokens.jsonl \
    --output_dir data/shards \
    --shard_max_size 1000000000 \
    --shard_max_count 5000
```

## 核心技术组件
1. **WebDataset 数据集抽象**：
   - [omnivoice/data/dataset.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/data/dataset.py) 实现高吞吐流式迭代与断点恢复。
2. **动态 Batching 算法**：
   - [omnivoice/data/batching.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/data/batching.py) 动态统计序列长度（文本 Token + 音频 Token），按 Token 总数预算聚合同步 Batch，长短序列自适应合并。
3. **Collator 填充与对齐**：
   - [omnivoice/data/collator.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/data/collator.py) 构建 Attention Mask 与用于扩散去噪的目标掩码张量。

## 实现入口与依赖
- **分片脚本**：[omnivoice/scripts/jsonl_to_webdataset.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/scripts/jsonl_to_webdataset.py)
- **数据加载模块**：[omnivoice/data/](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/data/) (`dataset.py`, `batching.py`, `collator.py`, `processor.py`)
- **依赖**：`webdataset`, `torch.utils.data`

## 当前状态
- **状态**：`verified` (在 `examples/run_emilia.sh` 等训练管道中作为核心加载器)
