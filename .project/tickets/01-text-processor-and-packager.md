# 01 - 核心文本清洗分句器与规范化打包器 (Text Processor & Packager Blueprint)

**What to build:**
构建纯 Python 实现、零 UI 依赖、高度可移植的文本清洗分句器与音频序号打包器：
1. `clean_and_split_text(text: str) -> List[str]`：
   - 换行优先原则：严格按 `\n` 切分句子；
   - 自动过滤所有纯空白行、首尾不可见字符及杂乱孤立符号；
   - 无换行兜底：长单行文本按句末终结标点（`。！？.!?`）智能拆句，保留标点；
   - 连句保护：逗号 `，`、顿号 `、`、分号 `；` 保持在原句内不拆分；
2. `pack_sentence_audio_zip(items: List[SentenceItem], output_path: str) -> str`：
   - 将内存中的多段分句音频按严格自然顺序导出为 ZIP 压缩包；
   - 内部音频命名规范：`{index:02d}_{sanitized_text[:15]}.wav`（如 `01_今天天气真好.wav`）；
   - 压缩包附带 `manifest.json` 与 `list.txt`，记录句序、原始文本、音频采样率与时长。

**Blocked by:** None - can start immediately

**Status:** complete

**Use check:**
运行单元测试，输入包含空行、多余换行、混合长句的脏文本，输出标准纯净的分句列表；传入模拟音频数据调用打包函数，成功生成包含排好序的 `01_xxx.wav`、`02_xxx.wav` 及元数据清单的 ZIP 文件，解压校验文件名与音频完整性。

- [x] 规范化分句逻辑覆盖换行优先、空行过滤与无换行标点兜底
- [x] 导出 ZIP 内音频文件名严格补零有序排序（01, 02...）
- [x] 附带 `manifest.json` 元数据索引清单
- [x] 编写并跑通全套独立单元测试

