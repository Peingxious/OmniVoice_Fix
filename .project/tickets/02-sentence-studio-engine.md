# 02 - 分句配音调度引擎与单句重试器 (Sentence Studio Engine & Retry)

**What to build:**
构建解耦的分句配音调度引擎（`SentenceStudioEngine`）与 OmniVoice 模型适配器：
1. **全局声纹 Prompt 一致性管理**：
   - 接受参考音频输入，单次提取 `VoiceClonePrompt` 并缓存在会话状态中；
   - 所有后续分句统一复用该声纹特征，确保全篇所有句子音色严格一致；
2. **批量分句按序推理**：
   - 接受分句列表，支持按序单句或小批量推理并实时汇报进度回调；
   - 输出统一的 24kHz 单声道音频并记录每句时长；
3. **单句现场重新生成与局部替换**：
   - 提供 `regenerate_sentence(index: int, new_text: Optional[str] = None)`；
   - 仅对指定索引号的分句重新调用模型推理，原地更新该句的音频波形与时长；
   - 其余句子的音频缓存保持完全不变，打包缓存自动标记为脏并按需刷新。

**Blocked by:** 01 - 核心文本清洗分句器与规范化打包器

**Status:** complete

**Use check:**
通过脚本调用引擎，上传参考音频，输入 3 句话，一次性生成 3 个音频分句；随后调用 `regenerate_sentence(index=2, new_text="修改后的第二句话")`，验证仅第 2 句音频重新生成且内容变化，第 1、3 句音频完全未受影响，最终导出 ZIP 正确更新。

- [x] 参考音频声纹 Prompt 仅提取一次，后续分句批量共享复用
- [x] 批量生成输出每句独立的 `SentenceItem` 结果对象
- [x] 单句热重跑逻辑生效，且支持就地修改文本重新合成
- [x] 保证音频数据均为标准的 24kHz 单声道格式
