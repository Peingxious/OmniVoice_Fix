# HANDOFF: Gradio WebUI 极简流重构、长音频波形自适应展示与参考音频黄金法则交接

## 1. 交接基本信息
- **源上下文**：基于用户对 WebUI 演示界面的实战反馈：
  1. 取消繁琐且易导致原音频破坏的“角色复合训练”Tab；
  2. 在“声音克隆”主流程中支持直接自定义输入角色名称并保存为 `.pt`，自动联动角色库；
  3. 修复长音频在播放器中波形截断、右半段无法显示以及 Windows 原生白色大滚动条遮挡时间标签的问题；
  4. 深入解答“音频越长是否越好”，并将 5~15s 黄金法则与底噪防漂移准则沉淀入 README 与标准文档；
  5. 彻底移除文本框下方生硬突兀、大面积占位的“韵律起伏与停顿控制点”按钮组，使文稿输入区恢复大方清爽，抑扬顿挫统一收敛至自然起伏度滑块与纯文本标记。
- **目标角色**：OmniVoice 专属 Agent。
- **仓库根路径**：`d:\BaiduNetdiskWorkspace\OmniVoice`
- **交接时点**：2026-09-05 完成代码重构、自愈补丁、界面减负、服务热重载与文档全面同步。

---

## 2. 变更路径清单 (Changed & Added Paths)

### 代码与配置层
- [omnivoice/cli/demo.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/cli/demo.py)：
  - **3-Tab 极简架构**：彻底移除冗余且易破坏文件的“角色训练 (复合训练)”Tab，将页面聚焦为：
    - `Tab 1: 声音克隆 (Voice Clone)`：支持音频上传、一键 Whisper 转写、文本合成、抑扬顿挫控制与**自定义角色命名保存**；
    - `Tab 2: 角色调用生成 (Direct PT Inference)`：直接下拉调用已存角色，零重复加载与 ASR；
    - `Tab 3: 音色设计 (Voice Design)`：自然语言属性定制。
  - **界面减负与文稿输入纯净化**：
    - 彻底删除了 Tab 1、Tab 2、Tab 3 文本输入框正下方大块的“🌊 韵律起伏与停顿控制点”按钮面板（`~`、`。`、`…`、`！`、`[哟]`、`[叹]`、`[嗯]`、`[诶]`）；
    - 清理了冗余的 `.prosody-tag-btn` 与 `.emotion-tag-btn` CSS 类，消除不必要的界面高度占用；
    - 抑扬顿挫与起伏控制统一通过“🎭 抑扬顿挫起伏度”滑块（默认 0.35）调节，用户文稿可直接编写自然标点与非言语标签。
  - **角色命名保存与防覆写安全机制**：
    - 新增 `vc_save_name` 文本框（如输入“杨总”），一键保存为 `voices/<角色名>.pt`；
    - 针对同名文件自动增加时间戳保护，绝不静默覆盖已有文件；
    - 保存成功后跨页自动刷新 Tab 2 的角色下拉框并自动选中该角色，无缝衔接长文本合成。
  - **参考音频一键 Whisper ASR**：在参考台词折叠栏内增设“🎙️ 自动识别参考音频台词”按钮。
  - **波形自适应与 Shadow DOM 穿透样式**：
    - 新增 `_ensure_audio_waveform_patch()` 启动自愈函数，将 Gradio 前端硬编码的 `minPxPerSec: 20` 修正为 `0`（全宽填充 `fillParent: true`）；
    - 在 `demo.launch(head=HEAD_HTML)` 注入 Shadow DOM 拦截器，将 WaveSurfer 内部滚动条定制为 5px 超纤细暗色半透明滑块（`rgba(99, 102, 241, 0.45)`）；
    - 重构 `div[data-testid="audio"]` 间距与时间栏隔离（`margin-top: 8px`、`z-index: 5`），彻底消除滚动条压字问题。
- [omnivoice/models/omnivoice.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/models/omnivoice.py)：
  - `transcribe` 显式注入 `return_timestamps=True`，彻底解决 >30 秒长音频 ASR 崩溃报错；
  - `_prepare_inference_inputs` 增加 Prompt 文本与音频时长动态安全对齐，杜绝幽灵文本引发的跨语种发音污染。
- [omnivoice/utils/text.py](file:///d:/BaiduNetdiskWorkspace/OmniVoice/omnivoice/utils/text.py)：
  - `SPLIT_PUNCTUATION` 纳入换行符 `\n\r`、破折号 `—` 与省略号，保障多行文案依语义分片。
- [start_demo.bat](file:///d:/BaiduNetdiskWorkspace/OmniVoice/start_demo.bat)：
  - 启动参数由固定本机环回 `--ip 127.0.0.1` 升级为全网卡监听 `--ip 0.0.0.0 --port 8001`；
  - 集成 Tailscale 状态自动探测，提取 IPv4 虚拟地址并在控制台实时输出双端访问地址（本机与远程）。

### 文档层
- [README.md](file:///d:/BaiduNetdiskWorkspace/OmniVoice/README.md)：
  - 在 Tips 章节全面增补“参考音频黄金法则（5~15s 最优，非越长越好）”以及过长音频四大弊端说明。
- [.doc/01-语音合成推理/INF-001-零样本声音复刻.md](file:///d:/BaiduNetdiskWorkspace/OmniVoice/.doc/01-语音合成推理/INF-001-零样本声音复刻.md)：
  - 增补参考音频选材准则、声文严格对齐原则与过长音频负面效应技术剖析。
- [.doc/01-语音合成推理/INF-004-文本规整与发音非言语控制.md](file:///d:/BaiduNetdiskWorkspace/OmniVoice/.doc/01-语音合成推理/INF-004-文本规整与发音非言语控制.md)：
  - 更新实现入口，明确文本框直输解析，去除冗余按钮。
- [.doc/02-交互与命令行工具/CLI-001-交互式网页演示服务.md](file:///d:/BaiduNetdiskWorkspace/OmniVoice/.doc/02-交互与命令行工具/CLI-001-交互式网页演示服务.md)：
  - 同步更新为 3-Tab 架构，记录自定义角色命名保存、文本输入区域减负纯净化、全景波形自适应展示、深色纤细滚动条实现与 Tailscale 远程访问。
- [.doc/02-交互与命令行工具/CLI-004-Windows启动与缓存管理脚本.md](file:///d:/BaiduNetdiskWorkspace/OmniVoice/.doc/02-交互与命令行工具/CLI-004-Windows启动与缓存管理脚本.md)：
  - 记录 `--ip 0.0.0.0` 全网卡监听机制与 Tailscale IPv4 探针设计。
- [.project/CURRENT.md](file:///d:/BaiduNetdiskWorkspace/OmniVoice/.project/CURRENT.md)：
  - 更新当前已交付功能、Tailscale 远程支持与验证状态。

---

## 3. 关键技术决策与实战依据 (Key Technical Decisions)

### 3.1 极简工作流与取消复合训练
- **痛点**：在零样本（Zero-Shot In-Context）TTS 中，将多个不同录音文件的声学 Token 与文本简单拼接进行“复合/增量训练”，极易引入不同的房间混响、背景底噪和语气冲突，导致原本纯正的音色基准被污染，甚至出现文件覆盖破坏。
- **决策**：果断取消 Tab 1（角色训练 / 复合训练），将工作流收敛为：
  - 在“声音克隆”中挑选一段最优质的声音（5~15s），直接取名保存为 `voices/<角色名>.pt`；
  - 在“角色调用”中秒级加载该 `.pt` 进行任意文本朗读。
  - 用户体验从“复杂的训练调试”回归到“极简的克隆即用”。

### 3.2 文本交互区减负与去除生硬控制面板
- **痛点**：在多行文案输入框下方直接铺设大块“韵律起伏与停顿控制点”和“情绪标签”按钮，视觉割裂严重，误导用户认为文稿必须强插奇怪的代码符号，严重挤占操作空间。
- **决策**：彻底移除按钮组，输入区回归纯粹的自然语言文本框。抑扬顿挫交由更科学全局的“抑扬顿挫起伏度”滑块调控，底层声学模型依据人类真实标点（如逗号、句号、破折号、问号）自然产生节奏停顿与升降调。

### 3.3 参考音频“黄金法则”（为什么音频非越长越好）
- **四大关键弊端**：
  1. **显存与首字延迟**：参考音频被转换为连续声学 Token，Token 随音频时长线性膨胀，严重挤占 Transformer 注意力上下文窗口，导致显存占用暴增、首字延迟成倍增加。
  2. **注意力漂移与幻听**：注意力计算在过长 Prompt 下极易失焦，引发停顿失真、吞字漏字、或跨语种发音混杂。
  3. **底噪被放大模仿**：长音频极难保证全程无吸气喷麦、口水音或环境回响，模型会将这些瑕疵当作说话人的核心声学特征强行学习。
  4. **ASR 窗口与对齐失准**：Whisper 单次声学特征提取上限为 30 秒（3000 mel 帧），超长音频切片边缘极易发生声文对齐偏移。
- **准则**：**5 ~ 15 秒纯净干声，代表性与纯净度远比时长重要**。

### 3.4 波形截断与亮白滚动条压字根治
- **根因**：Gradio 前端硬编码 `minPxPerSec: 20`，49 秒长音频宽度达到 980px+ 从而溢出；Windows Chromium 浏览器在 Shadow DOM 内部由于缺乏外部样式穿透，渲染了 17px 原生刺眼白色滚动条，直接覆盖紧贴在下方的 `0:45 / 0:49` 时间栏。
- **解决机制**：
  1. **自愈补丁**：将 `minPxPerSec` 改为 `0`（`fillParent: true`），使任何长度音频在任何分辨率下均 100% 自适应卡片全宽展现。
  2. **Shadow DOM 穿透**：通过 `Element.prototype.attachShadow` 拦截注入定制 CSS，强制暗色 5px 纤细半透明滚动条。
  3. **防压字布局**：重构时间栏与波形间距（`margin-top: 8px`、`z-index: 5`），确保播放进度永久清晰。

### 3.5 网络监听与 Tailscale 远程协同访问
- **痛点**：默认绑定 `127.0.0.1` 导致在局域网其他主机或通过 Tailscale 连接的移动设备（如手机/平板）无法访问 WebUI，用户无法在移动端或异地调试音色与朗读文稿。
- **决策**：
  - 启动脚本 `start_demo.bat` 绑定由 `127.0.0.1` 变更为 `0.0.0.0`；
  - 引入轻量级状态检测：利用 `tailscale ip -4` 自动嗅探虚拟网卡 IPv4，在控制台统一打印本地与远程访问 URL，兼顾易用性与便携性。

---

## 4. 后续接续与验证建议 (Next Actions)
1. **浏览器体验**：
   - 访问 `http://127.0.0.1:8001`（服务已在后台健康运行）。
   - 界面整体清爽整洁，文本输入框与下方参数布局舒展。
   - Tab 1 上传一段 8~12 秒纯净人声，输入角色名（如“测试男声”），点击保存；
   - 切换至 Tab 2，可见下拉列表已自动选中“测试男声.pt”，输入长篇文案，一键生成完整自然起伏语音。
2. **声音资产备份建议**：
   - 用户创建的声纹资产集中保存在 `voices/` 目录下，包含纯净原版备份（如 `voices/杨总_原始纯净.pt`），可长期稳定复用。
