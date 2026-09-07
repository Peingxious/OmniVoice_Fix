"""
Internationalization (i18n) resources and settings persistence for the
OmniVoice Gradio demo.

The UI supports two display languages: Chinese (``zh``) and English (``en``).
Every user-visible string lives here so the demo can switch languages at
runtime without mixing the two.

Two dictionaries are exposed:

``UI``    -- per-component properties (label / placeholder / info / value ...).
             Each entry maps a language code to a dict of keyword arguments
             that is fed straight into ``gr.update(**kwargs)``.
``MSG``   -- plain runtime messages (status text, errors). Each entry maps a
             language code to a string.

Voice-design attributes live in ``CATEGORIES``. Each option carries a stable
``value`` (the text actually sent to the model) so the display language can
change without altering what the model receives.
"""

import json
import os

# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

LANGS = ("zh", "en")
DEFAULT_LANG = "zh"

LANG_CHOICES = [("zh", "中文"), ("en", "English")]

# ---------------------------------------------------------------------------
# Settings persistence (kept inside the project folder)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SETTINGS_PATH = os.path.join(_PROJECT_ROOT, "settings.json")


def load_settings() -> dict:
    """Load the settings file, returning ``{}`` if missing or invalid."""
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_settings(patch: dict) -> bool:
    """Merge ``patch`` into the settings file. Returns True on success."""
    try:
        data = load_settings()
        data.update(patch)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def get_saved_lang() -> str:
    """Return the persisted UI language, falling back to the default."""
    lang = load_settings().get("language")
    return lang if lang in LANGS else DEFAULT_LANG


# ---------------------------------------------------------------------------
# Voice design attributes
#
# Each option: {"zh": <chinese>, "en": <english>, "value": <text for model>}
# The model expects English for most attributes, but Chinese for dialects,
# so "value" is what actually gets sent regardless of display language.
# ---------------------------------------------------------------------------

AUTO = "auto"

AUTO_TEXT = {"zh": "自动", "en": "Auto"}

CATEGORIES = [
    {
        "key": "gender",
        "label": {"zh": "性别", "en": "Gender"},
        "info": None,
        "options": [
            {"zh": "男", "en": "Male", "value": "male"},
            {"zh": "女", "en": "Female", "value": "female"},
        ],
    },
    {
        "key": "age",
        "label": {"zh": "年龄", "en": "Age"},
        "info": None,
        "options": [
            {"zh": "儿童", "en": "Child", "value": "child"},
            {"zh": "少年", "en": "Teenager", "value": "teenager"},
            {"zh": "青年", "en": "Young Adult", "value": "young adult"},
            {"zh": "中年", "en": "Middle-aged", "value": "middle-aged"},
            {"zh": "老年", "en": "Elderly", "value": "elderly"},
        ],
    },
    {
        "key": "pitch",
        "label": {"zh": "音调", "en": "Pitch"},
        "info": None,
        "options": [
            {"zh": "极低音调", "en": "Very Low Pitch", "value": "very low pitch"},
            {"zh": "低音调", "en": "Low Pitch", "value": "low pitch"},
            {"zh": "中音调", "en": "Moderate Pitch", "value": "moderate pitch"},
            {"zh": "高音调", "en": "High Pitch", "value": "high pitch"},
            {"zh": "极高音调", "en": "Very High Pitch", "value": "very high pitch"},
        ],
    },
    {
        "key": "style",
        "label": {"zh": "风格", "en": "Style"},
        "info": None,
        "options": [
            {"zh": "耳语", "en": "Whisper", "value": "whisper"},
        ],
    },
    {
        "key": "english_accent",
        "label": {"zh": "英文口音", "en": "English Accent"},
        "info": {
            "zh": "仅对英文语音生效。",
            "en": "Only effective for English speech.",
        },
        "options": [
            {"zh": "美式口音", "en": "American Accent", "value": "American accent"},
            {
                "zh": "澳大利亚口音",
                "en": "Australian Accent",
                "value": "Australian accent",
            },
            {"zh": "英国口音", "en": "British Accent", "value": "British accent"},
            {"zh": "中国口音", "en": "Chinese Accent", "value": "Chinese accent"},
            {"zh": "加拿大口音", "en": "Canadian Accent", "value": "Canadian accent"},
            {"zh": "印度口音", "en": "Indian Accent", "value": "Indian accent"},
            {"zh": "韩国口音", "en": "Korean Accent", "value": "Korean accent"},
            {
                "zh": "葡萄牙口音",
                "en": "Portuguese Accent",
                "value": "Portuguese accent",
            },
            {"zh": "俄罗斯口音", "en": "Russian Accent", "value": "Russian accent"},
            {"zh": "日本口音", "en": "Japanese Accent", "value": "Japanese accent"},
        ],
    },
    {
        "key": "chinese_dialect",
        "label": {"zh": "中文方言", "en": "Chinese Dialect"},
        "info": {
            "zh": "仅对中文语音生效。",
            "en": "Only effective for Chinese speech.",
        },
        # Dialects have no English equivalent -- the Chinese text IS the value.
        "options": [
            {"zh": "河南话", "en": "河南话", "value": "河南话"},
            {"zh": "陕西话", "en": "陕西话", "value": "陕西话"},
            {"zh": "四川话", "en": "四川话", "value": "四川话"},
            {"zh": "贵州话", "en": "贵州话", "value": "贵州话"},
            {"zh": "云南话", "en": "云南话", "value": "云南话"},
            {"zh": "桂林话", "en": "桂林话", "value": "桂林话"},
            {"zh": "济南话", "en": "济南话", "value": "济南话"},
            {"zh": "石家庄话", "en": "石家庄话", "value": "石家庄话"},
            {"zh": "甘肃话", "en": "甘肃话", "value": "甘肃话"},
            {"zh": "宁夏话", "en": "宁夏话", "value": "宁夏话"},
            {"zh": "青岛话", "en": "青岛话", "value": "青岛话"},
            {"zh": "东北话", "en": "东北话", "value": "东北话"},
        ],
    },
]


def category_choices(index: int, lang: str):
    """Dropdown choices for category ``index`` in the given language."""
    return [AUTO_TEXT[lang]] + [o[lang] for o in CATEGORIES[index]["options"]]


def display_to_value(text):
    """Map a displayed option back to the text sent to the model.

    Language-agnostic: matches against both zh and en display strings.
    Returns ``None`` for the "Auto" placeholder.
    """
    if not text or text in (AUTO_TEXT["zh"], AUTO_TEXT["en"]):
        return None
    for cat in CATEGORIES:
        for opt in cat["options"]:
            if text in (opt["zh"], opt["en"]):
                return opt["value"]
    return text


def value_to_display(value, lang: str, index: int):
    """Map a model value back to its display text in ``lang``."""
    if not value:
        return AUTO_TEXT[lang]
    for opt in CATEGORIES[index]["options"]:
        if opt["value"] == value:
            return opt[lang]
    return AUTO_TEXT[lang]


# ---------------------------------------------------------------------------
# Component strings
# ---------------------------------------------------------------------------

UI = {
    # ---- header ----
    "intro": {
        "zh": {
            "value": (
                "<div class='header-banner'>"
                "<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;'>"
                "<div>"
                "<h1 style='margin:0;font-size:26px;font-weight:800;background:linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>"
                "OmniVoice 语音合成系统"
                "</h1>"
                "<p style='margin:6px 0 0 0;color:#94a3b8;font-size:14px;'>"
                "业界领先的多语言零样本语音生成大模型 · 支持 600+ 语言 · 角色持久化 · 情绪与非语言发音控制"
                "</p>"
                "</div>"
                "<div style='display:flex;gap:8px;flex-wrap:wrap;'>"
                "<span class='hero-chip'>🎙️ 零样本复刻</span>"
                "<span class='hero-chip'>🔄 二次增量训练</span>"
                "<span class='hero-chip'>⚡ 秒级直调</span>"
                "<span class='hero-chip'>🎭 细粒度情绪</span>"
                "</div>"
                "</div>"
                "</div>"
            )
        },
        "en": {
            "value": (
                "<div class='header-banner'>"
                "<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;'>"
                "<div>"
                "<h1 style='margin:0;font-size:26px;font-weight:800;background:linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>"
                "OmniVoice TTS System"
                "</h1>"
                "<p style='margin:6px 0 0 0;color:#94a3b8;font-size:14px;'>"
                "Zero-shot multilingual text-to-speech for 600+ languages · Voice persistence · Emotion & non-verbal control"
                "</p>"
                "</div>"
                "<div style='display:flex;gap:8px;flex-wrap:wrap;'>"
                "<span class='hero-chip'>🎙️ Zero-Shot Clone</span>"
                "<span class='hero-chip'>🔄 Incremental Training</span>"
                "<span class='hero-chip'>⚡ Fast PT Inference</span>"
                "<span class='hero-chip'>🎭 Emotion Control</span>"
                "</div>"
                "</div>"
                "</div>"
            )
        },
    },
    "ui_lang": {
        "zh": {"label": "界面语言", "info": "切换后立即生效，并自动保存。"},
        "en": {
            "label": "Interface Language",
            "info": "Takes effect immediately and is saved automatically.",
        },
    },
    # ---- tabs ----
    "tab_train": {"zh": {"label": "🎙️ 角色训练"}, "en": {"label": "🎙️ Voice Training"}},
    "tab_clone": {"zh": {"label": "👥 声音克隆"}, "en": {"label": "👥 Voice Clone"}},
    "tab_pt": {"zh": {"label": "⚡ 角色调用生成"}, "en": {"label": "⚡ Role Synthesis"}},
    "tab_design": {"zh": {"label": "🎨 声音设计"}, "en": {"label": "🎨 Voice Design"}},
    # ---- emotion & quick controls ----
    "emotion_label": {
        "zh": {
            "value": (
                "<div style='margin-bottom:4px;'>"
                "<span style='font-size:13.5px;font-weight:700;color:#f472b6;'>🎭 局部情绪与语气标记（在兴奋、起伏词句处插入，保留原声特有讲述情绪）：</span>"
                "</div>"
            )
        },
        "en": {
            "value": (
                "<div style='margin-bottom:4px;'>"
                "<span style='font-size:13.5px;font-weight:700;color:#f472b6;'>🎭 Inline Emotion & Expression (Preserve speaker tone, insert at excited parts):</span>"
                "</div>"
            )
        },
    },
    "quick_lang_label": {
        "zh": {
            "value": (
                "<div style='margin-bottom:4px;'>"
                "<span style='font-size:13.5px;font-weight:700;color:#818cf8;'>🌐 语种快速切换（中文转英文 / 英文转中文跨语种极速直切）：</span>"
                "</div>"
            )
        },
        "en": {
            "value": (
                "<div style='margin-bottom:4px;'>"
                "<span style='font-size:13.5px;font-weight:700;color:#818cf8;'>🌐 Quick Language Mode (Instant Chinese ⇄ English Switch):</span>"
                "</div>"
            )
        },
    },
    # ---- shared inputs ----
    "text": {
        "zh": {"label": "待合成文本", "placeholder": "输入你想合成的文本…"},
        "en": {
            "label": "Text to Synthesize",
            "placeholder": "Enter the text you want to synthesize...",
        },
    },
    "ref_audio": {"zh": {"label": "参考音频"}, "en": {"label": "Reference Audio"}},
    "ref_audio_hint": {
        "zh": {"value": "<span style='font-size:0.85em;color:#888;'>建议 3–10 秒音频。</span>"},
        "en": {
            "value": (
                "<span style='font-size:0.85em;color:#888;'>"
                "Recommended: 3&ndash;10 seconds audio.</span>"
            )
        },
    },
    "ref_text": {
        "zh": {
            "label": "参考音频文本（可选）",
            "placeholder": "参考音频的文字内容。留空则用 ASR 自动识别。",
        },
        "en": {
            "label": "Reference Text (optional)",
            "placeholder": "Transcript of the reference audio. Leave empty"
            " to auto-transcribe via ASR.",
        },
    },
    "language": {
        "zh": {"label": "语种（可选）", "info": "保持「自动」即为自动检测语种。"},
        "en": {
            "label": "Language (optional)",
            "info": "Keep as Auto to auto-detect the language.",
        },
    },
    "instruct_accordion": {
        "zh": {"label": "指令（可选）"},
        "en": {"label": "Instruct (optional)"},
    },
    "instruct": {
        "zh": {"label": "指令", "placeholder": "例如：female, low pitch"},
        "en": {"label": "Instruct", "placeholder": "e.g. female, low pitch"},
    },
    # ---- generation settings ----
    "gen_settings": {
        "zh": {"label": "生成参数（可选）"},
        "en": {"label": "Generation Settings (optional)"},
    },
    "speed": {
        "zh": {"label": "语速", "info": "1.0 为正常。大于 1 更快，小于 1 更慢。设定时长时忽略此项。"},
        "en": {
            "label": "Speed",
            "info": "1.0 = normal. >1 faster, <1 slower. Ignored if"
            " Duration is set.",
        },
    },
    "duration": {
        "zh": {"label": "时长（秒）", "info": "留空则用语速控制。设定后将覆盖语速。"},
        "en": {
            "label": "Duration (seconds)",
            "info": "Leave empty to use speed. Set a fixed duration to"
            " override speed.",
        },
    },
    "steps": {
        "zh": {"label": "推理步数", "info": "默认 32。越小越快，越大越好。"},
        "en": {
            "label": "Inference Steps",
            "info": "Default: 32. Lower = faster, higher = better quality.",
        },
    },
    "denoise": {
        "zh": {"label": "降噪", "info": "默认开启。取消勾选可关闭降噪。"},
        "en": {
            "label": "Denoise",
            "info": "Default: enabled. Uncheck to disable denoising.",
        },
    },
    "guidance": {
        "zh": {"label": "引导强度（CFG）", "info": "默认 2.0。"},
        "en": {"label": "Guidance Scale (CFG)", "info": "Default: 2.0."},
    },
    "preprocess": {
        "zh": {
            "label": "预处理提示",
            "info": "对参考音频做静音切除与裁剪，并在参考文本末尾补标点（若原本没有）。",
        },
        "en": {
            "label": "Preprocess Prompt",
            "info": "Apply silence removal and trimming to the reference"
            " audio, add punctuation at the end of the reference text"
            " (if not already).",
        },
    },
    "postprocess": {
        "zh": {"label": "后处理输出", "info": "移除生成音频中的长静音。"},
        "en": {
            "label": "Postprocess Output",
            "info": "Remove long silences from generated audio.",
        },
    },
    # ---- outputs ----
    "generate": {"zh": {"value": "生成"}, "en": {"value": "Generate"}},
    "output_audio": {"zh": {"label": "合成结果"}, "en": {"label": "Output Audio"}},
    "status": {"zh": {"label": "状态"}, "en": {"label": "Status"}},
    # ---- save voice ----
    "save_accordion": {
        "zh": {"label": "保存声音（可复用）"},
        "en": {"label": "Save Voice (reusable)"},
    },
    "save_btn": {"zh": {"value": "保存声音"}, "en": {"value": "Save Voice"}},
    "save_file": {"zh": {"label": "下载 .pt 文件"}, "en": {"label": "Download .pt file"}},
    "save_hint": {
        "zh": {
            "value": (
                "<span style='font-size:0.85em;color:#888;'>"
                "从参考音频中提取音色，保存为极小的 "
                "<code>.pt</code> 文件，存到项目 "
                "<code>voices/</code> 目录，之后可复用，无需重复上传音频。"
                "</span>"
            )
        },
        "en": {
            "value": (
                "<span style='font-size:0.85em;color:#888;'>"
                "Extracts the voice from the reference audio and saves it as"
                " a small <code>.pt</code> file in the project's"
                " <code>voices/</code> folder. Reuse it later instead of"
                " re-uploading the audio.</span>"
            )
        },
    },
    # ---- Tab 1: Voice Training ----
    "train_mode": {
        "zh": {"label": "训练模式"},
        "en": {"label": "Training Mode"},
    },
    "train_char_name": {
        "zh": {
            "label": "角色名称",
            "placeholder": "例如：我的声音 / 雷军 / 自定义名称（留空自动使用时间戳）",
        },
        "en": {
            "label": "Role Name",
            "placeholder": "e.g. MyVoice / custom name (leave empty for timestamp)",
        },
    },
    "train_base_select": {
        "zh": {
            "label": "选择已有角色（追加二次训练）",
            "info": "选择后将在此角色已有声纹基础上追加新内容",
        },
        "en": {
            "label": "Select Existing Role (Incremental)",
            "info": "Append new voice data to this existing character",
        },
    },
    "train_ref_audio": {
        "zh": {"label": "训练参考音频（麦克风录音或文件上传）"},
        "en": {"label": "Training Audio (Record or Upload)"},
    },
    "train_ref_audio_hint": {
        "zh": {"value": "<span style='font-size:0.85em;color:#888;'>录入 3–15 秒清晰语音。多次录入不同语气，声音表现力更丰富。</span>"},
        "en": {
            "value": (
                "<span style='font-size:0.85em;color:#888;'>"
                "Recommended: 3&ndash;15 seconds clear audio. Record multiple clips for richer expression.</span>"
            )
        },
    },
    "train_ref_text": {
        "zh": {
            "label": "参考音频文本",
            "placeholder": "所录制/上传音频对应的文字内容。可手动输入，或点击下方按钮智能识别。",
        },
        "en": {
            "label": "Audio Transcript",
            "placeholder": "Transcript of the audio. Type manually or click transcribe button below.",
        },
    },
    "transcribe_btn": {
        "zh": {"value": "🎙️ 智能识别音频文本 (ASR)"},
        "en": {"value": "🎙️ Transcribe Audio (ASR)"},
    },
    "train_btn": {
        "zh": {"value": "⚡ 开始训练并保存角色 (.pt)"},
        "en": {"value": "⚡ Start Training & Save Role (.pt)"},
    },
    "train_status": {
        "zh": {"label": "训练状态"},
        "en": {"label": "Training Status"},
    },
    "train_file": {
        "zh": {"label": "下载角色 .pt 文件"},
        "en": {"label": "Download Role .pt File"},
    },
    "train_info": {
        "zh": {"label": "当前角色声纹详情"},
        "en": {"label": "Character Voice Info"},
    },
    # ---- Tab 3: Direct PT Role Synthesis ----
    "pt_select": {
        "zh": {
            "label": "选择已训练角色 (PT 文件)",
            "info": "直接从 voices/ 目录中选择已训练好的角色",
        },
        "en": {
            "label": "Select Trained Role (.pt)",
            "info": "Select a trained character directly from voices/",
        },
    },
    "pt_refresh_btn": {
        "zh": {"value": "🔄 刷新角色列表"},
        "en": {"value": "🔄 Refresh Roles"},
    },
    "pt_upload": {
        "zh": {
            "label": "或者：直接上传外部 .pt 角色文件",
            "info": "若使用从他处获取的角色文件，可直接拖入此处",
        },
        "en": {
            "label": "Or: Upload External .pt File",
            "info": "Drag & drop any exported .pt voice prompt file here",
        },
    },
    "pt_info": {
        "zh": {"label": "选中角色信息"},
        "en": {"label": "Selected Role Info"},
    },
    "pt_text": {
        "zh": {"label": "待合成文本", "placeholder": "输入你想让该角色朗读的内容…"},
        "en": {
            "label": "Text to Synthesize",
            "placeholder": "Enter the text you want this character to speak...",
        },
    },
    "pt_gen_btn": {
        "zh": {"value": "🚀 立即生成语音"},
        "en": {"value": "🚀 Generate Speech"},
    },
    # ---- voice design attributes (filled in per category at build time) ----
}

# Add voice-design category dropdowns to UI so they share the update path.
for _i, _cat in enumerate(CATEGORIES):
    UI["cat_%d" % _i] = {
        "zh": {
            "label": _cat["label"]["zh"],
            "info": _cat["info"]["zh"] if _cat["info"] else None,
        },
        "en": {
            "label": _cat["label"]["en"],
            "info": _cat["info"]["en"] if _cat["info"] else None,
        },
    }


# Collapsible section titles (folded away by default to keep the UI compact)
UI["ref_instruct_accordion"] = {
    "zh": {"label": "参考文本 & 指令（可选）"},
    "en": {"label": "Reference Text & Instruct (optional)"},
}
UI["attr_accordion"] = {
    "zh": {"label": "说话人属性（可选）"},
    "en": {"label": "Speaker Attributes (optional)"},
}


# ---------------------------------------------------------------------------
# Runtime messages
# ---------------------------------------------------------------------------

MSG = {
    "done": {"zh": "完成。", "en": "Done."},
    "need_text": {"zh": "请输入要合成的文本。", "en": "Please enter the text to synthesize."},
    "need_ref_audio": {
        "zh": "请先上传参考音频。",
        "en": "Please upload a reference audio.",
    },
    "need_ref_audio_for_save": {
        "zh": "请先上传参考音频。",
        "en": "Please upload a reference audio first.",
    },
    "save_ok": {"zh": "已保存：%s（%.1f KB）→ voices/", "en": "Saved: %s (%.1f KB) -> voices/"},
    "save_failed": {"zh": "保存失败：%s", "en": "Save failed: %s"},
    "error": {"zh": "错误：%s", "en": "Error: %s"},
    "train_ok": {
        "zh": "角色「%s」训练保存成功！（%.1f KB）已就绪，可在第三个页面直接调用生成。",
        "en": "Role '%s' successfully trained and saved! (%.1f KB) Ready for generation in Tab 3.",
    },
    "append_ok": {
        "zh": "角色「%s」二次训练/追加内容完成！（原 %d 帧 + 新增 %d 帧 = 共 %d 帧，%.1f KB）。可在第三个页面直接调用生成。",
        "en": "Incremental training for '%s' complete! (%d + %d = %d frames, %.1f KB). Ready in Tab 3.",
    },
    "no_audio_train": {
        "zh": "请先录制或上传用于训练的音频。",
        "en": "Please record or upload audio for training first.",
    },
    "no_base_selected": {
        "zh": "请先在下拉列表中选择需要二次训练的已有角色。",
        "en": "Please select an existing role to append to.",
    },
    "no_pt_selected": {
        "zh": "请先选择已训练的角色或上传 .pt 文件。",
        "en": "Please select a trained role or upload a .pt file first.",
    },
    "transcribe_done": {
        "zh": "音频识别完成，已自动填充文本框。",
        "en": "Transcription complete and filled into the text box.",
    },
    "transcribe_no_audio": {
        "zh": "请先录制或上传音频再执行识别。",
        "en": "Please record or upload audio before transcribing.",
    },
    "transcribe_no_asr": {
        "zh": "ASR 语音识别模型未加载，请手动输入参考文本。",
        "en": "ASR model is not loaded. Please enter transcript manually.",
    },
}
