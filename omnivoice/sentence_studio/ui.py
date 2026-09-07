# Copyright    2026  OmniVoice Project Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gradio UI Tab for Sentence Studio: line-by-line synthesis, role assignment, retry, and zip export."""

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional
import gradio as gr
import numpy as np

import re
import soundfile as sf

from omnivoice.sentence_studio.engine import SentenceStudioEngine
from omnivoice.sentence_studio.models import SentenceItem
from omnivoice.sentence_studio.text_processor import clean_and_split_text

logger = logging.getLogger(__name__)

# Maximum sentence capacity in the studio interface
MAX_STUDIO_ROWS = 60

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def create_sentence_studio_tab(
    model: Any,
    sampling_rate: int = 24000,
    lang: str = "zh",
    all_languages: Optional[List[str]] = None,
    default_lang_label: str = "Auto",
    list_saved_voices_fn: Optional[Callable[[], List[str]]] = None,
    get_voice_info_fn: Optional[Callable[[str, str], str]] = None,
    voice_dir: Optional[str] = None,
):
    """Build and mount the Sentence Studio Tab into Gradio Blocks.

    Args:
        model: OmniVoice model instance.
        sampling_rate: Output audio sampling rate.
        lang: Current UI language ("zh" or "en").
        all_languages: List of supported language strings.
        default_lang_label: Default label for language dropdown.
        list_saved_voices_fn: Optional callable returning saved role .pt filenames.
        get_voice_info_fn: Optional callable(filename, lang) returning role markdown info.
        voice_dir: Optional path to voices/ directory.
    """
    engine = SentenceStudioEngine(sampling_rate=sampling_rate)
    lang_choices = all_languages if all_languages else ["Auto", "Chinese", "English"]

    # ---------------- UI Labels ----------------
    is_zh = (lang == "zh")
    tab_title = "✂️ 分句配音工作室" if is_zh else "✂️ Sentence Studio"
    header_title = (
        "### ✂️ 分句配音工作室 (Sentence Studio)"
        if is_zh
        else "### ✂️ Sentence Studio: Line-by-Line Dubbing & Packaging"
    )
    header_desc = (
        "一次性输入长文，按行智能清洗切分，逐句配音输出。直接调用已训练好的角色（如杨总.pt）进行全局或逐句角色分配，"
        "亦支持临时上传参考音频。全篇音色高度可控，界面一行接一行呈现，支持单句微调重算与一键批量 ZIP 下载。"
        if is_zh
        else "Paste long text, auto-clean and split into lines, synthesize sentence by sentence. "
        "Assign saved roles (.pt) globally or per sentence, or upload reference audio. "
        "Strictly preserves unified voice timbre, provides layered preview with per-line retry and batch ZIP export."
    )

    if lang_choices and default_lang_label not in lang_choices:
        lang_choices = [default_lang_label] + [c for c in lang_choices if c != "Auto"]

    saved_voices = list_saved_voices_fn() if list_saved_voices_fn else []
    follow_global_label = "跟随全局默认角色" if is_zh else "Follow Global Role"

    with gr.TabItem(tab_title):
        gr.HTML("""
        <style>
        .studio-pagination-bar {
            background: rgba(15, 23, 42, 0.55) !important;
            border: 1px solid rgba(148, 163, 184, 0.15) !important;
            border-radius: 8px !important;
            padding: 4px 10px !important;
            margin-bottom: 6px !important;
            align-items: center !important;
        }
        .studio-pagination-bar .gradio-radio {
            margin: 0 !important;
            padding: 0 !important;
        }
        .studio-pagination-bar button {
            height: 30px !important;
            min-height: 30px !important;
            padding: 0 10px !important;
            font-size: 12px !important;
            border-radius: 6px !important;
        }
        @keyframes studio-pulse {
            0%, 100% { opacity: 1; border-color: rgba(56, 189, 248, 0.6); box-shadow: 0 0 8px rgba(56, 189, 248, 0.25); }
            50% { opacity: 0.8; border-color: rgba(56, 189, 248, 0.3); box-shadow: none; }
        }
        @keyframes studio-spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        .studio-generating-cell {
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 160px;
            height: 32px;
            padding: 0 10px;
            background: rgba(56, 189, 248, 0.12) !important;
            border: 1px solid rgba(56, 189, 248, 0.5) !important;
            border-radius: 6px;
            color: #38bdf8 !important;
            font-size: 12.5px;
            font-weight: 600;
            box-sizing: border-box;
            animation: studio-pulse 1.5s ease-in-out infinite;
        }
        .studio-spin-icon {
            display: inline-block;
            margin-right: 6px;
            animation: studio-spin 1.2s linear infinite;
        }
        .studio-table-header {
            background: rgba(30, 41, 59, 0.7) !important;
            border: 1px solid rgba(148, 163, 184, 0.22) !important;
            border-radius: 8px !important;
            padding: 2px 8px !important;
            margin-bottom: 6px !important;
            min-height: 36px !important;
            max-height: 36px !important;
            height: 36px !important;
            box-sizing: border-box !important;
        }
        .studio-table-header:hover {
            background: rgba(30, 41, 59, 0.7) !important;
            border-color: rgba(148, 163, 184, 0.22) !important;
        }
        .studio-compact-row {
            display: grid !important;
            grid-template-columns: 46px 160px minmax(260px, 1.2fr) minmax(360px, 1fr) 75px !important;
            align-items: center !important;
            gap: 8px !important;
            padding: 2px 8px !important;
            margin-bottom: 4px !important;
            background: rgba(15, 23, 42, 0.45) !important;
            border: 1px solid rgba(148, 163, 184, 0.12) !important;
            border-radius: 8px !important;
            min-height: 46px !important;
            max-height: 48px !important;
            height: 46px !important;
            box-sizing: border-box !important;
            width: 100% !important;
            transition: background 0.15s ease !important;
        }
        .studio-compact-row:nth-child(even) {
            background: rgba(30, 41, 59, 0.35) !important;
        }
        .studio-compact-row:hover {
            background: rgba(49, 46, 129, 0.25) !important;
            border-color: rgba(129, 140, 248, 0.4) !important;
        }
        .studio-compact-row > div {
            display: flex !important;
            align-items: center !important;
            width: 100% !important;
            min-width: 0 !important;
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: visible !important;
        }
        .studio-compact-row .gradio-dropdown,
        .studio-compact-row .gradio-textbox,
        .studio-compact-row .gradio-audio,
        .studio-compact-row .gradio-html,
        .studio-compact-row .block {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        .studio-compact-row .html-container,
        .studio-compact-row .prose,
        .studio-compact-row .gradio-style,
        .studio-compact-row [class*="html-container"],
        .studio-compact-row [class*="prose"],
        .studio-compact-row > div:nth-child(4),
        .studio-compact-row > div:nth-child(4) > div,
        .studio-compact-row > div:nth-child(4) .html-container,
        .studio-compact-row > div:nth-child(4) .prose {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            display: flex !important;
            flex: 1 1 100% !important;
            align-items: center !important;
            box-sizing: border-box !important;
        }
        .studio-compact-row div[data-testid="audio"] {
            background: #0f172a !important;
            border: 1px solid rgba(148, 163, 184, 0.25) !important;
            border-radius: 8px !important;
            padding: 2px 6px !important;
            margin: 0 !important;
            min-height: 38px !important;
            max-height: 40px !important;
            height: 38px !important;
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
            overflow: visible !important;
            box-sizing: border-box !important;
        }
        .studio-compact-row div[data-testid="audio"] .component-wrapper {
            padding: 0 !important;
            margin: 0 !important;
            width: 100% !important;
            height: 100% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
        }
        .studio-compact-row div[data-testid="audio"] .waveform-container {
            min-height: 24px !important;
            height: 24px !important;
            margin: 0 6px !important;
            flex: 1 !important;
        }
        .studio-compact-row div[data-testid="audio"] .timestamps {
            margin: 0 !important;
            padding: 0 4px !important;
            border-top: none !important;
            font-size: 11px !important;
            color: #94a3b8 !important;
        }
        .studio-compact-row div[data-testid="audio"] button {
            height: 28px !important;
            min-height: 28px !important;
            width: 28px !important;
            min-width: 28px !important;
            padding: 0 !important;
            border-radius: 6px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            cursor: pointer !important;
            background: rgba(99, 102, 241, 0.25) !important;
            border: 1px solid rgba(99, 102, 241, 0.5) !important;
            color: #a5b4fc !important;
        }
        .studio-compact-row div[data-testid="audio"] button:hover {
            background: rgba(99, 102, 241, 0.5) !important;
            color: #ffffff !important;
        }
        .studio-compact-row div[data-testid="audio"] button svg {
            width: 14px !important;
            height: 14px !important;
        }
        .studio-compact-row input,
        .studio-compact-row select {
            height: 34px !important;
            min-height: 34px !important;
            line-height: 34px !important;
            font-size: 13px !important;
            padding: 0 8px !important;
            border-radius: 6px !important;
            background: #1e293b !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            color: #f1f5f9 !important;
        }
        .studio-compact-row audio {
            height: 32px !important;
            max-height: 32px !important;
            width: 100% !important;
            min-width: 100% !important;
            display: block !important;
            color-scheme: dark !important;
            background: #1e293b !important;
            border-radius: 6px !important;
            border: 1px solid rgba(148, 163, 184, 0.25) !important;
            outline: none !important;
            box-sizing: border-box !important;
        }
        .studio-compact-row audio::-webkit-media-controls-panel {
            background-color: #1e293b !important;
        }
        .studio-compact-row button {
            height: 32px !important;
            min-height: 32px !important;
            width: 100% !important;
            padding: 0 10px !important;
            font-size: 12.5px !important;
            border-radius: 6px !important;
            white-space: nowrap !important;
        }
        </style>
        """)
        gr.Markdown(f"{header_title}\n\n{header_desc}")

        # Top Control Area: Inputs & Audio Reference
        with gr.Row():
            with gr.Column(scale=1):
                # 1. Primary Role Selector (Exact match with PT Synthesis Tab)
                with gr.Row():
                    studio_pt_select = gr.Dropdown(
                        label="选择已训练角色 (PT 文件)" if is_zh else "Select Trained Role (PT File)",
                        info="直接从 voices/ 目录中选择已训练好的角色" if is_zh else "Directly select trained role from voices/ folder",
                        choices=saved_voices,
                        value=saved_voices[0] if saved_voices else None,
                        interactive=True,
                        scale=4,
                    )
                    studio_pt_refresh = gr.Button(
                        "🔄 刷新角色列表" if is_zh else "🔄 Refresh Roles",
                        scale=1,
                    )

                with gr.Accordion(
                    "或者: 直接上传外部 .pt 角色文件" if is_zh else "Or: Directly upload external .pt role file",
                    open=False,
                ):
                    studio_pt_upload = gr.File(
                        label="上传外部 .pt 角色文件" if is_zh else "Upload external .pt role file",
                        file_types=[".pt"],
                        type="filepath",
                    )

                initial_pt_info = ""
                if saved_voices and get_voice_info_fn:
                    initial_pt_info = get_voice_info_fn(saved_voices[0], lang)
                studio_pt_info = gr.Markdown(value=initial_pt_info, elem_classes="role-info-card")

                # 2. Alternative: Upload raw reference audio
                with gr.Accordion(
                    "🎙️ 或者: 临时上传参考音频进行克隆 (无需 .pt 文件)"
                    if is_zh
                    else "🎙️ Or: Upload temporary reference audio (No .pt needed)",
                    open=False,
                ):
                    studio_ref_audio = gr.Audio(
                        label="参考音频 (用于提取音色)" if is_zh else "Reference Audio (For Timbre)",
                        type="filepath",
                        elem_classes="compact-audio",
                    )
                    studio_ref_text = gr.Textbox(
                        label="参考音频台词 (可选)" if is_zh else "Reference Text (Optional)",
                        lines=2,
                        placeholder="可手动填写，或点击下方按钮自动识别..." if is_zh else "Type transcript or click transcribe below...",
                    )
                    studio_transcribe_btn = gr.Button(
                        "🎙️ 自动识别台词 (Whisper ASR)" if is_zh else "🎙️ Auto-Transcribe (Whisper ASR)",
                        size="sm",
                    )

                # 3. Tuning settings
                with gr.Accordion("⚙️ 发音与语速设置" if is_zh else "⚙️ Generation Settings", open=False):
                    studio_lang = gr.Dropdown(
                        label="目标发音语言" if is_zh else "Target Language",
                        choices=lang_choices,
                        value=default_lang_label,
                        interactive=True,
                    )
                    studio_speed = gr.Slider(
                        0.5, 1.5, value=1.0, step=0.05,
                        label="语速倍率 (Speed)" if is_zh else "Speaking Speed",
                    )
                    studio_instruct = gr.Textbox(
                        label="语气/情绪微调 (可选)" if is_zh else "Emotion / Style (Optional)",
                        placeholder="例如: 亲切自然，沉稳大方" if is_zh else "e.g. cheerful, calm, narrative",
                    )

            with gr.Column(scale=1):
                studio_raw_text = gr.Textbox(
                    label="2. 输入大段台词文本 (每行一句，空行自动清洗)" if is_zh else "2. Input Long Script (Line-first, auto-cleaned)",
                    lines=8,
                    placeholder=(
                        "请直接在此粘贴要配音的长文内容...\n"
                        "支持换行分句，一行就是一句话；\n"
                        "如果中间有逗号、顿号等连句，不会强行拆开；\n"
                        "即便是格式混杂（有多余空行或标点），系统也会自动清洗并对齐序号。"
                        if is_zh
                        else "Paste your script text here...\n"
                        "Each line will be treated as one sentence.\n"
                        "Commas and pauses within a line remain connected.\n"
                        "Empty lines and noise will be automatically cleaned."
                    ),
                )
                with gr.Row():
                    studio_gen_btn = gr.Button(
                        "🚀 一次性分开输出 (批量生成)" if is_zh else "🚀 Batch Generate All Lines",
                        variant="primary",
                        scale=2,
                    )
                    studio_stop_btn = gr.Button(
                        "⏹️ 停止生成" if is_zh else "⏹️ Stop",
                        variant="stop",
                        scale=1,
                    )
                    studio_split_btn = gr.Button(
                        "📝 仅拆分文本预览" if is_zh else "📝 Preview Split Only",
                        variant="secondary",
                        scale=1,
                    )
                    studio_clear_btn = gr.Button(
                        "🗑️ 清空重置" if is_zh else "🗑️ Clear",
                        size="sm",
                        scale=1,
                    )

        # Status & ZIP Export Bar
        with gr.Row(elem_classes="role-info-card"):
            with gr.Column(scale=3):
                studio_status = gr.HTML(
                    "<div style='font-size:13.5px; color:#cbd5e1; padding: 4px 0;'>ℹ️ <b>准备就绪</b>：已就绪角色，请在右侧输入台词，点击【一次性分开输出】启动批量生成。</div>"
                    if is_zh
                    else "<div style='font-size:13.5px; color:#cbd5e1; padding: 4px 0;'>ℹ️ <b>Ready</b>: Role ready. Paste script on the right, then click 'Batch Generate All Lines'.</div>"
                )
            with gr.Column(scale=2):
                studio_zip_btn = gr.DownloadButton(
                    "📦 一键下载全部音频 (ZIP)" if is_zh else "📦 Download All (ZIP)",
                    variant="primary",
                    visible=False,
                )

        # Main Layered Output Area: 60 pre-allocated sentence rows
        gr.Markdown(
            "#### 📋 分句输出与单句微调列表 (支持单行直接修改与独立换角色重跑)"
            if is_zh
            else "#### 📋 Layered Sentence Table (Inline edit and role switch supported)"
        )

        # Pagination Bar
        with gr.Row(elem_classes="studio-pagination-bar"):
            with gr.Column(scale=4):
                studio_page_size = gr.Radio(
                    label="每页显示条数" if is_zh else "Page Size",
                    choices=["10 句/页", "20 句/页", "50 句/页", "全部展开"],
                    value="10 句/页",
                    container=False,
                    interactive=True,
                )
            with gr.Column(scale=5):
                with gr.Row():
                    studio_first_page_btn = gr.Button("⏮️ 首页" if is_zh else "⏮️ First", size="sm", min_width=50)
                    studio_prev_page_btn = gr.Button("◀ 上一页" if is_zh else "◀ Prev", size="sm", min_width=50)
                    studio_page_info = gr.Markdown("📄 **第 1 / 1 页**" if is_zh else "📄 **Page 1 / 1**")
                    studio_next_page_btn = gr.Button("下一页 ▶" if is_zh else "Next ▶", size="sm", min_width=50)
                    studio_last_page_btn = gr.Button("末页 ⏭️" if is_zh else "Last ⏭️", size="sm", min_width=50)

        studio_cur_page = gr.State(1)
        studio_total_count = gr.State(0)

        def _render_generating_html(idx: int) -> str:
            text = f"⚡ 正在合成第 {idx:02d} 句..." if is_zh else f"⚡ Synthesizing #{idx:02d}..."
            return (
                f'<div class="studio-generating-cell">'
                f'<span class="studio-spin-icon">⏳</span> {text}'
                f'</div>'
            )

        def _render_audio_html(wav_path: Optional[str]) -> str:
            if not wav_path or not os.path.isfile(wav_path):
                empty_text = "等待生成..." if is_zh else "Waiting..."
                return (
                    f'<div style="font-size:12px; color:#64748b; padding-left:6px; '
                    f'width:100%; font-style:italic; display:flex; align-items:center; height:32px;">{empty_text}</div>'
                )
            abs_path = os.path.abspath(wav_path).replace("\\", "/")
            return (
                f'<div style="width:100%; display:flex; align-items:center; flex:1 1 100%; min-width:0;">'
                f'<audio controls preload="metadata" src="/gradio_api/file={abs_path}" '
                f'style="height:32px; width:100%; min-width:100%; color-scheme:dark; '
                f'background:#1e293b; border-radius:6px; border:1px solid rgba(148,163,184,0.25); outline:none; display:block;">'
                f'</audio></div>'
            )

        row_containers: List[Any] = []
        row_role_boxes: List[Any] = []
        row_text_boxes: List[Any] = []
        row_audio_boxes: List[Any] = []
        row_regen_btns: List[Any] = []

        role_choices = [follow_global_label] + saved_voices

        # Single compact table header row with exact matching grid columns
        with gr.Row(elem_classes=["studio-table-header", "studio-compact-row"]):
            gr.HTML(
                f"<div style='font-size:13px;font-weight:600;color:#94a3b8;text-align:center;width:100%;'>{'序号' if is_zh else '#'}</div>",
                container=False,
            )
            gr.HTML(
                f"<div style='font-size:13px;font-weight:600;color:#94a3b8;padding-left:4px;width:100%;'>{'分配角色' if is_zh else 'Role'}</div>",
                container=False,
            )
            gr.HTML(
                f"<div style='font-size:13px;font-weight:600;color:#94a3b8;padding-left:4px;width:100%;'>{'台词文本 (单行直接可改)' if is_zh else 'Script Line (Editable)'}</div>",
                container=False,
            )
            gr.HTML(
                f"<div style='font-size:13px;font-weight:600;color:#94a3b8;padding-left:4px;width:100%;'>{'试听预览' if is_zh else 'Audio Preview'}</div>",
                container=False,
            )
            gr.HTML(
                f"<div style='font-size:13px;font-weight:600;color:#94a3b8;text-align:center;width:100%;'>{'操作' if is_zh else 'Action'}</div>",
                container=False,
            )

        with gr.Column():
            for i in range(MAX_STUDIO_ROWS):
                with gr.Row(visible=False, elem_classes="studio-compact-row") as row_cont:
                    idx_label = gr.HTML(
                        f"<div style='min-width:36px;font-size:13.5px;font-weight:700;color:#38bdf8;text-align:center;'>#{i+1:02d}</div>",
                        container=False,
                    )
                    sent_role = gr.Dropdown(
                        show_label=False,
                        container=False,
                        choices=role_choices,
                        value=follow_global_label,
                        interactive=True,
                        scale=2,
                    )
                    sent_text = gr.Textbox(
                        show_label=False,
                        container=False,
                        lines=1,
                        max_lines=1,
                        placeholder=f"#{i+1:02d} 台词内容..." if is_zh else f"#{i+1:02d} Line...",
                        scale=5,
                    )
                    sent_audio = gr.HTML(
                        value=_render_audio_html(None),
                        container=False,
                        scale=4,
                    )
                    sent_regen = gr.Button(
                        "🔄 重跑" if is_zh else "🔄 Retry",
                        variant="secondary",
                        size="sm",
                        scale=1,
                    )

                    row_containers.append(row_cont)
                    row_role_boxes.append(sent_role)
                    row_text_boxes.append(sent_text)
                    row_audio_boxes.append(sent_audio)
                    row_regen_btns.append(sent_regen)

        # ---------------- Interaction Handlers ----------------

        # Helper to resolve pt path
        def _resolve_pt_path(filename: Optional[str]) -> Optional[str]:
            if not filename:
                return None
            if os.path.isabs(filename) and os.path.isfile(filename):
                return filename
            if voice_dir:
                candidate = os.path.join(voice_dir, filename)
                if os.path.isfile(candidate):
                    return candidate
            return filename

        # Role selection change
        def _on_role_select(chosen_file):
            if not chosen_file or not get_voice_info_fn:
                return ""
            return get_voice_info_fn(chosen_file, lang)

        studio_pt_select.change(
            fn=_on_role_select,
            inputs=[studio_pt_select],
            outputs=[studio_pt_info],
        )

        # Upload external role change
        def _on_upload_pt(uploaded_file):
            if not uploaded_file or not get_voice_info_fn:
                return ""
            path = uploaded_file.name if hasattr(uploaded_file, "name") else str(uploaded_file)
            return get_voice_info_fn(path, lang)

        studio_pt_upload.change(
            fn=_on_upload_pt,
            inputs=[studio_pt_upload],
            outputs=[studio_pt_info],
        )

        # Refresh roles list
        def _on_refresh_roles():
            new_voices = list_saved_voices_fn() if list_saved_voices_fn else []
            new_val = new_voices[0] if new_voices else None
            new_info = get_voice_info_fn(new_val, lang) if (new_val and get_voice_info_fn) else ""
            new_row_choices = [follow_global_label] + new_voices
            row_updates = [gr.update(choices=new_row_choices) for _ in range(MAX_STUDIO_ROWS)]
            return [gr.update(choices=new_voices, value=new_val), new_info] + row_updates

        studio_pt_refresh.click(
            fn=_on_refresh_roles,
            inputs=[],
            outputs=[studio_pt_select, studio_pt_info] + row_role_boxes,
        )

        # Transcribe reference audio
        def _on_transcribe(audio_path):
            if not audio_path:
                return "", "⚠️ 请先上传参考音频！" if is_zh else "⚠️ Please upload reference audio first!"
            if not hasattr(model, "transcribe"):
                return "", "⚠️ 当前模型未加载 ASR 模块。" if is_zh else "⚠️ ASR module not loaded in model."
            try:
                text = model.transcribe(audio_path)
                return text, "✅ 参考音频台词识别完成！" if is_zh else "✅ Audio transcript recognized!"
            except Exception as e:
                return "", f"❌ 识别失败: {e}"

        studio_transcribe_btn.click(
            fn=_on_transcribe,
            inputs=[studio_ref_audio],
            outputs=[studio_ref_text, studio_status],
        )

        # Temporary directory for clean WAV audio streaming
        studio_audio_dir = os.path.join(_PROJECT_ROOT, ".cache", "tmp", "studio_audios")
        os.makedirs(studio_audio_dir, exist_ok=True)

        # Helper to calculate pagination visibility
        def _calc_pagination(page: int, page_size_choice: str, total_count: int):
            if not page_size_choice or "10" in page_size_choice:
                size = 10
            elif "20" in page_size_choice:
                size = 20
            elif "50" in page_size_choice:
                size = 50
            else:  # 全部展开
                size = 9999

            total_pages = max(1, (total_count + size - 1) // size) if total_count > 0 else 1
            page = max(1, min(page, total_pages))

            start_idx = (page - 1) * size
            end_idx = start_idx + size

            row_updates = []
            for i in range(MAX_STUDIO_ROWS):
                is_visible = (i < total_count) and (start_idx <= i < end_idx)
                row_updates.append(gr.update(visible=is_visible))

            info_text = (
                f"📄 **第 {page} / {total_pages} 页**"
                if is_zh
                else f"📄 **Page {page} / {total_pages}**"
            )
            return page, info_text, row_updates

        # Pagination button callbacks
        def _on_page_size_change(size_choice, cur_page, total_count):
            page, info_text, row_updates = _calc_pagination(1, size_choice, total_count)
            return [page, info_text] + row_updates

        def _on_first_page(size_choice, total_count):
            page, info_text, row_updates = _calc_pagination(1, size_choice, total_count)
            return [page, info_text] + row_updates

        def _on_prev_page(cur_page, size_choice, total_count):
            page, info_text, row_updates = _calc_pagination(cur_page - 1, size_choice, total_count)
            return [page, info_text] + row_updates

        def _on_next_page(cur_page, size_choice, total_count):
            page, info_text, row_updates = _calc_pagination(cur_page + 1, size_choice, total_count)
            return [page, info_text] + row_updates

        def _on_last_page(size_choice, total_count):
            page, info_text, row_updates = _calc_pagination(9999, size_choice, total_count)
            return [page, info_text] + row_updates

        studio_page_size.change(
            fn=_on_page_size_change,
            inputs=[studio_page_size, studio_cur_page, studio_total_count],
            outputs=[studio_cur_page, studio_page_info] + row_containers,
        )
        studio_first_page_btn.click(
            fn=_on_first_page,
            inputs=[studio_page_size, studio_total_count],
            outputs=[studio_cur_page, studio_page_info] + row_containers,
        )
        studio_prev_page_btn.click(
            fn=_on_prev_page,
            inputs=[studio_cur_page, studio_page_size, studio_total_count],
            outputs=[studio_cur_page, studio_page_info] + row_containers,
        )
        studio_next_page_btn.click(
            fn=_on_next_page,
            inputs=[studio_cur_page, studio_page_size, studio_total_count],
            outputs=[studio_cur_page, studio_page_info] + row_containers,
        )
        studio_last_page_btn.click(
            fn=_on_last_page,
            inputs=[studio_page_size, studio_total_count],
            outputs=[studio_cur_page, studio_page_info] + row_containers,
        )

        # Preview split only
        def _on_split_preview(raw_text, page_size_choice):
            if not raw_text or not raw_text.strip():
                status = (
                    "<div style='font-size:13.5px; color:#f59e0b; padding: 4px 0;'>⚠️ 请输入需要拆分的文本内容！</div>"
                    if is_zh
                    else "<div style='font-size:13.5px; color:#f59e0b; padding: 4px 0;'>⚠️ Please enter text to split!</div>"
                )
                empty_res = [status, 1, 0, "📄 **第 1 / 1 页**" if is_zh else "📄 **Page 1 / 1**"]
                for _ in range(MAX_STUDIO_ROWS):
                    empty_res.extend([
                        gr.update(visible=False),
                        gr.update(value=follow_global_label),
                        gr.update(value=""),
                        gr.update(value=_render_audio_html(None)),
                    ])
                return empty_res

            sentences = clean_and_split_text(raw_text)
            n = len(sentences)
            if n > MAX_STUDIO_ROWS:
                sentences = sentences[:MAX_STUDIO_ROWS]
                n = MAX_STUDIO_ROWS
                status = (
                    f"<div style='font-size:13.5px; color:#f59e0b; padding: 4px 0;'>⚠️ 文本超出上限，已截取前 <b>{MAX_STUDIO_ROWS}</b> 句。可在下方分页查看或直接修改。</div>"
                    if is_zh
                    else f"<div style='font-size:13.5px; color:#f59e0b; padding: 4px 0;'>⚠️ Script exceeded limit, truncated to first {MAX_STUDIO_ROWS} lines.</div>"
                )
            else:
                status = (
                    f"<div style='font-size:13.5px; color:#34d399; padding: 4px 0;'>✅ 成功清洗并拆分为 <b>{n}</b> 句话！已默认分页呈现，可在下方修改文本或分配角色，随后点击【一次性分开输出】批量生成。</div>"
                    if is_zh
                    else f"<div style='font-size:13.5px; color:#34d399; padding: 4px 0;'>✅ Split into <b>{n}</b> clean sentences! Review below, then click Batch Generate.</div>"
                )

            # Pre-initialize items in engine
            engine.items = [
                SentenceItem(index=i + 1, text=s, raw_text=s, sample_rate=sampling_rate, status="pending")
                for i, s in enumerate(sentences)
            ]

            page, info_text, row_visibilities = _calc_pagination(1, page_size_choice, n)

            updates = [status, page, n, info_text]
            for i in range(MAX_STUDIO_ROWS):
                if i < n:
                    updates.append(row_visibilities[i])
                    updates.append(gr.update(value=follow_global_label))
                    updates.append(gr.update(value=sentences[i]))
                    updates.append(gr.update(value=_render_audio_html(None)))
                else:
                    updates.append(gr.update(visible=False))
                    updates.append(gr.update(value=follow_global_label))
                    updates.append(gr.update(value=""))
                    updates.append(gr.update(value=_render_audio_html(None)))

            return updates

        batch_row_components = []
        for i in range(MAX_STUDIO_ROWS):
            batch_row_components.extend([
                row_containers[i],
                row_role_boxes[i],
                row_text_boxes[i],
                row_audio_boxes[i],
            ])

        split_output_components = [
            studio_status,
            studio_cur_page,
            studio_total_count,
            studio_page_info,
        ] + batch_row_components

        studio_split_btn.click(
            fn=_on_split_preview,
            inputs=[studio_raw_text, studio_page_size],
            outputs=split_output_components,
            show_progress="hidden",
        )

        # Render clean HTML progress bar without blocking component overlays
        def _render_progress_html(
            title: str,
            cur_idx: int,
            total: int,
            pct: float,
            cur_sentence_text: str = "",
            role_name: str = "",
            is_finished: bool = False,
            is_error: bool = False,
        ) -> str:
            if is_error:
                return f"""
                <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 8px; padding: 8px 14px; margin-bottom: 4px;">
                    <span style="font-size:13.5px; font-weight:600; color:#f87171;">{title}</span>
                </div>
                """
            if is_finished:
                return f"""
                <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.45); border-radius: 8px; padding: 10px 14px; margin-bottom: 4px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:13.5px; font-weight:600; color:#34d399;">{title}</span>
                        <span style="font-size:12px; font-weight:bold; color:#10b981; font-family:monospace;">共 {total} 句全部就绪 (100%)</span>
                    </div>
                </div>
                """

            clamped_pct = max(0.0, min(pct, 100.0))
            snippet = (cur_sentence_text[:30] + "...") if len(cur_sentence_text) > 30 else cur_sentence_text
            role_badge = f'<span style="font-size:11px; padding:1px 6px; border-radius:4px; background:rgba(99,102,241,0.25); color:#a5b4fc; border:1px solid rgba(129,140,248,0.3); margin-left:6px;">{role_name}</span>' if role_name else ""

            return f"""
            <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 8px; padding: 8px 14px; margin-bottom: 4px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;">
                    <div style="display:flex; align-items:center; font-size:13px; color:#f1f5f9; font-weight:600;">
                        <span>{title}</span>
                        {role_badge}
                        {f'<span style="color:#94a3b8; font-size:12px; margin-left:8px; font-weight:normal;">"{snippet}"</span>' if snippet else ''}
                    </div>
                    <span style="font-size:12.5px; font-weight:bold; color:#38bdf8; font-family:monospace;">
                        {cur_idx} / {total} 句 ({clamped_pct:.0f}%)
                    </span>
                </div>
                <div style="width: 100%; height: 5px; background: rgba(30, 41, 59, 0.9); border-radius: 9999px; overflow: hidden;">
                    <div style="width: {clamped_pct}%; height: 100%; background: linear-gradient(90deg, #6366f1, #a855f7); border-radius: 9999px; transition: width 0.25s ease;"></div>
                </div>
            </div>
            """

        # Batch generate streaming (Strictly progressive line-by-line reveal)
        def _on_batch_generate(
            pt_choice,
            pt_upload_file,
            ref_audio,
            ref_text,
            raw_text,
            target_lang,
            speed_val,
            instruct_text,
            page_size_choice,
            *per_row_roles,
        ):
            # 1. Determine global voice source: Uploaded PT -> Selected PT -> Raw Audio
            global_pt_path = None
            if pt_upload_file is not None:
                global_pt_path = pt_upload_file.name if hasattr(pt_upload_file, "name") else str(pt_upload_file)
            elif pt_choice:
                global_pt_path = _resolve_pt_path(pt_choice)

            if global_pt_path and os.path.isfile(global_pt_path):
                yield [
                    _render_progress_html(f"⏳ 正在加载全局角色声纹 `{os.path.basename(global_pt_path)}`...", 0, 0, 0.0),
                    gr.update(visible=False),
                    1,
                    0,
                    "📄 **第 1 / 1 页**" if is_zh else "📄 **Page 1 / 1**",
                ] + [gr.update() for _ in range(MAX_STUDIO_ROWS * 4)]

                try:
                    engine.prepare_voice_prompt(source_type="pt", pt_path=global_pt_path)
                except Exception as e:
                    logger.error(f"Failed to load role prompt: {e}", exc_info=True)
                    yield [
                        _render_progress_html(f"❌ 加载角色失败: {e}", 0, 0, 0.0, is_error=True),
                        gr.update(visible=False),
                        1,
                        0,
                        "📄 **第 1 / 1 页**" if is_zh else "📄 **Page 1 / 1**",
                    ] + [gr.update() for _ in range(MAX_STUDIO_ROWS * 4)]
                    return
            elif ref_audio:
                yield [
                    _render_progress_html("⏳ 正在提取参考音频声纹特征（全局统一样本音色）...", 0, 0, 0.0),
                    gr.update(visible=False),
                    1,
                    0,
                    "📄 **第 1 / 1 页**" if is_zh else "📄 **Page 1 / 1**",
                ] + [gr.update() for _ in range(MAX_STUDIO_ROWS * 4)]

                try:
                    engine.prepare_voice_prompt(model=model, source_type="audio", ref_audio=ref_audio, ref_text=ref_text)
                except Exception as e:
                    logger.error(f"Failed to extract prompt: {e}", exc_info=True)
                    yield [
                        _render_progress_html(f"❌ 提取参考音频失败: {e}", 0, 0, 0.0, is_error=True),
                        gr.update(visible=False),
                        1,
                        0,
                        "📄 **第 1 / 1 页**" if is_zh else "📄 **Page 1 / 1**",
                    ] + [gr.update() for _ in range(MAX_STUDIO_ROWS * 4)]
                    return
            else:
                yield [
                    _render_progress_html("⚠️ 请在左侧选择已保存的角色（如 杨总.pt）或在折叠面板中上传参考音频！", 0, 0, 0.0, is_error=True),
                    gr.update(visible=False),
                    1,
                    0,
                    "📄 **第 1 / 1 页**" if is_zh else "📄 **Page 1 / 1**",
                ] + [gr.update() for _ in range(MAX_STUDIO_ROWS * 4)]
                return

            if not raw_text or not raw_text.strip():
                yield [
                    _render_progress_html("⚠️ 请输入需要配音的台词文本！", 0, 0, 0.0, is_error=True),
                    gr.update(visible=False),
                    1,
                    0,
                    "📄 **第 1 / 1 页**" if is_zh else "📄 **Page 1 / 1**",
                ] + [gr.update() for _ in range(MAX_STUDIO_ROWS * 4)]
                return

            # Initialize sentences
            sentences = clean_and_split_text(raw_text)
            n = len(sentences)
            if n > MAX_STUDIO_ROWS:
                sentences = sentences[:MAX_STUDIO_ROWS]
                n = MAX_STUDIO_ROWS

            items = engine.initialize_sentences("\n".join(sentences))
            total = len(items)

            page, info_text, _ = _calc_pagination(1, page_size_choice, total)
            if not page_size_choice or "10" in page_size_choice:
                p_size = 10
            elif "20" in page_size_choice:
                p_size = 20
            elif "50" in page_size_choice:
                p_size = 50
            else:
                p_size = 9999
            start_idx = (page - 1) * p_size
            end_idx = start_idx + p_size

            # Progressive line-by-line reveal: Initially all rows are hidden!
            current_row_updates = []
            for i in range(MAX_STUDIO_ROWS):
                assigned_role = per_row_roles[i] if i < len(per_row_roles) else follow_global_label
                text_val = items[i].text if i < n else ""
                current_row_updates.extend([
                    gr.update(visible=False),
                    gr.update(value=assigned_role),
                    gr.update(value=text_val),
                    gr.update(value=_render_audio_html(None)),
                ])

            role_desc = f"角色 `{os.path.basename(engine.ref_audio_source)}`" if engine.ref_audio_source else "参考音色"
            yield [
                _render_progress_html(f"🚀 已就绪 {total} 句话，开始逐句合成...", 0, total, 0.0, role_name=role_desc),
                gr.update(visible=False),
                page,
                total,
                info_text,
            ] + current_row_updates

            # Sequential synthesis with yield: One line at a time!
            lang_param = None if target_lang in ("Auto", "auto") else target_lang

            engine.cancel_requested = False
            cur_page = 1
            for idx, item in enumerate(items):
                # Check cancellation request before starting next sentence
                if getattr(engine, "cancel_requested", False):
                    logger.info("Batch generation interrupted by user request before sentence %d", idx + 1)
                    break

                item.status = "generating"
                assigned_role = per_row_roles[idx] if idx < len(per_row_roles) else follow_global_label

                # Auto-page to the page containing current idx so user sees real-time progress
                cur_page = (idx // p_size) + 1
                page, info_text, _ = _calc_pagination(cur_page, page_size_choice, total)
                start_idx = (cur_page - 1) * p_size
                end_idx = start_idx + p_size

                # Make ONLY rows 0..idx visible within the current page (strictly one by one!)
                for i in range(MAX_STUDIO_ROWS):
                    should_show = (i <= idx) and (i < n) and (start_idx <= i < end_idx)
                    current_row_updates[i * 4] = gr.update(visible=should_show)

                # Show current sentence actively generating badge!
                current_row_updates[idx * 4 + 3] = gr.update(value=_render_generating_html(idx + 1))

                progress_html = _render_progress_html(
                    f"🎙️ 正在逐句合成 (第 {idx + 1}/{total} 句)...",
                    idx,
                    total,
                    (idx / total) * 100.0,
                    cur_sentence_text=item.text,
                    role_name=assigned_role if assigned_role != follow_global_label else "全局默认角色",
                )
                yield [
                    progress_html,
                    gr.update(visible=False),
                    cur_page,
                    total,
                    info_text,
                ] + current_row_updates

                # Check custom role vs cached prompt
                if assigned_role and assigned_role != follow_global_label:
                    item.role_name = assigned_role
                    custom_pt = _resolve_pt_path(assigned_role)
                    if custom_pt and os.path.isfile(custom_pt):
                        prompt_to_use = engine.get_role_prompt(custom_pt)
                    else:
                        prompt_to_use = engine.cached_prompt
                else:
                    item.role_name = os.path.basename(engine.ref_audio_source) if engine.ref_audio_source else None
                    prompt_to_use = engine.cached_prompt

                try:
                    kw: Dict[str, Any] = {
                        "text": item.text.strip(),
                        "language": lang_param,
                        "voice_clone_prompt": prompt_to_use,
                    }
                    if speed_val != 1.0:
                        kw["speed"] = float(speed_val)
                    if instruct_text and instruct_text.strip():
                        kw["instruct"] = instruct_text.strip()

                    audios = model.generate(**kw)
                    waveform = audios[0]
                    if waveform.ndim > 1:
                        waveform = waveform.squeeze()

                    item.waveform = waveform
                    item.duration = len(waveform) / sampling_rate
                    item.status = "done"

                    # Convert to 16-bit PCM and save as proper WAV file for smooth browser streaming
                    int16_wave = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
                    wav_filename = f"sentence_{item.formatted_index}_{int(time.time() * 1000)}.wav"
                    wav_path = os.path.join(studio_audio_dir, wav_filename)
                    sf.write(wav_path, int16_wave, sampling_rate)
                    audio_val = _render_audio_html(wav_path)
                except Exception as e:
                    logger.error(f"Error on sentence {item.formatted_index}: {e}", exc_info=True)
                    item.status = "error"
                    item.error_msg = str(e)
                    audio_val = _render_audio_html(None)

                # Update row audio immediately upon completion!
                current_row_updates[idx * 4 + 3] = gr.update(value=audio_val)

                done_html = _render_progress_html(
                    f"✅ 第 {idx + 1} 句就绪 (时长 {item.duration:.1f}s)",
                    idx + 1,
                    total,
                    ((idx + 1) / total) * 100.0,
                    cur_sentence_text=item.text,
                    role_name=assigned_role if assigned_role != follow_global_label else "全局默认角色",
                )
                yield [
                    done_html,
                    gr.update(visible=False),
                    cur_page,
                    total,
                    info_text,
                ] + current_row_updates

                # Check if user requested stop while this sentence was computing
                if getattr(engine, "cancel_requested", False):
                    logger.info("Batch generation interrupted by user request after sentence %d", idx + 1)
                    break

            # If stopped early by user
            if getattr(engine, "cancel_requested", False):
                total_done = sum(1 for it in items if it.waveform is not None)
                stop_msg = (
                    f"⏹️ **批量合成已中止！** 已完成前 **{total_done}** 句并全部保留，您可在下方试听或打包下载；"
                    f"随时可以点击对应行的【🔄 重跑】重算或继续编辑。"
                    if is_zh
                    else f"⏹️ **Batch Stopped!** {total_done} lines completed and preserved. You can listen, download, or retry below."
                )
                stop_html = _render_progress_html(
                    stop_msg,
                    total_done,
                    total,
                    (total_done / total) * 100.0 if total > 0 else 0.0,
                    is_finished=True,
                )
                final_page = cur_page if total > 0 else 1
                final_start = (final_page - 1) * p_size
                final_end = final_start + p_size
                for i in range(MAX_STUDIO_ROWS):
                    is_vis = (i < total) and (final_start <= i < final_end)
                    current_row_updates[i * 4] = gr.update(visible=is_vis)
                    # Revert any unfinished rows back to empty state
                    if i >= total_done and (i >= len(items) or items[i].waveform is None):
                        current_row_updates[i * 4 + 3] = gr.update(value=_render_audio_html(None))

                yield [
                    stop_html,
                    gr.update(visible=(total_done > 0)),
                    final_page,
                    total,
                    info_text,
                ] + current_row_updates
                return

            # Finished all
            total_sec = sum(it.duration for it in items if it.waveform is not None)
            finish_msg = (
                f"🎉 **全篇合成完毕！** 共完成 **{total}** 句，总时长 **{total_sec:.1f}秒**。"
                f"您可直接在每行试听或修改台词；点击上方【📦 一键下载全部音频 (ZIP)】即可直接下载压缩包。"
                if is_zh
                else f"🎉 **All {total} lines finished!** Total duration: **{total_sec:.1f}s**. "
                f"You can listen or edit inline; click 'Download All (ZIP)' above to download directly."
            )
            finish_html = _render_progress_html(
                finish_msg,
                total,
                total,
                100.0,
                is_finished=True,
            )

            # When finished, make sure all rows belonging to cur_page are visible
            final_page = cur_page if total > 0 else 1
            final_start = (final_page - 1) * p_size
            final_end = final_start + p_size
            for i in range(MAX_STUDIO_ROWS):
                is_vis = (i < total) and (final_start <= i < final_end)
                current_row_updates[i * 4] = gr.update(visible=is_vis)

            # Mark studio_zip_btn visible without preloading value (preventing double downloads!)
            yield [
                finish_html,
                gr.update(visible=True),
                final_page,
                total,
                info_text,
            ] + current_row_updates

        batch_full_outputs = [
            studio_status,
            studio_zip_btn,
            studio_cur_page,
            studio_total_count,
            studio_page_info,
        ] + batch_row_components

        batch_click_ev = studio_gen_btn.click(
            fn=_on_batch_generate,
            inputs=[
                studio_pt_select,
                studio_pt_upload,
                studio_ref_audio,
                studio_ref_text,
                studio_raw_text,
                studio_lang,
                studio_speed,
                studio_instruct,
                studio_page_size,
            ] + row_role_boxes,
            outputs=batch_full_outputs,
            show_progress="hidden",
        )

        def _on_stop_batch():
            engine.cancel_requested = True
            stop_msg = (
                "<div style='font-size:13.5px; color:#f59e0b; padding:4px 0;'>⏹️ <b>已请求停止生成</b>：当前正在合成的单句完成后将立即中止批量流程...</div>"
                if is_zh
                else "<div style='font-size:13.5px; color:#f59e0b; padding:4px 0;'>⏹️ <b>Stop Requested</b>: Halting after current sentence...</div>"
            )
            return stop_msg

        studio_stop_btn.click(
            fn=_on_stop_batch,
            inputs=[],
            outputs=[studio_status],
            cancels=[batch_click_ev],
        )

        # Per-row retry handlers
        def _make_regen_fn(row_index: int):
            def _regen(cur_text, assigned_role, pt_choice, pt_upload_file, ref_audio, ref_text, target_lang, speed_val, instruct_text):
                if not cur_text or not cur_text.strip():
                    return gr.update(), f"⚠️ #{row_index + 1:02d} 文本不能为空！"

                prompt_to_use = None
                if assigned_role and assigned_role != follow_global_label:
                    custom_pt = _resolve_pt_path(assigned_role)
                    if custom_pt and os.path.isfile(custom_pt):
                        prompt_to_use = engine.get_role_prompt(custom_pt)

                if prompt_to_use is None:
                    if engine.cached_prompt is None:
                        pt_path = None
                        if pt_upload_file is not None:
                            pt_path = pt_upload_file.name if hasattr(pt_upload_file, "name") else str(pt_upload_file)
                        elif pt_choice:
                            pt_path = _resolve_pt_path(pt_choice)

                        if pt_path and os.path.isfile(pt_path):
                            prompt_to_use = engine.prepare_voice_prompt(source_type="pt", pt_path=pt_path)
                        elif ref_audio:
                            prompt_to_use = engine.prepare_voice_prompt(model=model, source_type="audio", ref_audio=ref_audio, ref_text=ref_text)
                        else:
                            return gr.update(), "⚠️ 缺少音色来源，无法进行重算！"
                    else:
                        prompt_to_use = engine.cached_prompt

                lang_param = None if target_lang in ("Auto", "auto") else target_lang
                try:
                    # Update item text and role
                    target_item = None
                    for it in engine.items:
                        if it.index == row_index + 1:
                            target_item = it
                            break

                    if target_item:
                        target_item.text = cur_text.strip()
                        target_item.role_name = assigned_role if assigned_role != follow_global_label else None

                    kw: Dict[str, Any] = {
                        "text": cur_text.strip(),
                        "language": lang_param,
                        "voice_clone_prompt": prompt_to_use,
                    }
                    if speed_val != 1.0:
                        kw["speed"] = float(speed_val)
                    if instruct_text and instruct_text.strip():
                        kw["instruct"] = instruct_text.strip()

                    audios = model.generate(**kw)
                    waveform = audios[0]
                    if waveform.ndim > 1:
                        waveform = waveform.squeeze()

                    if target_item:
                        target_item.waveform = waveform
                        target_item.duration = len(waveform) / sampling_rate
                        target_item.status = "done"

                    int16_wave = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
                    wav_filename = f"sentence_regen_{row_index + 1:03d}_{int(time.time() * 1000)}.wav"
                    wav_path = os.path.join(studio_audio_dir, wav_filename)
                    sf.write(wav_path, int16_wave, sampling_rate)

                    msg = (
                        f"✅ **#{row_index + 1:02d}** 重跑完成！时长 {len(waveform) / sampling_rate:.2f}秒。"
                        if is_zh
                        else f"✅ **#{row_index + 1:02d}** regenerated successfully!"
                    )
                    return _render_audio_html(wav_path), msg
                except Exception as e:
                    logger.error(f"Error regenerating row {row_index + 1}: {e}", exc_info=True)
                    return gr.update(), f"❌ 重跑失败: {e}"

            return _regen

        for i in range(MAX_STUDIO_ROWS):
            row_regen_btns[i].click(
                fn=_make_regen_fn(i),
                inputs=[
                    row_text_boxes[i],
                    row_role_boxes[i],
                    studio_pt_select,
                    studio_pt_upload,
                    studio_ref_audio,
                    studio_ref_text,
                    studio_lang,
                    studio_speed,
                    studio_instruct,
                ],
                outputs=[row_audio_boxes[i], studio_status],
                show_progress="hidden",
            )

        # Export ZIP / Direct 1-Click Download (Invoked strictly on user click, downloading exactly once!)
        def _on_export_zip():
            cache_tmp_dir = os.path.join(_PROJECT_ROOT, ".cache", "tmp")
            os.makedirs(cache_tmp_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            zip_filename = os.path.join(cache_tmp_dir, f"sentence_studio_{timestamp}.zip")

            try:
                out_path = engine.export_zip(zip_filename)
                return out_path
            except Exception as e:
                logger.error(f"Export zip failed: {e}", exc_info=True)
                return None

        studio_zip_btn.click(
            fn=_on_export_zip,
            inputs=[],
            outputs=[studio_zip_btn],
            show_progress="hidden",
        )

        # Clear / Reset
        def _on_clear():
            engine.items = []
            engine.last_zip_path = None
            resets = [
                "<div style='font-size:13.5px; color:#cbd5e1; padding: 4px 0;'>ℹ️ 已清空重置。</div>" if is_zh else "<div style='font-size:13.5px; color:#cbd5e1; padding: 4px 0;'>ℹ️ Cleared.</div>",
                gr.update(value=None, visible=False),
                1,
                0,
                "📄 **第 1 / 1 页**" if is_zh else "📄 **Page 1 / 1**",
            ]
            for _ in range(MAX_STUDIO_ROWS):
                resets.extend([
                    gr.update(visible=False),
                    gr.update(value=follow_global_label),
                    gr.update(value=""),
                    gr.update(value=_render_audio_html(None)),
                ])
            return resets

        studio_clear_btn.click(
            fn=_on_clear,
            inputs=[],
            outputs=batch_full_outputs,
            show_progress="hidden",
        )
