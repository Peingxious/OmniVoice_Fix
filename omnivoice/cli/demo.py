#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
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
"""
Gradio demo for OmniVoice.

Supports voice cloning and voice design.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import logging
import math
import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig, VoiceClonePrompt
from omnivoice.sentence_studio import create_sentence_studio_tab
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name


# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Saved voice prompts — kept inside the project folder (voices/)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
VOICE_DIR = os.path.join(_PROJECT_ROOT, "voices")


def _clean_filename(name: str) -> str:
    """Sanitize character name to be a safe filename."""
    if not name or not name.strip():
        return ""
    cleaned = re.sub(r'[<>:"/\\|?*\s]+', '_', name.strip())
    return cleaned.strip('_')


def list_saved_voices() -> List[str]:
    """Return all .pt files in voices/, sorted by newest first."""
    os.makedirs(VOICE_DIR, exist_ok=True)
    files = [f for f in os.listdir(VOICE_DIR) if f.endswith(".pt")]
    files.sort(
        key=lambda x: os.path.getmtime(os.path.join(VOICE_DIR, x)),
        reverse=True,
    )
    return files


# ---------------------------------------------------------------------------
# In-memory prompt & info caches for zero-latency role switching
# ---------------------------------------------------------------------------
_PROMPT_CACHE: Dict[str, tuple] = {}
_VOICE_INFO_CACHE: Dict[tuple, tuple] = {}


def get_cached_prompt(path: str) -> Optional[VoiceClonePrompt]:
    """Load VoiceClonePrompt with memory caching based on file mtime."""
    if not path or not os.path.isfile(path):
        logger.warning(f"Role file not found: {path}")
        return None
    mtime = os.path.getmtime(path)
    cached = _PROMPT_CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        prompt = VoiceClonePrompt.load(path)
        _PROMPT_CACHE[path] = (mtime, prompt)
        return prompt
    except Exception as e:
        logger.error(f"Failed to load VoiceClonePrompt from {path}: {e}")
        return None


def invalidate_voice_cache(path: Optional[str] = None):
    """Invalidate memory cache when a role is trained or modified."""
    if path:
        _PROMPT_CACHE.pop(path, None)
    else:
        _PROMPT_CACHE.clear()
    _VOICE_INFO_CACHE.clear()


def get_voice_info_markdown(filename_or_path: Optional[str], active_lang: str = "zh") -> str:
    """Extract voice prompt details and format as readable Markdown."""
    if not filename_or_path:
        return (
            "*(请选择已训练的角色或上传 .pt 文件以查看详情)*"
            if active_lang == "zh"
            else "*(Please select or upload a role .pt file to view details)*"
        )

    if os.path.isabs(filename_or_path) and os.path.isfile(filename_or_path):
        path = filename_or_path
    else:
        path = os.path.join(VOICE_DIR, filename_or_path)

    if not os.path.isfile(path):
        return (
            f"*(文件不存在: `{os.path.basename(path)}`)*"
            if active_lang == "zh"
            else f"*(File not found: `{os.path.basename(path)}`)*"
        )

    cache_key = (path, active_lang)
    mtime = os.path.getmtime(path)
    cached = _VOICE_INFO_CACHE.get(cache_key)
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        prompt = get_cached_prompt(path)
        if prompt is None:
            return (
                f"*(角色文件无法解析: `{os.path.basename(path)}`)*"
                if active_lang == "zh"
                else f"*(Cannot parse role file: `{os.path.basename(path)}`)*"
            )
        num_tokens = prompt.ref_audio_tokens.shape[-1]
        est_sec = num_tokens / 12.5
        size_kb = os.path.getsize(path) / 1024.0
        text = prompt.ref_text or ("(无文本记录)" if active_lang == "zh" else "(No transcript)")
        preview_text = text[:140] + ("..." if len(text) > 140 else "")

        if active_lang == "zh":
            res = (
                f"**👤 角色文件**：`{os.path.basename(path)}`（大小：`{size_kb:.1f} KB`）\n\n"
                f"- **声纹总长度**：`{num_tokens}` 帧（约 `{est_sec:.1f}` 秒原声特征）\n"
                f"- **参考 RMS 响度**：`{prompt.ref_rms:.4f}`\n"
                f"- **参考文本内容**：\n> {preview_text}"
            )
        else:
            res = (
                f"**👤 Role File**: `{os.path.basename(path)}` (Size: `{size_kb:.1f} KB`)\n\n"
                f"- **Voice Duration**: `{num_tokens}` frames (~`{est_sec:.1f}`s features)\n"
                f"- **Reference RMS**: `{prompt.ref_rms:.4f}`\n"
                f"- **Transcript**:\n> {preview_text}"
            )
        _VOICE_INFO_CACHE[cache_key] = (mtime, res)
        return res
    except Exception as e:
        return (
            f"⚠️ 读取角色信息失败: {type(e).__name__}: {e}"
            if active_lang == "zh"
            else f"⚠️ Failed to read role info: {type(e).__name__}: {e}"
        )


# ---------------------------------------------------------------------------
# Voice design attributes & i18n
# ---------------------------------------------------------------------------
# Display text is decoupled from what the model receives: the UI shows the
# selected language, while ``CATEGORIES[i]["options"][j]["value"]`` is always
# what gets sent (English for most attributes, Chinese for dialects).
from omnivoice.cli.i18n import (  # noqa: E402
    AUTO_TEXT,
    CATEGORIES,
    LANG_CHOICES,
    MSG,
    UI,
    category_choices,
    display_to_value,
    get_saved_lang,
    save_settings,
    value_to_display,
)

# Mutable holder for the active UI language (avoids `global` statements).
_LANG = {"current": get_saved_lang()}


def _m(key, *args):
    """Return a runtime message in the active UI language."""
    return MSG[key][_LANG["current"]] % args if args else MSG[key][_LANG["current"]]


def _is_auto(value) -> bool:
    """True when a target-language dropdown is left on Auto."""
    return not value or value in (AUTO_TEXT["zh"], AUTO_TEXT["en"], "Auto")


# "中文" -> "zh", "English" -> "en"
_LANG_FROM_LABEL = {label: code for code, label in LANG_CHOICES}

# Gradio theme with modern dark slate styling and refined typography
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
).set(
    body_background_fill="#090d16",
    body_text_color="#f1f5f9",
    block_background_fill="#0f172a",
    block_border_color="rgba(148, 163, 184, 0.12)",
    block_radius="14px",
    block_shadow="0 4px 20px rgba(0, 0, 0, 0.25)",
    button_primary_background_fill="linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
    button_primary_text_color="#ffffff",
    button_primary_border_color="transparent",
    button_secondary_background_fill="rgba(30, 41, 59, 0.8)",
    button_secondary_background_fill_hover="rgba(51, 65, 85, 0.9)",
    button_secondary_text_color="#e2e8f0",
    button_secondary_border_color="rgba(148, 163, 184, 0.2)",
    input_background_fill="#1e293b",
    input_border_color="rgba(148, 163, 184, 0.2)",
    input_radius="10px",
    slider_color="#6366f1",
)

CSS = """
/* Global container & typography */
.gradio-container {
    max-width: 1420px !important;
    margin: 0 auto !important;
    padding-top: 14px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* Header hero banner */
.header-banner {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.7) 0%, rgba(46, 16, 101, 0.5) 50%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(12px);
}

.hero-chip {
    display: inline-flex;
    align-items: center;
    padding: 5px 13px;
    border-radius: 9999px;
    font-size: 12.5px;
    font-weight: 500;
    color: #e2e8f0;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(6px);
}

/* Modern segmented pill tab navigation */
.tab-nav, div[role="tablist"] {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    gap: 8px !important;
    margin-bottom: 22px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
}

.tab-nav button, div[role="tablist"] button {
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.tab-nav button:hover, div[role="tablist"] button:hover {
    color: #f8fafc !important;
    background: rgba(255, 255, 255, 0.06) !important;
}

.tab-nav button.selected, div[role="tablist"] button.selected {
    color: #ffffff !important;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: 1px solid rgba(192, 132, 252, 0.4) !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.45) !important;
}

/* Quick language buttons (Large & Spacious for Auto, Chinese, English) */
.quick-tag-btn-lg {
    border-radius: 12px !important;
    padding: 8px 18px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    min-width: 140px !important;
    height: 40px !important;
    background: rgba(30, 41, 59, 0.85) !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    color: #f1f5f9 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
.quick-tag-btn-lg:hover {
    background: rgba(99, 102, 241, 0.35) !important;
    border-color: rgba(129, 140, 248, 0.8) !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}

/* Preset intonation dynamic range buttons */
.preset-dyn-btn {
    border-radius: 10px !important;
    padding: 5px 10px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    color: #cbd5e1 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
.preset-dyn-btn:hover {
    background: rgba(99, 102, 241, 0.25) !important;
    border-color: rgba(129, 140, 248, 0.6) !important;
    color: #ffffff !important;
}

/* Primary generate / train action buttons */
button.primary, .primary-action-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 18px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.2s ease !important;
}
button.primary:hover, .primary-action-btn:hover {
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.6) !important;
    transform: translateY(-1px) !important;
}

/* Character info card container */
.role-info-card {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    font-size: 13.5px !important;
    line-height: 1.6 !important;
    margin-bottom: 10px !important;
}

/* Custom Elegant Scrollbar for Dark Theme */
*::-webkit-scrollbar {
    width: 6px !important;
    height: 6px !important;
}
*::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.6) !important;
    border-radius: 9999px !important;
}
*::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.4) !important;
    border-radius: 9999px !important;
}
*::-webkit-scrollbar-thumb:hover {
    background: rgba(129, 140, 248, 0.8) !important;
}

/* Audio Player Container & Waveform Full Display */
div[data-testid="audio"] {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}

div[data-testid="audio"] .component-wrapper {
    padding: 12px 16px !important;
}

div[data-testid="audio"] .waveform-container {
    min-height: 52px !important;
    position: relative !important;
    margin-bottom: 4px !important;
}

div[data-testid="audio"] .timestamps {
    margin-top: 8px !important;
    padding: 2px 4px !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    border-top: 1px solid rgba(148, 163, 184, 0.12) !important;
    padding-top: 6px !important;
    position: relative !important;
    z-index: 5 !important;
}

div[data-testid="audio"] #time,
div[data-testid="audio"] #duration,
div[data-testid="audio"] time {
    color: #cbd5e1 !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* Compact audio waveforms */
.compact-audio audio {height: 32px !important;}
.compact-audio .waveform {min-height: 36px !important;}

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

/* Studio Table Header & Compact Sentence Rows */
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

.studio-compact-row:hover {
    background: rgba(30, 41, 59, 0.6) !important;
    border-color: rgba(99, 102, 241, 0.3) !important;
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

.studio-compact-row input,
.studio-compact-row select {
    height: 32px !important;
    min-height: 32px !important;
    line-height: 32px !important;
    font-size: 13px !important;
    padding: 0 8px !important;
    border-radius: 6px !important;
    background: #1e293b !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    color: #f1f5f9 !important;
}

.studio-compact-row {
    min-height: 46px !important;
    max-height: 48px !important;
    height: 46px !important;
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
"""

HEAD_HTML = """
<script>
(function() {
    // Intercept shadow root creation in WaveSurfer to inject dark scrollbar styling
    const origAttachShadow = Element.prototype.attachShadow;
    Element.prototype.attachShadow = function(init) {
        const shadow = origAttachShadow.apply(this, arguments);
        try {
            const style = document.createElement('style');
            style.textContent = `
                :host {
                    position: relative !important;
                }
                :host .scroll {
                    overflow-x: auto !important;
                    overflow-y: hidden !important;
                    scrollbar-width: thin !important;
                    scrollbar-color: rgba(99, 102, 241, 0.45) rgba(15, 23, 42, 0.3) !important;
                    padding-bottom: 2px !important;
                }
                :host .scroll::-webkit-scrollbar {
                    height: 5px !important;
                    background: rgba(15, 23, 42, 0.3) !important;
                }
                :host .scroll::-webkit-scrollbar-track {
                    background: rgba(15, 23, 42, 0.4) !important;
                    border-radius: 9999px !important;
                }
                :host .scroll::-webkit-scrollbar-thumb {
                    background: rgba(99, 102, 241, 0.45) !important;
                    border-radius: 9999px !important;
                }
                :host .scroll::-webkit-scrollbar-thumb:hover {
                    background: rgba(129, 140, 248, 0.8) !important;
                }
            `;
            shadow.appendChild(style);
        } catch(e) {}
        return shadow;
    };
})();
</script>
"""


def _ensure_audio_waveform_patch():
    """Ensure Gradio frontend assets use responsive full-width waveform (minPxPerSec: 0)."""
    try:
        import gradio as gr
        frontend_dir = os.path.join(os.path.dirname(gr.__file__), "templates", "frontend", "assets")
        if not os.path.isdir(frontend_dir):
            return
        for fname in os.listdir(frontend_dir):
            if fname.endswith(".js") and ("index" in fname.lower() or "audio" in fname.lower()):
                fpath = os.path.join(frontend_dir, fname)
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if "minPxPerSec: 20" in content:
                    new_content = content.replace("minPxPerSec: 20", "minPxPerSec: 0")
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    logger.info(f"Patched full-width audio waveform in {fname}")
    except Exception as e:
        logger.warning(f"Waveform auto-patch notice: {e}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR model. Reference text auto-transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model path or HuggingFace repo id"
        " (default: openai/whisper-large-v3-turbo).",
    )
    parser.add_argument(
        "--lang",
        default=None,
        choices=["zh", "en"],
        help="Interface language: zh (Chinese) or en (English). Defaults to"
        " the value saved in settings.json, or zh on first run.",
    )
    return parser


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
    lang: str | None = None,
) -> gr.Blocks:
    sampling_rate = model.sampling_rate

    lang = lang if lang in ("zh", "en") else _LANG["current"]

    # Every labelled component registers here so the whole UI can be
    # re-labelled in one pass when the interface language changes.
    registry: list = []

    def _reg(comp, key):
        registry.append((comp, key))
        return comp

    def _apply_lang(new_lang, *cat_values):
        """Re-label every registered component for ``new_lang``."""
        nonlocal lang
        new_lang = _LANG_FROM_LABEL.get(new_lang, new_lang)
        if new_lang not in ("zh", "en"):
            return [gr.update()] * len(registry)
        lang = new_lang
        _LANG["current"] = new_lang
        save_settings({"language": new_lang})

        updates = []
        cat_i = 0
        for _comp, key in registry:
            kw = dict(UI[key][new_lang])
            if key == "language":
                # Speech language list: keep the names, translate "Auto".
                kw["choices"] = [AUTO_TEXT[new_lang]] + _ALL_LANGUAGES[1:]
                kw["value"] = AUTO_TEXT[new_lang]
            elif key.startswith("cat_"):
                idx = int(key[4:])
                kw["choices"] = category_choices(idx, new_lang)
                current = cat_values[cat_i] if cat_i < len(cat_values) else None
                kw["value"] = value_to_display(
                    display_to_value(current), new_lang, idx
                )
                cat_i += 1

            # Gradio validates incoming values against the choices captured
            # when the component was built, so the live component's list has
            # to follow the language switch too -- `gr.update` only reaches
            # the frontend.
            if "choices" in kw:
                try:
                    # Gradio normalizes choices into (display, value) pairs at
                    # build time; mirror that here or api_info()/validation
                    # will index into plain strings and break.
                    _comp.choices = [(str(c), c) for c in kw["choices"]]
                except Exception:
                    pass

            updates.append(gr.update(**kw))
        return updates

    # -- shared generation core --
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        ref_text=None,
        voice_clone_prompt=None,
        class_temperature=0.35,
    ):
        # Determine chunking threshold:
        # Long text (e.g. >100 characters, or multiple newlines/sentences) should chunk
        # cleanly into ~12s segments to avoid diffusion attention collapse and silence cutoffs.
        raw_text = text.strip()
        num_chars = len(raw_text)
        has_multiple_lines = raw_text.count("\n") >= 2
        chunk_thresh = 15.0 if (num_chars > 120 or has_multiple_lines) else 25.0

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
            class_temperature=float(class_temperature if class_temperature is not None else 0.35),
            audio_chunk_duration=12.0,
            audio_chunk_threshold=chunk_thresh,
        )

        lang = None if _is_auto(language) else language

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if not ref_audio:
                return None, _m("need_ref_audio")
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
        elif mode == "pt":
            if voice_clone_prompt is None:
                return None, _m("no_pt_selected")
            kw["voice_clone_prompt"] = voice_clone_prompt

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, _m("error", f"{type(e).__name__}: {e}")

        waveform = (audio[0] * 32767).astype(np.int16)
        return (sampling_rate, waveform), _m("done")

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

    # =====================================================================
    # UI
    # =====================================================================
    # Reusable: target-language dropdown (language OF THE SPEECH, not the UI)
    def _lang_dropdown(key="language"):
        s = UI[key][lang]
        return _reg(
            gr.Dropdown(
                label=s["label"],
                info=s.get("info"),
                choices=[AUTO_TEXT[lang]] + _ALL_LANGUAGES[1:],
                value=AUTO_TEXT[lang],
                allow_custom_value=False,
                interactive=True,
            ),
            key,
        )

    # Reusable: optional generation settings accordion
    def _gen_settings():
        with _reg(
            gr.Accordion(UI["gen_settings"][lang]["label"], open=False),
            "gen_settings",
        ):
            sp = _reg(
                gr.Slider(0.5, 1.5, value=1.0, step=0.05, **UI["speed"][lang]),
                "speed",
            )
            du = _reg(gr.Number(value=None, **UI["duration"][lang]), "duration")
            ns = _reg(
                gr.Slider(4, 64, value=32, step=1, **UI["steps"][lang]),
                "steps",
            )
            dn = _reg(gr.Checkbox(value=True, **UI["denoise"][lang]), "denoise")
            gs = _reg(
                gr.Slider(0.0, 4.0, value=2.0, step=0.1, **UI["guidance"][lang]),
                "guidance",
            )
            pp = _reg(
                gr.Checkbox(value=True, **UI["preprocess"][lang]),
                "preprocess",
            )
            po = _reg(
                gr.Checkbox(value=True, **UI["postprocess"][lang]),
                "postprocess",
            )
        return ns, gs, dn, sp, du, pp, po

    _ensure_audio_waveform_patch()

    with gr.Blocks(
        title=("OmniVoice 语音合成" if lang == "zh" else "OmniVoice Demo"),
    ) as demo:
        _reg(gr.Markdown(UI["intro"][lang]["value"]), "intro")

        with gr.Tabs():
            # ==============================================================
            # Tab 1: 声音克隆 (Voice Clone)
            # ==============================================================
            with _reg(gr.TabItem(UI["tab_clone"][lang]["label"]), "tab_clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = _reg(
                            gr.Textbox(lines=4, **UI["text"][lang]), "text"
                        )

                        vc_ref_audio = _reg(
                            gr.Audio(
                                label=UI["ref_audio"][lang]["label"],
                                type="filepath",
                                elem_classes="compact-audio",
                            ),
                            "ref_audio",
                        )
                        _reg(
                            gr.Markdown(UI["ref_audio_hint"][lang]["value"]),
                            "ref_audio_hint",
                        )

                        with gr.Group():
                            gr.Markdown(UI["quick_lang_label"][lang]["value"])
                            with gr.Row():
                                vc_quick_btns = []
                                for chip_label, chip_val in [
                                    ("🌐 自动检测", AUTO_TEXT[lang]),
                                    ("🇨🇳 中文 (Chinese)", "Chinese"),
                                    ("🇬🇧 英文 (English)", "English"),
                                ]:
                                    b = gr.Button(chip_label, size="sm", elem_classes="quick-tag-btn-lg")
                                    vc_quick_btns.append((b, chip_val))

                        vc_lang = _lang_dropdown()
                        for b, val in vc_quick_btns:
                            b.click(lambda v=val: v, outputs=[vc_lang])

                        with gr.Group():
                            vc_expressiveness = gr.Slider(
                                minimum=0.0,
                                maximum=0.8,
                                value=0.35,
                                step=0.05,
                                label="🎭 抑扬顿挫起伏度 (自然起伏区间)" if lang == "zh" else "🎭 Intonation Dynamics Range",
                                info="0.0 平直叙述 ｜ 0.35 推荐自然口语（默认） ｜ 0.55 生动起伏 ｜ 0.75 鲜明富有张力" if lang == "zh" else "0.0 Flat | 0.35 Natural (Default) | 0.55 Lively | 0.75 Dynamic",
                            )
                            with gr.Row():
                                vc_p1 = gr.Button("平缓叙述 (0.0)", size="sm", elem_classes="preset-dyn-btn")
                                vc_p2 = gr.Button("自然抑扬 (0.35 - 推荐)", size="sm", elem_classes="preset-dyn-btn")
                                vc_p3 = gr.Button("生动起伏 (0.55)", size="sm", elem_classes="preset-dyn-btn")
                                vc_p4 = gr.Button("充沛高昂 (0.75)", size="sm", elem_classes="preset-dyn-btn")
                            vc_p1.click(lambda: 0.0, outputs=[vc_expressiveness])
                            vc_p2.click(lambda: 0.35, outputs=[vc_expressiveness])
                            vc_p3.click(lambda: 0.55, outputs=[vc_expressiveness])
                            vc_p4.click(lambda: 0.75, outputs=[vc_expressiveness])

                        with _reg(
                            gr.Accordion(
                                UI["ref_instruct_accordion"][lang]["label"],
                                open=False,
                            ),
                            "ref_instruct_accordion",
                        ):
                            vc_ref_text = _reg(
                                gr.Textbox(lines=2, **UI["ref_text"][lang]),
                                "ref_text",
                            )
                            vc_transcribe_btn = _reg(
                                gr.Button(
                                    "🎙️ 自动识别参考音频台词 (Whisper ASR)" if lang == "zh" else "🎙️ Auto-Transcribe Reference Audio",
                                    size="sm",
                                    variant="secondary",
                                ),
                                "transcribe_btn",
                            )
                            vc_instruct = _reg(
                                gr.Textbox(lines=2, **UI["instruct"][lang]),
                                "instruct",
                            )
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                        ) = _gen_settings()
                        vc_btn = _reg(
                            gr.Button(
                                UI["generate"][lang]["value"],
                                variant="primary",
                                elem_classes="primary-action-btn",
                            ),
                            "generate",
                        )
                        with _reg(
                            gr.Accordion(
                                UI["save_accordion"][lang]["label"], open=False
                            ),
                            "save_accordion",
                        ):
                            with gr.Row():
                                vc_save_name = _reg(
                                    gr.Textbox(
                                        label="角色名称" if lang == "zh" else "Role Name",
                                        placeholder="输入角色名 (如: 杨总、科技男声；留空自动使用时间戳)" if lang == "zh" else "Enter role name (e.g. CEO_Yang, Narrator; empty for timestamp)",
                                        lines=1,
                                        scale=3,
                                    ),
                                    "save_char_name",
                                )
                                vc_save_btn = _reg(
                                    gr.Button(
                                        UI["save_btn"][lang]["value"],
                                        variant="secondary",
                                        scale=1,
                                    ),
                                    "save_btn",
                                )
                            vc_save_file = _reg(
                                gr.File(
                                    label=UI["save_file"][lang]["label"],
                                    interactive=False,
                                ),
                                "save_file",
                            )
                            _reg(
                                gr.Markdown(UI["save_hint"][lang]["value"]),
                                "save_hint",
                            )
                    with gr.Column(scale=1):
                        vc_audio = _reg(
                            gr.Audio(
                                label=UI["output_audio"][lang]["label"],
                                type="numpy",
                            ),
                            "output_audio",
                        )
                        vc_status = _reg(
                            gr.Textbox(**UI["status"][lang], lines=2), "status"
                        )

                def _clone_fn(
                    text, lang, ref_aud, ref_text, instruct, expressiveness, ns, gs, dn, sp, du, pp, po
                ):
                    return _gen(
                        text,
                        lang,
                        ref_aud,
                        instruct,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="clone",
                        ref_text=ref_text or None,
                        class_temperature=float(expressiveness if expressiveness is not None else 0.35),
                    )

                def _transcribe_fn(aud):
                    if not aud:
                        return "", _m("transcribe_no_audio")
                    if model._asr_pipe is None:
                        return "", _m("transcribe_no_asr")
                    try:
                        res = model.transcribe(aud)
                        return res, _m("transcribe_done")
                    except Exception as e:
                        return "", _m("error", f"{type(e).__name__}: {e}")

                vc_transcribe_btn.click(
                    _transcribe_fn,
                    inputs=[vc_ref_audio],
                    outputs=[vc_ref_text, vc_status],
                )

                vc_btn.click(
                    _clone_fn,
                    inputs=[
                        vc_text,
                        vc_lang,
                        vc_ref_audio,
                        vc_ref_text,
                        vc_instruct,
                        vc_expressiveness,
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                    ],
                    outputs=[vc_audio, vc_status],
                )

                def _save_voice_fn(ref_aud, ref_text, char_name):
                    """Save the reference voice as a reusable `.pt` prompt with custom role name."""
                    if not ref_aud:
                        return None, _m("need_ref_audio_for_save"), gr.update(), ""
                    try:
                        prompt = model.create_voice_clone_prompt(
                            ref_audio=ref_aud,
                            ref_text=ref_text or None,
                        )
                        os.makedirs(VOICE_DIR, exist_ok=True)
                        cleaned = _clean_filename(char_name) if char_name else ""
                        if cleaned:
                            filename = f"{cleaned}.pt"
                        else:
                            stamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"voice_{stamp}.pt"

                        path = os.path.join(VOICE_DIR, filename)
                        if os.path.exists(path):
                            stamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"{cleaned}_{stamp}.pt" if cleaned else f"voice_{stamp}.pt"
                            path = os.path.join(VOICE_DIR, filename)

                        prompt.save(path)
                        invalidate_voice_cache(path)
                        size_kb = os.path.getsize(path) / 1024.0
                        new_choices = list_saved_voices()
                        info_md = get_voice_info_markdown(path, lang)
                        return (
                            path,
                            _m("save_ok", filename, size_kb),
                            gr.update(choices=new_choices, value=filename),
                            info_md,
                        )
                    except Exception as e:
                        return None, _m("save_failed", f"{type(e).__name__}: {e}"), gr.update(), ""

            # ==============================================================
            # Tab 2: 分句配音工作室 (Sentence Studio)
            # ==============================================================
            create_sentence_studio_tab(
                model=model,
                sampling_rate=sampling_rate,
                lang=lang,
                all_languages=_ALL_LANGUAGES,
                default_lang_label=AUTO_TEXT[lang],
                list_saved_voices_fn=list_saved_voices,
                get_voice_info_fn=get_voice_info_markdown,
                voice_dir=VOICE_DIR,
            )

            # ==============================================================
            # Tab 3: 角色调用生成 (Direct PT Voice Role Synthesis)
            # ==============================================================
            with _reg(gr.TabItem(UI["tab_pt"][lang]["label"]), "tab_pt"):
                with gr.Row():
                    with gr.Column(scale=1):
                        saved_voices = list_saved_voices()
                        with gr.Row():
                            pt_select = _reg(
                                gr.Dropdown(
                                    label=UI["pt_select"][lang]["label"],
                                    info=UI["pt_select"][lang].get("info"),
                                    choices=saved_voices,
                                    value=saved_voices[0] if saved_voices else None,
                                    interactive=True,
                                    scale=4,
                                ),
                                "pt_select",
                            )
                            pt_refresh_btn = _reg(
                                gr.Button(
                                    UI["pt_refresh_btn"][lang]["value"],
                                    scale=1,
                                ),
                                "pt_refresh_btn",
                            )
                        with gr.Accordion(UI["pt_upload"][lang]["label"], open=False):
                            pt_upload = _reg(
                                gr.File(
                                    label=UI["pt_upload"][lang]["label"],
                                    file_types=[".pt"],
                                    type="filepath",
                                ),
                                "pt_upload",
                            )
                        pt_info = _reg(
                            gr.Markdown(
                                get_voice_info_markdown(
                                    saved_voices[0] if saved_voices else None,
                                    lang,
                                ),
                                elem_classes="role-info-card",
                            ),
                            "pt_info",
                        )
                        pt_text = _reg(
                            gr.Textbox(lines=4, **UI["pt_text"][lang]), "pt_text"
                        )


                        with gr.Accordion(
                            "🎙️ 进阶语气引导 (可选，默认留空以100%保留角色原声讲述情绪)" if lang == "zh" else "🎙️ Tone Guidance (Optional, leave empty to preserve voice emotion)",
                            open=False,
                        ):
                            pt_instruct = gr.Textbox(
                                label="语气引导提示词" if lang == "zh" else "Tone Instruction",
                                placeholder="通常留空即可，依靠上方行内情绪标记精准表达；仅在需要整体风格微调时填写" if lang == "zh" else "Usually leave empty; inline emotion tags handle expressive moments",
                                lines=1,
                            )

                        # --- Quick Language Tags & Dropdown ---
                        with gr.Group():
                            gr.Markdown(UI["quick_lang_label"][lang]["value"])
                            with gr.Row():
                                pt_quick_btns = []
                                for chip_label, chip_val in [
                                    ("🌐 自动检测", AUTO_TEXT[lang]),
                                    ("🇨🇳 中文 (Chinese)", "Chinese"),
                                    ("🇬🇧 英文 (English)", "English"),
                                ]:
                                    qb = gr.Button(chip_label, size="sm", elem_classes="quick-tag-btn-lg")
                                    pt_quick_btns.append((qb, chip_val))

                        pt_lang = _lang_dropdown()
                        for qb, val in pt_quick_btns:
                            qb.click(lambda v=val: v, outputs=[pt_lang])

                        with gr.Group():
                            pt_expressiveness = gr.Slider(
                                minimum=0.0,
                                maximum=0.8,
                                value=0.35,
                                step=0.05,
                                label="🎭 抑扬顿挫起伏度 (自然起伏区间)" if lang == "zh" else "🎭 Intonation Dynamics Range",
                                info="0.0 平直叙述 ｜ 0.35 推荐自然口语（默认） ｜ 0.55 生动起伏 ｜ 0.75 鲜明富有张力" if lang == "zh" else "0.0 Flat | 0.35 Natural (Default) | 0.55 Lively | 0.75 Dynamic",
                            )
                            with gr.Row():
                                pt_p1 = gr.Button("平缓叙述 (0.0)", size="sm", elem_classes="preset-dyn-btn")
                                pt_p2 = gr.Button("自然抑扬 (0.35 - 推荐)", size="sm", elem_classes="preset-dyn-btn")
                                pt_p3 = gr.Button("生动起伏 (0.55)", size="sm", elem_classes="preset-dyn-btn")
                                pt_p4 = gr.Button("充沛高昂 (0.75)", size="sm", elem_classes="preset-dyn-btn")
                            pt_p1.click(lambda: 0.0, outputs=[pt_expressiveness])
                            pt_p2.click(lambda: 0.35, outputs=[pt_expressiveness])
                            pt_p3.click(lambda: 0.55, outputs=[pt_expressiveness])
                            pt_p4.click(lambda: 0.75, outputs=[pt_expressiveness])

                        (
                            pt_ns,
                            pt_gs,
                            pt_dn,
                            pt_sp,
                            pt_du,
                            pt_pp,
                            pt_po,
                        ) = _gen_settings()
                        pt_gen_btn = _reg(
                            gr.Button(
                                UI["pt_gen_btn"][lang]["value"],
                                variant="primary",
                                elem_classes="primary-action-btn",
                            ),
                            "pt_gen_btn",
                        )
                    with gr.Column(scale=1):
                        pt_audio = _reg(
                            gr.Audio(
                                label=UI["output_audio"][lang]["label"],
                                type="numpy",
                            ),
                            "output_audio",
                        )
                        pt_status = _reg(
                            gr.Textbox(**UI["status"][lang], lines=2), "status"
                        )

                vc_save_btn.click(
                    _save_voice_fn,
                    inputs=[vc_ref_audio, vc_ref_text, vc_save_name],
                    outputs=[vc_save_file, vc_status, pt_select, pt_info],
                )

                def _refresh_voices_fn():
                    v = list_saved_voices()
                    val = v[0] if v else None
                    info = get_voice_info_markdown(val, lang)
                    return (
                        gr.update(choices=v, value=val),
                        info,
                    )

                pt_refresh_btn.click(
                    _refresh_voices_fn,
                    outputs=[pt_select, pt_info],
                )

                demo.load(
                    _refresh_voices_fn,
                    outputs=[pt_select, pt_info],
                )

                pt_select.change(
                    lambda val: get_voice_info_markdown(val, lang),
                    inputs=[pt_select],
                    outputs=[pt_info],
                )

                pt_upload.change(
                    lambda f, sel: get_voice_info_markdown(f, lang) if f else get_voice_info_markdown(sel, lang),
                    inputs=[pt_upload, pt_select],
                    outputs=[pt_info],
                )

                def _pt_infer_fn(
                    selected_voice,
                    uploaded_pt,
                    text,
                    lang_choice,
                    instruct,
                    expressiveness,
                    ns,
                    gs,
                    dn,
                    sp,
                    du,
                    pp,
                    po,
                ):
                    pt_path = None
                    if uploaded_pt is not None:
                        pt_path = uploaded_pt.name if hasattr(uploaded_pt, "name") else str(uploaded_pt)
                    elif selected_voice:
                        candidate = os.path.join(VOICE_DIR, selected_voice)
                        if os.path.isfile(candidate):
                            pt_path = candidate

                    if not pt_path or not os.path.isfile(pt_path):
                        return None, _m("no_pt_selected")

                    try:
                        prompt = get_cached_prompt(pt_path)
                    except Exception as e:
                        return None, _m("error", f"Failed to load PT: {e}")

                    effective_instruct = instruct.strip() if instruct and instruct.strip() else None

                    return _gen(
                        text,
                        lang_choice,
                        None,
                        effective_instruct,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="pt",
                        ref_text=None,
                        voice_clone_prompt=prompt,
                        class_temperature=float(expressiveness if expressiveness is not None else 0.35),
                    )

                pt_gen_btn.click(
                    _pt_infer_fn,
                    inputs=[
                        pt_select,
                        pt_upload,
                        pt_text,
                        pt_lang,
                        pt_instruct,
                        pt_expressiveness,
                        pt_ns,
                        pt_gs,
                        pt_dn,
                        pt_sp,
                        pt_du,
                        pt_pp,
                        pt_po,
                    ],
                    outputs=[pt_audio, pt_status],
                )

            # ==============================================================
            # Voice Design
            # ==============================================================
            with _reg(gr.TabItem(UI["tab_design"][lang]["label"]), "tab_design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = _reg(
                            gr.Textbox(lines=4, **UI["text"][lang]), "text"
                        )

                        with gr.Group():
                            gr.Markdown(UI["quick_lang_label"][lang]["value"])
                            with gr.Row():
                                vd_quick_btns = []
                                for chip_label, chip_val in [
                                    ("🌐 自动检测", AUTO_TEXT[lang]),
                                    ("🇨🇳 中文 (Chinese)", "Chinese"),
                                    ("🇬🇧 英文 (English)", "English"),
                                ]:
                                    b = gr.Button(chip_label, size="sm", elem_classes="quick-tag-btn-lg")
                                    vd_quick_btns.append((b, chip_val))

                        vd_lang = _lang_dropdown()
                        for b, val in vd_quick_btns:
                            b.click(lambda v=val: v, outputs=[vd_lang])

                        with gr.Group():
                            vd_expressiveness = gr.Slider(
                                minimum=0.0,
                                maximum=0.8,
                                value=0.35,
                                step=0.05,
                                label="🎭 抑扬顿挫起伏度 (自然起伏区间)" if lang == "zh" else "🎭 Intonation Dynamics Range",
                                info="0.0 平直叙述 ｜ 0.35 推荐自然口语（默认） ｜ 0.55 生动起伏 ｜ 0.75 鲜明富有张力" if lang == "zh" else "0.0 Flat | 0.35 Natural (Default) | 0.55 Lively | 0.75 Dynamic",
                            )
                            with gr.Row():
                                vd_p1 = gr.Button("平缓叙述 (0.0)", size="sm", elem_classes="preset-dyn-btn")
                                vd_p2 = gr.Button("自然抑扬 (0.35 - 推荐)", size="sm", elem_classes="preset-dyn-btn")
                                vd_p3 = gr.Button("生动起伏 (0.55)", size="sm", elem_classes="preset-dyn-btn")
                                vd_p4 = gr.Button("充沛高昂 (0.75)", size="sm", elem_classes="preset-dyn-btn")
                            vd_p1.click(lambda: 0.0, outputs=[vd_expressiveness])
                            vd_p2.click(lambda: 0.35, outputs=[vd_expressiveness])
                            vd_p3.click(lambda: 0.55, outputs=[vd_expressiveness])
                            vd_p4.click(lambda: 0.75, outputs=[vd_expressiveness])

                        vd_groups = []
                        with _reg(
                            gr.Accordion(
                                UI["attr_accordion"][lang]["label"], open=False
                            ),
                            "attr_accordion",
                        ):
                            for _i, _cat in enumerate(CATEGORIES):
                                _key = "cat_%d" % _i
                                vd_groups.append(
                                    _reg(
                                        gr.Dropdown(
                                            label=UI[_key][lang]["label"],
                                            info=UI[_key][lang].get("info"),
                                            choices=category_choices(_i, lang),
                                            value=AUTO_TEXT[lang],
                                        ),
                                        _key,
                                    )
                                )

                        (
                            vd_ns,
                            vd_gs,
                            vd_dn,
                            vd_sp,
                            vd_du,
                            vd_pp,
                            vd_po,
                        ) = _gen_settings()
                        vd_btn = _reg(
                            gr.Button(
                                UI["generate"][lang]["value"],
                                variant="primary",
                                elem_classes="primary-action-btn",
                            ),
                            "generate",
                        )
                    with gr.Column(scale=1):
                        vd_audio = _reg(
                            gr.Audio(
                                label=UI["output_audio"][lang]["label"],
                                type="numpy",
                            ),
                            "output_audio",
                        )
                        vd_status = _reg(
                            gr.Textbox(**UI["status"][lang], lines=2), "status"
                        )

                def _build_instruct(groups):
                    """Map selected display labels to the text sent to the model.

                    Language-agnostic: each option carries a stable ``value``,
                    so the same code works for the Chinese and English UI.
                    """
                    parts = []
                    for g in groups:
                        v = display_to_value(g)
                        if v:
                            parts.append(v)
                    return ", ".join(parts) if parts else None

                def _design_fn(text, lang, expressiveness, ns, gs, dn, sp, du, pp, po, *groups):
                    return _gen(
                        text,
                        lang,
                        None,
                        _build_instruct(groups),
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="design",
                        class_temperature=float(expressiveness if expressiveness is not None else 0.35),
                    )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text,
                        vd_lang,
                        vd_expressiveness,
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                    ]
                    + vd_groups,
                    outputs=[vd_audio, vd_status],
                )



    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    print("Model loaded.")

    lang = args.lang or get_saved_lang()
    logging.info(f"Interface language: {lang} (persisted in settings.json)")
    demo = build_demo(model, checkpoint, lang=lang)

    # `allowed_paths` lets Gradio serve saved voice prompts and generated audios from .cache
    os.makedirs(VOICE_DIR, exist_ok=True)
    cache_dir = os.path.abspath(os.path.join(_PROJECT_ROOT, ".cache"))
    os.makedirs(cache_dir, exist_ok=True)
    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
        allowed_paths=[VOICE_DIR, cache_dir, os.path.abspath(_PROJECT_ROOT)],
        theme=THEME,
        css=CSS,
        head=HEAD_HTML,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
