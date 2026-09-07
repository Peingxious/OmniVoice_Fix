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
"""ZIP packager and manifest exporter for Sentence Studio."""

import io
import json
import os
import zipfile
from typing import List, Optional
import numpy as np
import soundfile as sf

from omnivoice.sentence_studio.models import SentenceItem


def _waveform_to_wav_bytes(waveform: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Encode a 1D/2D numpy array waveform into 16-bit WAV bytes."""
    # Ensure 1D
    if waveform.ndim > 1:
        waveform = waveform.squeeze()

    # Convert float [-1.0, 1.0] or int16 to int16 format
    if waveform.dtype == np.float32 or waveform.dtype == np.float64:
        # Clip to prevent overflow distortion
        waveform = np.clip(waveform, -1.0, 1.0)
        pcm16 = (waveform * 32767).astype(np.int16)
    elif waveform.dtype == np.int16:
        pcm16 = waveform
    else:
        pcm16 = waveform.astype(np.int16)

    buffer = io.BytesIO()
    sf.write(buffer, pcm16, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def pack_sentence_audio_zip(
    items: List[SentenceItem],
    output_zip_path: str,
    include_manifest: bool = True,
) -> str:
    """Pack generated sentence audios into an ordered ZIP file.

    Files inside the ZIP are strictly ordered with zero-padded prefixes:
    `01_xxx.wav`, `02_xxx.wav`, etc.

    Args:
        items: List of SentenceItem objects.
        output_zip_path: Destination path for the .zip file.
        include_manifest: Whether to include manifest.json and list.txt.

    Returns:
        Absolute path to the created ZIP file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_zip_path)), exist_ok=True)

    manifest_entries = []
    list_lines = []
    total_duration = 0.0

    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in items:
            if item.waveform is None:
                continue

            wav_filename = f"{item.safe_filename_prefix}.wav"
            wav_bytes = _waveform_to_wav_bytes(item.waveform, item.sample_rate)
            zf.writestr(wav_filename, wav_bytes)

            duration = round(item.duration, 3)
            if duration <= 0 and len(item.waveform) > 0:
                duration = round(len(item.waveform) / item.sample_rate, 3)
            total_duration += duration

            manifest_entries.append({
                "index": item.index,
                "formatted_index": item.formatted_index,
                "filename": wav_filename,
                "text": item.text,
                "duration": duration,
                "sample_rate": item.sample_rate,
            })
            list_lines.append(f"{item.formatted_index}: {item.text}")

        if include_manifest:
            manifest_data = {
                "total_sentences": len(manifest_entries),
                "total_duration_sec": round(total_duration, 3),
                "sample_rate": items[0].sample_rate if items else 24000,
                "sentences": manifest_entries,
            }
            manifest_json_str = json.dumps(manifest_data, ensure_ascii=False, indent=2)
            zf.writestr("manifest.json", manifest_json_str.encode("utf-8"))
            zf.writestr("list.txt", "\n".join(list_lines).encode("utf-8"))

    return os.path.abspath(output_zip_path)
