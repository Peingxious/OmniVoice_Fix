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
"""Data structures for Sentence Studio."""

from dataclasses import dataclass, field
import re
from typing import Optional
import numpy as np


@dataclass
class SentenceItem:
    """Represents a single sentence unit in the Sentence Studio workflow.

    Attributes:
        index: 1-based sequential index (1, 2, ...).
        text: Current working text of the sentence (can be user-edited).
        raw_text: Original text parsed from the raw input before user edits.
        waveform: Generated audio waveform (1D numpy array), if generated.
        sample_rate: Sampling rate of the audio (default: 24000 Hz).
        duration: Audio duration in seconds.
        status: State ('pending', 'generating', 'done', 'error').
        error_msg: Error message if generation failed.
    """

    index: int
    text: str
    raw_text: str = ""
    waveform: Optional[np.ndarray] = None
    sample_rate: int = 24000
    duration: float = 0.0
    status: str = "pending"
    error_msg: Optional[str] = None
    role_name: Optional[str] = None

    def __post_init__(self):
        if not self.raw_text:
            self.raw_text = self.text

    @property
    def formatted_index(self) -> str:
        """Return 2-digit zero-padded index (e.g. '01', '02')."""
        return f"{self.index:02d}"

    @property
    def safe_filename_prefix(self) -> str:
        """Return clean prefix for file naming: '01_xxx' or '01_role_xxx'."""
        clean_text = re.sub(r'[<>:"/\\|?*\s\r\n。！？!?.,，、；;]+', '_', self.text.strip())
        short_text = clean_text[:15].strip('_')
        if not short_text:
            short_text = "sentence"

        role_tag = ""
        if self.role_name and self.role_name.strip() and self.role_name != "跟随全局":
            clean_role = re.sub(r'[<>:"/\\|?*\s\r\n.pt]+', '', self.role_name.strip())
            if clean_role:
                role_tag = f"_{clean_role}"

        return f"{self.formatted_index}{role_tag}_{short_text}"
