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
"""Sentence Studio package: modular sentence-by-sentence voice synthesis and export."""

from omnivoice.sentence_studio.models import SentenceItem
from omnivoice.sentence_studio.text_processor import clean_and_split_text
from omnivoice.sentence_studio.packager import pack_sentence_audio_zip
from omnivoice.sentence_studio.engine import SentenceStudioEngine
from omnivoice.sentence_studio.ui import create_sentence_studio_tab

__all__ = [
    "SentenceItem",
    "clean_and_split_text",
    "pack_sentence_audio_zip",
    "SentenceStudioEngine",
    "create_sentence_studio_tab",
]
