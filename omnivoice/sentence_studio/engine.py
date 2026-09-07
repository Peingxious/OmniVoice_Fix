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
"""Sentence synthesis orchestration engine and single-sentence retry controller."""

import logging
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from omnivoice.sentence_studio.models import SentenceItem
from omnivoice.sentence_studio.text_processor import clean_and_split_text
from omnivoice.sentence_studio.packager import pack_sentence_audio_zip

logger = logging.getLogger(__name__)


class SentenceStudioEngine:
    """Orchestrates sentence-by-sentence voice synthesis, caching, and per-line retries.

    Architecture highlights:
    - Decoupled from Gradio UI: Can be used in CLI, web services, or background tasks.
    - Global VoiceClonePrompt reuse: Reference audio is processed once, and the
      voice prompt is shared across all sentences for 100% vocal consistency.
    - Fine-grained retry: Individual sentences can be re-synthesized in-place
      without invalidating or recalculating other sentences.
    """

    def __init__(self, sampling_rate: int = 24000):
        self.sampling_rate = sampling_rate
        self.cached_prompt: Any = None
        self.ref_audio_source: Any = None
        self.ref_text_source: Optional[str] = None
        self.role_prompts: Dict[str, Any] = {}
        self.items: List[SentenceItem] = []
        self.last_zip_path: Optional[str] = None

    def get_role_prompt(self, pt_path: str) -> Any:
        """Get or load a role prompt by path with memory caching."""
        if pt_path in self.role_prompts:
            return self.role_prompts[pt_path]
        logger.info(f"Loading cached role prompt from: {pt_path}")
        from omnivoice import VoiceClonePrompt
        prompt = VoiceClonePrompt.load(pt_path)
        self.role_prompts[pt_path] = prompt
        return prompt

    def prepare_voice_prompt(
        self,
        model: Any = None,
        ref_audio: Any = None,
        ref_text: Optional[str] = None,
        pt_path: Optional[str] = None,
        source_type: str = "audio",
        force_refresh: bool = False,
    ) -> Any:
        """Extract or reuse cached VoiceClonePrompt from reference audio or .pt file.

        Args:
            model: OmniVoice model instance (required for audio extraction).
            ref_audio: Audio file path or (waveform, sr) tuple.
            ref_text: Reference audio transcript (optional).
            pt_path: Absolute or relative path to a saved .pt VoiceClonePrompt file.
            source_type: "audio" or "pt".
            force_refresh: If True, re-extract/reload even if source hasn't changed.

        Returns:
            The reusable VoiceClonePrompt object.
        """
        if source_type == "pt" or (pt_path and not ref_audio):
            if not pt_path:
                raise ValueError("No .pt role file specified.")
            if not force_refresh and self.cached_prompt is not None and self.ref_audio_source == pt_path:
                return self.cached_prompt

            logger.info(f"Loading VoiceClonePrompt from role file: {pt_path}")
            from omnivoice import VoiceClonePrompt
            prompt = VoiceClonePrompt.load(pt_path)
            self.cached_prompt = prompt
            self.ref_audio_source = pt_path
            self.ref_text_source = None
            return prompt

        if (
            not force_refresh
            and self.cached_prompt is not None
            and self.ref_audio_source == ref_audio
            and self.ref_text_source == ref_text
        ):
            return self.cached_prompt

        if model is None:
            raise ValueError("Model is required to extract VoiceClonePrompt from audio.")

        logger.info("Extracting VoiceClonePrompt for SentenceStudioEngine...")
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        self.cached_prompt = prompt
        self.ref_audio_source = ref_audio
        self.ref_text_source = ref_text
        return prompt

    def set_prompt_directly(self, prompt: Any):
        """Allow setting an already-loaded VoiceClonePrompt (e.g. from preset .pt)."""
        self.cached_prompt = prompt
        self.ref_audio_source = None
        self.ref_text_source = None

    def initialize_sentences(self, raw_text: str) -> List[SentenceItem]:
        """Split raw text and initialize SentenceItem objects in 'pending' status."""
        cleaned_sentences = clean_and_split_text(raw_text)
        self.items = [
            SentenceItem(
                index=i + 1,
                text=sentence,
                raw_text=sentence,
                sample_rate=self.sampling_rate,
                status="pending",
            )
            for i, sentence in enumerate(cleaned_sentences)
        ]
        self.last_zip_path = None
        return self.items

    def generate_all(
        self,
        model: Any,
        raw_text: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        instruct: Optional[str] = None,
        gen_config: Optional[Any] = None,
        progress_callback: Optional[Callable[[int, int, SentenceItem], None]] = None,
    ) -> List[SentenceItem]:
        """Synthesize all sentences sequentially, sharing the cached voice prompt.

        Args:
            model: OmniVoice model instance.
            raw_text: Input text (if not already initialized).
            language: Target language code or name.
            speed: Speaking speed multiplier.
            instruct: Voice/style instruction string.
            gen_config: Generation configuration options.
            progress_callback: Optional callback(current_idx, total_count, item).

        Returns:
            The completed list of SentenceItem objects.
        """
        if raw_text is not None:
            self.initialize_sentences(raw_text)

        if not self.items:
            return []

        total = len(self.items)
        for i, item in enumerate(self.items):
            item.status = "generating"
            try:
                kw: Dict[str, Any] = {
                    "text": item.text.strip(),
                    "language": language,
                    "voice_clone_prompt": self.cached_prompt,
                    "generation_config": gen_config,
                }
                if speed != 1.0:
                    kw["speed"] = float(speed)
                if instruct and instruct.strip():
                    kw["instruct"] = instruct.strip()

                audios = model.generate(**kw)
                waveform = audios[0]
                if waveform.ndim > 1:
                    waveform = waveform.squeeze()

                item.waveform = waveform
                item.duration = len(waveform) / self.sampling_rate
                item.status = "done"
                item.error_msg = None
            except Exception as e:
                logger.error(f"Error generating sentence {item.formatted_index}: {e}", exc_info=True)
                item.status = "error"
                item.error_msg = str(e)

            if progress_callback is not None:
                progress_callback(i + 1, total, item)

        self.last_zip_path = None  # Invalidate zip cache
        return self.items

    def regenerate_sentence(
        self,
        model: Any,
        index: int,
        new_text: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        instruct: Optional[str] = None,
        gen_config: Optional[Any] = None,
    ) -> SentenceItem:
        """Regenerate a single sentence in-place without touching other sentences.

        Args:
            model: OmniVoice model instance.
            index: 1-based sentence index.
            new_text: Optional replacement text for this sentence.
            language: Target language.
            speed: Speaking speed.
            instruct: Style instruction.
            gen_config: Generation configuration.

        Returns:
            The updated SentenceItem.
        """
        target_item = None
        for item in self.items:
            if item.index == index:
                target_item = item
                break

        if target_item is None:
            raise ValueError(f"Sentence with index {index} not found.")

        if new_text is not None and new_text.strip():
            target_item.text = new_text.strip()

        target_item.status = "generating"
        try:
            kw: Dict[str, Any] = {
                "text": target_item.text,
                "language": language,
                "voice_clone_prompt": self.cached_prompt,
                "generation_config": gen_config,
            }
            if speed != 1.0:
                kw["speed"] = float(speed)
            if instruct and instruct.strip():
                kw["instruct"] = instruct.strip()

            audios = model.generate(**kw)
            waveform = audios[0]
            if waveform.ndim > 1:
                waveform = waveform.squeeze()

            target_item.waveform = waveform
            target_item.duration = len(waveform) / self.sampling_rate
            target_item.status = "done"
            target_item.error_msg = None
        except Exception as e:
            logger.error(f"Error regenerating sentence {target_item.formatted_index}: {e}", exc_info=True)
            target_item.status = "error"
            target_item.error_msg = str(e)

        self.last_zip_path = None  # Invalidate zip cache
        return target_item

    def export_zip(self, output_zip_path: str) -> str:
        """Pack all completed sentences into an ordered ZIP file."""
        ready_items = [it for it in self.items if it.waveform is not None]
        if not ready_items:
            raise RuntimeError("No generated audio sentences available to export.")
        zip_file = pack_sentence_audio_zip(ready_items, output_zip_path)
        self.last_zip_path = zip_file
        return zip_file
