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
"""Text normalization and sentence splitting engine for Sentence Studio."""

import re
from typing import List

# Sentence ending punctuation marks
_SPLIT_PUNCTUATION = set("。！？.!?\n")
_CLOSING_MARKS = set("”’\"')）]}》〉")
_COMMAS_AND_PAUSES = set("，、；,;")

# Common abbreviations where periods should not trigger a split
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e", "no", "vol",
    "am", "pm", "a.m", "p.m",
}


def _is_isolated_punctuation(s: str) -> bool:
    """Check if a string contains only punctuation or whitespace."""
    stripped = s.strip()
    if not stripped:
        return True
    return bool(re.fullmatch(r"[\s\W_]+", stripped)) and not re.search(r"[\w\u4e00-\u9fa5]", stripped)


def _split_by_punctuation(text: str) -> List[str]:
    """Split a single block of text by sentence-ending punctuation marks."""
    sentences: List[str] = []
    current: List[str] = []
    tokens = list(text)
    n = len(tokens)

    i = 0
    while i < n:
        char = tokens[i]
        current.append(char)

        if char in "。！？!?":
            # Consume any immediately following punctuation or closing quotes
            while i + 1 < n and (tokens[i + 1] in "。！？!?\"'”’）)》〉"):
                i += 1
                current.append(tokens[i])
            sent = "".join(current).strip()
            if sent and not _is_isolated_punctuation(sent):
                sentences.append(sent)
            current = []
        elif char == ".":
            # Check if period is immediately followed by a non-space (e.g. a.m., domain.com, etc.)
            has_next_non_space = (
                i + 1 < n
                and not tokens[i + 1].isspace()
                and tokens[i + 1] not in "\"'”’）)》〉\n"
            )

            # Check if this is a decimal number (e.g. 3.14)
            is_decimal = False
            if i > 0 and tokens[i - 1].isdigit() and i + 1 < n and tokens[i + 1].isdigit():
                is_decimal = True

            # Check if this is a known abbreviation or single-letter initial
            is_abbrev = False
            if not is_decimal:
                prev_text = "".join(current[:-1]).strip()
                if prev_text:
                    last_word = prev_text.split()[-1].lower().rstrip(".")
                    # Single-letter initial (e.g. J. K. Rowling, a.m.)
                    if len(last_word) == 1 and last_word.isalpha():
                        is_abbrev = True
                    elif last_word in _ABBREVIATIONS or last_word.replace(".", "") in _ABBREVIATIONS:
                        is_abbrev = True

            if not is_decimal and not is_abbrev and not has_next_non_space:
                # Consume any trailing quotes
                while i + 1 < n and (tokens[i + 1] in "\"'”’）)》〉"):
                    i += 1
                    current.append(tokens[i])
                sent = "".join(current).strip()
                if sent and not _is_isolated_punctuation(sent):
                    sentences.append(sent)
                current = []
        i += 1

    if current:
        sent = "".join(current).strip()
        if sent and not _is_isolated_punctuation(sent):
            sentences.append(sent)

    return sentences


def clean_and_split_text(text: str, split_long_lines: bool = False, max_chars_per_line: int = 120) -> List[str]:
    """Clean and split raw input text into an ordered list of sentences.

    Processing contract:
    1. Line-first precedence: If the text has newlines, treat each non-empty
       line as one sentence, preserving run-on clauses (commas, semicolons).
    2. Empty line & noise stripping: Filter out blank lines and lines containing
       only isolated punctuation marks.
    3. Fallback for single-line text: If the input has no newlines, split by
       sentence-ending punctuation marks (。！？.!?).
    4. Optional long line splitting: If enabled and a line exceeds max_chars_per_line,
       it will be intelligently split by punctuation.

    Args:
        text: Raw text string (may contain messy linebreaks, blank lines, etc.).
        split_long_lines: Whether to further split excessively long lines.
        max_chars_per_line: Character threshold to trigger splitting long lines.

    Returns:
        A list of clean, non-empty, sequential sentence strings.
    """
    if not text or not text.strip():
        return []

    # Normalize line endings
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    raw_lines = normalized.split("\n")
    cleaned_lines = [line.strip() for line in raw_lines]
    # Filter out empty and punctuation-only lines
    valid_lines = [line for line in cleaned_lines if line and not _is_isolated_punctuation(line)]

    if not valid_lines:
        return []

    # If user provided multiple distinct lines (line-first)
    if len(valid_lines) > 1 or "\n" in normalized:
        result: List[str] = []
        for line in valid_lines:
            if split_long_lines and len(line) > max_chars_per_line:
                sub_sentences = _split_by_punctuation(line)
                result.extend(sub_sentences if sub_sentences else [line])
            else:
                result.append(line)
        return result

    # Fallback: Exactly 1 line without newlines -> split by punctuation
    single_line = valid_lines[0]
    sub_sentences = _split_by_punctuation(single_line)
    return sub_sentences if sub_sentences else [single_line]
