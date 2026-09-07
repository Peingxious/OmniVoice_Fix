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
"""Unit tests for Sentence Studio core text processor and packager."""

import json
import os
import tempfile
import zipfile
import numpy as np
import soundfile as sf

from omnivoice.sentence_studio import (
    SentenceItem,
    clean_and_split_text,
    pack_sentence_audio_zip,
)


def test_clean_and_split_text_line_first():
    """Verify that multi-line text splits strictly by line and discards blank/dirty lines."""
    raw_text = """
    
    第一句话：这是一个测试，连句不分开。
    
    
    第二句话！注意这里有感叹号！
    
       。 
    第三句话？这是问题吗？
    
    """
    sentences = clean_and_split_text(raw_text)
    assert len(sentences) == 3
    assert sentences[0] == "第一句话：这是一个测试，连句不分开。"
    assert sentences[1] == "第二句话！注意这里有感叹号！"
    assert sentences[2] == "第三句话？这是问题吗？"


def test_clean_and_split_text_single_line_fallback():
    """Verify that single-line text without newlines falls back to punctuation splitting."""
    single_line = "今天天气真好。我们一起去公园吧！你觉得呢？好的，就这么定了。"
    sentences = clean_and_split_text(single_line)
    assert len(sentences) == 4
    assert sentences[0] == "今天天气真好。"
    assert sentences[1] == "我们一起去公园吧！"
    assert sentences[2] == "你觉得呢？"
    assert sentences[3] == "好的，就这么定了。"


def test_clean_and_split_text_abbreviations_and_decimals():
    """Verify that decimals (3.14) and common abbreviations (Dr., e.g.) are not split."""
    text = "The value of pi is 3.14 approximately. Dr. Smith arrived at 10 a.m. today!"
    sentences = clean_and_split_text(text)
    assert len(sentences) == 2
    assert "3.14" in sentences[0]
    assert "Dr. Smith" in sentences[1]


def test_clean_and_split_text_empty_and_noise():
    """Verify handling of empty or noise-only text."""
    assert clean_and_split_text("") == []
    assert clean_and_split_text("   \n\n   ") == []
    assert clean_and_split_text("。。。\n   \n？？？") == []


def test_sentence_item_attributes():
    """Verify SentenceItem formatted_index and safe_filename_prefix."""
    item1 = SentenceItem(index=1, text="第一句测试文本")
    assert item1.formatted_index == "01"
    assert item1.safe_filename_prefix == "01_第一句测试文本"

    item12 = SentenceItem(index=12, text="特殊符号/\\:*?\"<>|测试")
    assert item12.formatted_index == "12"
    assert "/" not in item12.safe_filename_prefix
    assert "*" not in item12.safe_filename_prefix


def test_pack_sentence_audio_zip():
    """Verify packing generated sentences into a well-formed, ordered ZIP file."""
    sr = 24000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    wave1 = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    wave2 = 0.5 * np.sin(2 * np.pi * 880 * t).astype(np.float32)

    item1 = SentenceItem(index=1, text="第一句", waveform=wave1, sample_rate=sr, duration=0.5)
    item2 = SentenceItem(index=2, text="第二句", waveform=wave2, sample_rate=sr, duration=0.5)

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "test_output.zip")
        out = pack_sentence_audio_zip([item1, item2], zip_path)
        assert os.path.isfile(out)

        with zipfile.ZipFile(out, "r") as zf:
            namelist = zf.namelist()
            assert "01_第一句.wav" in namelist
            assert "02_第二句.wav" in namelist
            assert "manifest.json" in namelist
            assert "list.txt" in namelist

            # Check manifest content
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            assert manifest["total_sentences"] == 2
            assert manifest["sample_rate"] == 24000
            assert manifest["sentences"][0]["filename"] == "01_第一句.wav"
            assert manifest["sentences"][1]["filename"] == "02_第二句.wav"

            # Check audio extractability
            wav_bytes = zf.read("01_第一句.wav")
            wav_tmp = os.path.join(tmp_dir, "extracted_01.wav")
            with open(wav_tmp, "wb") as f:
                f.write(wav_bytes)

            data, loaded_sr = sf.read(wav_tmp)
            assert loaded_sr == 24000
            assert len(data) == len(wave1)


def test_clean_and_split_text_chinese_mixed_multiline():
    """Verify Chinese mixed multi-line text with irregular breaks and empty lines."""
    raw = """
    这是第一行：介绍 OmniVoice 系统，支持 600+ 种语言。
    
    这是第二行，连句逗号不拆开，顿号、分号；都保留在一起！
    
    
    
    这是第三行？疑问语气！
    """
    sentences = clean_and_split_text(raw)
    assert len(sentences) == 3
    assert sentences[0] == "这是第一行：介绍 OmniVoice 系统，支持 600+ 种语言。"
    assert sentences[1] == "这是第二行，连句逗号不拆开，顿号、分号；都保留在一起！"
    assert sentences[2] == "这是第三行？疑问语气！"


def test_pack_large_batch_zip():
    """Verify packing 15 sentences to check zero-padded formatting (01 .. 15)."""
    sr = 24000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    wave = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    items = [
        SentenceItem(index=i + 1, text=f"第{i+1}句话测试", waveform=wave, sample_rate=sr, duration=0.1)
        for i in range(15)
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "batch_15.zip")
        out = pack_sentence_audio_zip(items, zip_path)
        assert os.path.isfile(out)

        with zipfile.ZipFile(out, "r") as zf:
            namelist = zf.namelist()
            assert "01_第1句话测试.wav" in namelist
            assert "09_第9句话测试.wav" in namelist
            assert "10_第10句话测试.wav" in namelist
            assert "15_第15句话测试.wav" in namelist

            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            assert manifest["total_sentences"] == 15
            assert manifest["sentences"][0]["formatted_index"] == "01"
            assert manifest["sentences"][14]["formatted_index"] == "15"


def test_sentence_studio_engine():
    """Verify SentenceStudioEngine end-to-end with prompt caching and single-sentence retry."""
    from omnivoice.sentence_studio import SentenceStudioEngine

    class MockOmniVoiceModel:
        def __init__(self):
            self.create_prompt_call_count = 0
            self.generate_call_count = 0

        def create_voice_clone_prompt(self, ref_audio, ref_text=None):
            self.create_prompt_call_count += 1
            return {"fake_prompt": "voice_123"}

        def generate(self, text, language=None, voice_clone_prompt=None, speed=1.0, **kwargs):
            self.generate_call_count += 1
            # Return dummy wave proportional to text length
            sr = 24000
            duration = max(0.1, len(text) * 0.05)
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            wave = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            return [wave]

    model = MockOmniVoiceModel()
    engine = SentenceStudioEngine(sampling_rate=24000)

    # 1. Prepare prompt
    prompt = engine.prepare_voice_prompt(model, ref_audio="dummy_ref.wav", ref_text="测试")
    assert prompt == {"fake_prompt": "voice_123"}
    assert model.create_prompt_call_count == 1

    # Call again with same audio -> should hit cache
    prompt2 = engine.prepare_voice_prompt(model, ref_audio="dummy_ref.wav", ref_text="测试")
    assert model.create_prompt_call_count == 1  # No extra calls

    # 2. Generate all sentences
    raw_text = "第一句测试。\n第二句测试。\n第三句测试。"
    items = engine.generate_all(model, raw_text=raw_text)
    assert len(items) == 3
    assert model.generate_call_count == 3
    for it in items:
        assert it.status == "done"
        assert it.waveform is not None
        assert it.duration > 0

    item1_wave = items[0].waveform.copy()
    item2_wave = items[1].waveform.copy()
    item3_wave = items[2].waveform.copy()

    # 3. Single-sentence retry (regenerate sentence 2 with new text)
    updated_item2 = engine.regenerate_sentence(
        model, index=2, new_text="这是修改后的超长第二句话，重新生成！"
    )
    assert model.generate_call_count == 4  # exactly 1 more call
    assert updated_item2.index == 2
    assert updated_item2.text == "这是修改后的超长第二句话，重新生成！"
    assert updated_item2.status == "done"

    # Sentence 1 and 3 should be completely untouched
    np.testing.assert_array_equal(items[0].waveform, item1_wave)
    np.testing.assert_array_equal(items[2].waveform, item3_wave)
    # Sentence 2 wave should be changed
    assert len(items[1].waveform) != len(item2_wave)

    # 4. Test export ZIP
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_file = os.path.join(tmp_dir, "engine_export.zip")
        out_zip = engine.export_zip(zip_file)
        assert os.path.isfile(out_zip)
        with zipfile.ZipFile(out_zip, "r") as zf:
            namelist = zf.namelist()
            assert f"{items[0].safe_filename_prefix}.wav" in namelist
            assert f"{items[1].safe_filename_prefix}.wav" in namelist
            assert f"{items[2].safe_filename_prefix}.wav" in namelist
            assert "manifest.json" in namelist
            assert "list.txt" in namelist


def test_sentence_studio_engine_pt_source():
    """Verify SentenceStudioEngine loading VoiceClonePrompt directly from a .pt role file."""
    from omnivoice.sentence_studio import SentenceStudioEngine
    import torch
    from omnivoice import VoiceClonePrompt

    engine = SentenceStudioEngine(sampling_rate=24000)
    with tempfile.TemporaryDirectory() as tmp_dir:
        pt_path = os.path.join(tmp_dir, "test_role.pt")
        prompt = VoiceClonePrompt(
            ref_text="角色台词测试",
            ref_audio_tokens=torch.zeros((8, 50), dtype=torch.long),
            ref_rms=0.18,
        )
        prompt.save(pt_path)

        loaded_prompt = engine.prepare_voice_prompt(source_type="pt", pt_path=pt_path)
        assert loaded_prompt.ref_text == "角色台词测试"
        assert engine.ref_audio_source == pt_path

        # Repeated call should hit cache
        loaded_prompt2 = engine.prepare_voice_prompt(source_type="pt", pt_path=pt_path)
        assert loaded_prompt2 is loaded_prompt


if __name__ == "__main__":
    print("Running test_clean_and_split_text_line_first...")
    test_clean_and_split_text_line_first()
    print("Running test_clean_and_split_text_single_line_fallback...")
    test_clean_and_split_text_single_line_fallback()
    print("Running test_clean_and_split_text_abbreviations_and_decimals...")
    test_clean_and_split_text_abbreviations_and_decimals()
    print("Running test_clean_and_split_text_empty_and_noise...")
    test_clean_and_split_text_empty_and_noise()
    print("Running test_clean_and_split_text_chinese_mixed_multiline...")
    test_clean_and_split_text_chinese_mixed_multiline()
    print("Running test_sentence_item_attributes...")
    test_sentence_item_attributes()
    print("Running test_pack_sentence_audio_zip...")
    test_pack_sentence_audio_zip()
    print("Running test_pack_large_batch_zip...")
    test_pack_large_batch_zip()
    print("Running test_sentence_studio_engine...")
    test_sentence_studio_engine()
    print("Running test_sentence_studio_engine_pt_source...")
    test_sentence_studio_engine_pt_source()
    print("ALL 10 TESTS PASSED SUCCESSFULLY!")


