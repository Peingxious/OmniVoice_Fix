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
"""Decoupled FastAPI backend for OmniVoice Sentence Studio."""

import asyncio
import json
import logging
import os
import re
import shutil
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig, VoiceClonePrompt
from omnivoice.sentence_studio.engine import SentenceStudioEngine
from omnivoice.sentence_studio.models import SentenceItem
from omnivoice.sentence_studio.packager import pack_sentence_audio_zip
from omnivoice.sentence_studio.text_processor import clean_and_split_text
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

logger = logging.getLogger("omnivoice.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
VOICE_DIR = os.path.join(_PROJECT_ROOT, "voices")
CACHE_TMP_DIR = os.path.join(_PROJECT_ROOT, ".cache", "tmp")
WEB_DIR = os.path.join(_PROJECT_ROOT, "web")

os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(CACHE_TMP_DIR, exist_ok=True)
os.makedirs(WEB_DIR, exist_ok=True)

# Global Model & Engine instances
_GLOBAL_MODEL: Optional[OmniVoice] = None
_GLOBAL_ENGINE: Optional[SentenceStudioEngine] = None
_TASK_STORE: Dict[str, Dict[str, Any]] = {}


def get_model() -> OmniVoice:
    """Retrieve or lazily initialize the OmniVoice inference model."""
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        checkpoint = os.environ.get("OMNIVOICE_CHECKPOINT", "k2-fsa/OmniVoice")
        device = os.environ.get("OMNIVOICE_DEVICE") or get_best_device()
        logger.info(f"Loading OmniVoice model from {checkpoint} on {device} ...")
        _GLOBAL_MODEL = OmniVoice.from_pretrained(
            checkpoint,
            device_map=device,
            dtype=torch.float16,
            load_asr=False,
            asr_model_name=os.environ.get(
                "OMNIVOICE_ASR_MODEL", "openai/whisper-large-v3-turbo"
            ),
        )
        logger.info("OmniVoice model loaded successfully.")
    return _GLOBAL_MODEL


def get_engine() -> SentenceStudioEngine:
    """Retrieve or initialize SentenceStudioEngine."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        model = get_model()
        _GLOBAL_ENGINE = SentenceStudioEngine(sampling_rate=model.sampling_rate)
    return _GLOBAL_ENGINE


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------


class SplitRequest(BaseModel):
    text: str = Field(..., description="Raw script text to clean and split.")


class SplitResponse(BaseModel):
    sentences: List[str]
    count: int


class SentenceInput(BaseModel):
    index: int
    text: str
    role: Optional[str] = None


class BatchSynthesizeRequest(BaseModel):
    task_id: Optional[str] = None
    sentences: List[SentenceInput]
    global_role: Optional[str] = None
    speed: float = 1.0
    instruct: Optional[str] = None
    lang: Optional[str] = None
    force: bool = False


class SingleSynthesizeRequest(BaseModel):
    task_id: str
    index: int
    text: str
    role: Optional[str] = None
    speed: float = 1.0
    instruct: Optional[str] = None
    lang: Optional[str] = None


class DesignGenerateRequest(BaseModel):
    text: str
    attributes: List[str] = Field(default_factory=list)
    lang: Optional[str] = None
    speed: float = 1.0
    class_temperature: float = 0.35


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _resolve_pt_path(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    if os.path.isabs(filename) and os.path.isfile(filename):
        return filename
    candidate = os.path.join(VOICE_DIR, filename)
    if os.path.isfile(candidate):
        return candidate
    return None


def _clean_role_name(raw: str) -> str:
    name = os.path.basename(raw or "")
    name = re.sub(r'[\\/:*?"<>|]', "", name).strip().rstrip(".")
    if not name:
        name = "voice_" + time.strftime("%Y%m%d_%H%M%S")
    if not name.endswith(".pt"):
        name += ".pt"
    return name


def _unique_voice_path(name: str) -> str:
    path = os.path.join(VOICE_DIR, name)
    if not os.path.exists(path):
        return path
    stem = name[:-3]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(VOICE_DIR, f"{stem}_{stamp}.pt")


def _save_generated_wave(waveform, sampling_rate: int) -> Dict[str, Any]:
    """Persist a generated waveform under .cache/tmp/<task>/ and return its URL info."""
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu().numpy()
    if waveform.ndim > 1:
        waveform = waveform.squeeze()
    task_id = uuid.uuid4().hex[:8]
    task_dir = os.path.join(CACHE_TMP_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    filename = "01.wav"
    int16_wave = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
    sf.write(os.path.join(task_dir, filename), int16_wave, sampling_rate)
    duration = round(len(waveform) / sampling_rate, 2)
    return {
        "task_id": task_id,
        "audio_url": f"/api/audio/{task_id}/{filename}",
        "duration": duration,
    }


def _task_state_path(task_id: str) -> str:
    return os.path.join(CACHE_TMP_DIR, task_id, "task_state.json")


def _load_task_state(task_id: str) -> Optional[Dict[str, Any]]:
    try:
        with open(_task_state_path(task_id), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _save_task_state(task_id: str, entries: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    """Persist per-sentence results so tasks can resume after server restarts."""
    path = _task_state_path(task_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "items": entries}, f, ensure_ascii=False)
    os.replace(tmp, path)


def _item_from_disk(task_id: str, entry: Dict[str, Any]) -> Optional[SentenceItem]:
    filename = os.path.basename(entry.get("filename") or "")
    if not filename:
        return None
    wav_path = os.path.join(CACHE_TMP_DIR, task_id, filename)
    if not os.path.isfile(wav_path):
        return None
    try:
        wav, sr = sf.read(wav_path, dtype="float32")
    except Exception:
        return None
    if wav.ndim > 1:
        wav = wav.squeeze()
    duration = float(entry.get("duration") or 0) or len(wav) / max(int(sr), 1)
    return SentenceItem(
        index=int(entry.get("index", 0)),
        text=entry.get("text", ""),
        raw_text=entry.get("text", ""),
        sample_rate=int(sr),
        waveform=wav,
        duration=duration,
        status="done",
        role_name=entry.get("role_name"),
    )


def _items_from_task_state(task_id: str) -> List[SentenceItem]:
    state = _load_task_state(task_id)
    if not state:
        return []
    items = [it for it in (_item_from_disk(task_id, e) for e in state.get("items", [])) if it]
    items.sort(key=lambda x: x.index)
    return items


def _request_meta(req: "BatchSynthesizeRequest") -> Dict[str, Any]:
    return {
        "global_role": req.global_role,
        "speed": req.speed,
        "instruct": req.instruct,
        "lang": req.lang,
    }


def _meta_matches(saved: Any, meta: Dict[str, Any]) -> bool:
    if not isinstance(saved, dict):
        return False
    try:
        speed_ok = float(saved.get("speed")) == float(meta["speed"])
    except (TypeError, ValueError):
        speed_ok = False
    return (
        speed_ok
        and saved.get("global_role") == meta["global_role"]
        and saved.get("instruct") == meta["instruct"]
        and saved.get("lang") == meta["lang"]
    )


def _upsert_entry(entries: List[Dict[str, Any]], entry: Dict[str, Any]) -> None:
    entries[:] = [e for e in entries if e.get("index") != entry["index"]]
    entries.append(entry)
    entries.sort(key=lambda e: e.get("index", 0))


def _find_cached_item(
    task_id: str,
    index: int,
    text: str,
    assigned_role: Optional[str],
    meta: Dict[str, Any],
) -> Optional[SentenceItem]:
    """Reuse a previous done result (same index/text/role/settings) for resume."""
    store = _TASK_STORE.get(task_id)
    if store and _meta_matches(store, meta):
        for it in store.get("items", []):
            if (
                it.index == index
                and it.raw_text == text
                and it.status == "done"
                and it.waveform is not None
                and it.role_name == assigned_role
            ):
                return it

    state = _load_task_state(task_id)
    if state and _meta_matches(state.get("meta"), meta):
        for entry in state.get("items", []):
            if (
                entry.get("index") == index
                and entry.get("text") == text
                and entry.get("role_name") == assigned_role
            ):
                item = _item_from_disk(task_id, entry)
                if item is None:
                    return None
                store = _TASK_STORE.setdefault(task_id, {"items": []})
                merged = [e for e in store.get("items", []) if e.index != index]
                merged.append(item)
                merged.sort(key=lambda x: x.index)
                store["items"] = merged
                return item
    return None


def _generate_once(
    model: OmniVoice,
    text: str,
    lang: Optional[str],
    speed: float,
    class_temperature: float,
    instruct: Optional[str] = None,
    voice_clone_prompt: Any = None,
) -> Any:
    """Single-shot generation mirroring demo.py defaults (32 steps, gs 2.0)."""
    raw_text = text.strip()
    chunk_thresh = 15.0 if (len(raw_text) > 120 or raw_text.count("\n") >= 2) else 25.0
    gen_config = OmniVoiceGenerationConfig(
        num_step=32,
        guidance_scale=2.0,
        denoise=True,
        preprocess_prompt=True,
        postprocess_output=True,
        class_temperature=float(class_temperature),
        audio_chunk_duration=12.0,
        audio_chunk_threshold=chunk_thresh,
    )
    kw: Dict[str, Any] = dict(
        text=raw_text,
        language=None if lang in (None, "Auto", "auto") else lang,
        generation_config=gen_config,
    )
    if voice_clone_prompt is not None:
        kw["voice_clone_prompt"] = voice_clone_prompt
    if speed and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if instruct and instruct.strip():
        kw["instruct"] = instruct.strip()
    auds = model.generate(**kw)
    return auds[0]


def _get_voice_info(filename: str) -> Dict[str, Any]:
    path = _resolve_pt_path(filename)
    if not path or not os.path.isfile(path):
        return {"filename": filename, "exists": False}
    try:
        size_kb = round(os.path.getsize(path) / 1024.0, 1)
        prompt = VoiceClonePrompt.load(path)
        num_tokens = prompt.ref_audio_tokens.size(1)
        est_sec = round(num_tokens / 25.0, 1)
        return {
            "filename": filename,
            "exists": True,
            "size_kb": size_kb,
            "est_seconds": est_sec,
            "ref_rms": round(float(prompt.ref_rms), 4),
            "ref_text": (prompt.ref_text or "").strip()[:140],
        }
    except Exception as e:
        logger.error(f"Error loading voice info {filename}: {e}")
        return {"filename": filename, "exists": True, "error": str(e)}


# ---------------------------------------------------------------------------
# FastAPI App Factory
# ---------------------------------------------------------------------------


def create_app(model: Optional[OmniVoice] = None) -> FastAPI:
    global _GLOBAL_MODEL
    if model is not None:
        _GLOBAL_MODEL = model

    app = FastAPI(
        title="OmniVoice Sentence Studio Web API",
        description="Clean, decoupled REST and SSE API for line-by-line speech synthesis.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 1. Voices API
    @app.get("/api/voices")
    def list_voices():
        """List all saved .pt voices from voices/ folder."""
        files = [f for f in os.listdir(VOICE_DIR) if f.endswith(".pt")]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(VOICE_DIR, x)), reverse=True)
        infos = [_get_voice_info(f) for f in files]
        return {"voices": infos, "default": files[0] if files else None}

    # 2. Split Script API
    @app.post("/api/split", response_model=SplitResponse)
    def split_script(req: SplitRequest):
        """Clean and split text strictly line-first, preserving commas."""
        sentences = clean_and_split_text(req.text)
        return SplitResponse(sentences=sentences, count=len(sentences))

    # 3. Audio File Serving API
    @app.get("/api/audio/{task_id}/{filename}")
    def get_audio_file(task_id: str, filename: str):
        """Serve individual generated WAV audio file."""
        # Task ids are server-generated hex; filenames may legitimately contain
        # CJK characters (role names / sentence text), so only block traversal.
        safe_task = re.sub(r"[^a-zA-Z0-9_\-]", "", task_id)
        safe_filename = os.path.basename(filename.replace("\\", "/"))
        if (
            not safe_task
            or safe_task != task_id
            or not safe_filename
            or safe_filename in (".", "..")
        ):
            raise HTTPException(status_code=400, detail="Invalid audio path")
        file_path = os.path.join(CACHE_TMP_DIR, safe_task, safe_filename)
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        return FileResponse(file_path, media_type="audio/wav")

    # 4. ZIP Direct Download API
    @app.get("/api/download_zip/{task_id}")
    def download_zip(task_id: str):
        """Native direct ZIP download stream for the task."""
        safe_task = re.sub(r"[^a-zA-Z0-9_\-]", "", task_id)
        task_info = _TASK_STORE.get(safe_task)
        zip_path = task_info.get("zip_path") if task_info else None

        if not zip_path or not os.path.isfile(zip_path):
            # Try to rebuild: zip on disk (survives restarts), memory store, or task_state.json + wavs
            task_dir = os.path.join(CACHE_TMP_DIR, safe_task)
            default_zip = os.path.join(CACHE_TMP_DIR, f"sentence_studio_{safe_task}.zip")
            if os.path.isfile(default_zip):
                zip_path = default_zip
            elif os.path.isdir(task_dir):
                if task_info and task_info.get("items"):
                    rebuild_items = task_info["items"]
                else:
                    rebuild_items = _items_from_task_state(safe_task)
                if rebuild_items:
                    zip_path = default_zip
                    pack_sentence_audio_zip(rebuild_items, zip_path, include_manifest=False)
            if not zip_path or not os.path.isfile(zip_path):
                raise HTTPException(status_code=404, detail="ZIP archive not found or expired.")

        return FileResponse(
            zip_path,
            filename=f"sentence_studio_{safe_task}.zip",
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="sentence_studio_{safe_task}.zip"'},
        )

    # 5. Batch Synthesize via Server-Sent Events (SSE)
    @app.post("/api/synthesize_batch")
    async def synthesize_batch(req: BatchSynthesizeRequest):
        """Synthesize sentences line by line, streaming realtime SSE updates."""
        if not req.sentences:
            raise HTTPException(status_code=400, detail="Sentence list cannot be empty.")

        task_id = req.task_id or uuid.uuid4().hex[:8]
        task_dir = os.path.join(CACHE_TMP_DIR, task_id)
        os.makedirs(task_dir, exist_ok=True)

        async def sse_generator():
            # Model/engine load lazily so pure cache-hit resumes stay instant
            model_holder: Dict[str, Any] = {}
            prompt_cache: Dict[str, Any] = {}

            def _ensure_model():
                if "model" not in model_holder:
                    engine = get_engine()
                    m = get_model()
                    model_holder.update(model=m, engine=engine, sampling_rate=m.sampling_rate)
                return model_holder["model"], model_holder["engine"], model_holder["sampling_rate"]

            # Resolve global role path (no model load needed for cache hits)
            global_pt = _resolve_pt_path(req.global_role)
            if not global_pt and req.global_role:
                global_pt = _resolve_pt_path(req.global_role + ".pt")
            if not global_pt:
                # Default to first available voice
                available = [f for f in os.listdir(VOICE_DIR) if f.endswith(".pt")]
                if available:
                    global_pt = os.path.join(VOICE_DIR, available[0])

            if not global_pt or not os.path.isfile(global_pt):
                yield f"event: error\ndata: {json.dumps({'error': 'No valid role .pt file found.'})}\n\n"
                return

            lang_param = None if req.lang in (None, "Auto", "auto") else req.lang

            items: List[SentenceItem] = []
            total = len(req.sentences)

            meta = _request_meta(req)
            prev_state = _load_task_state(task_id)
            if req.force or not _meta_matches(prev_state.get("meta") if prev_state else None, meta):
                state_entries: List[Dict[str, Any]] = []
            else:
                state_entries = list(prev_state.get("items", []))

            yield f"event: start\ndata: {json.dumps({'task_id': task_id, 'total': total})}\n\n"

            for s_in in req.sentences:
                idx = s_in.index
                text = s_in.text.strip()
                if not text:
                    continue

                # Check custom role vs global (path only; prompt resolved lazily)
                custom_pt = None
                role_name = s_in.role
                if role_name and role_name not in ("跟随全局默认角色", "Follow Global Role"):
                    candidate = _resolve_pt_path(role_name) or _resolve_pt_path(role_name + ".pt")
                    if candidate and os.path.isfile(candidate):
                        custom_pt = candidate
                prompt_path = custom_pt or global_pt
                assigned_role = os.path.basename(prompt_path)

                # Resume: reuse existing audio for unchanged sentences unless forced
                if not req.force:
                    cached = _find_cached_item(task_id, idx, text, assigned_role, meta)
                    if cached is not None:
                        if not any(it.index == idx for it in items):
                            items.append(cached)
                        cached_filename = f"{cached.safe_filename_prefix}.wav"
                        yield f"event: sentence\ndata: {json.dumps({'index': idx, 'status': 'done', 'duration': round(cached.duration, 2), 'audio_url': f'/api/audio/{task_id}/{cached_filename}'})}\n\n"
                        continue

                # Yield generating status
                yield f"event: progress\ndata: {json.dumps({'index': idx, 'status': 'generating'})}\n\n"

                model, engine, sampling_rate = _ensure_model()
                if prompt_path not in prompt_cache:
                    prompt_cache[prompt_path] = engine.get_role_prompt(prompt_path)
                prompt_to_use = prompt_cache[prompt_path]

                # Synthesize on thread
                def _do_gen(p, t, l, sp, ins):
                    kw: Dict[str, Any] = {
                        "text": t,
                        "language": l,
                        "voice_clone_prompt": p,
                    }
                    if sp != 1.0:
                        kw["speed"] = float(sp)
                    if ins and ins.strip():
                        kw["instruct"] = ins.strip()
                    auds = model.generate(**kw)
                    w = auds[0]
                    if w.ndim > 1:
                        w = w.squeeze()
                    return w

                try:
                    waveform = await asyncio.to_thread(
                        _do_gen, prompt_to_use, text, lang_param, req.speed, req.instruct
                    )
                    dur = len(waveform) / sampling_rate
                    int16_wave = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)

                    # Save WAV
                    item = SentenceItem(
                        index=idx,
                        text=text,
                        raw_text=text,
                        sample_rate=sampling_rate,
                        waveform=waveform,
                        duration=dur,
                        status="done",
                        role_name=assigned_role,
                    )
                    filename = f"{item.safe_filename_prefix}.wav"
                    filepath = os.path.join(task_dir, filename)
                    sf.write(filepath, int16_wave, sampling_rate)

                    items.append(item)
                    _upsert_entry(state_entries, {
                        "index": idx,
                        "text": text,
                        "filename": filename,
                        "duration": round(dur, 2),
                        "role_name": assigned_role,
                        "sample_rate": sampling_rate,
                    })
                    _save_task_state(task_id, state_entries, meta)
                    audio_url = f"/api/audio/{task_id}/{filename}"

                    yield f"event: sentence\ndata: {json.dumps({'index': idx, 'status': 'done', 'duration': round(dur, 2), 'audio_url': audio_url})}\n\n"
                except Exception as e:
                    logger.error(f"Error synthesizing sentence #{idx}: {e}", exc_info=True)
                    yield f"event: sentence\ndata: {json.dumps({'index': idx, 'status': 'error', 'error': str(e)})}\n\n"

            # Package ZIP
            items.sort(key=lambda x: x.index)
            zip_filename = os.path.join(CACHE_TMP_DIR, f"sentence_studio_{task_id}.zip")
            pack_sentence_audio_zip(items, zip_filename, include_manifest=False)

            _TASK_STORE[task_id] = {
                "items": items,
                "zip_path": zip_filename,
                "global_role": req.global_role,
                "speed": req.speed,
                "instruct": req.instruct,
                "lang": req.lang,
            }

            total_dur = round(sum(it.duration for it in items if it.waveform is not None), 2)
            download_url = f"/api/download_zip/{task_id}"

            yield f"event: done\ndata: {json.dumps({'task_id': task_id, 'total': len(items), 'total_duration': total_dur, 'download_url': download_url})}\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    # 6. Single Sentence Retry API
    @app.post("/api/synthesize_single")
    async def synthesize_single(req: SingleSynthesizeRequest):
        """Regenerate a single sentence in-place, updating audio and ZIP."""
        task_id = req.task_id
        task_dir = os.path.join(CACHE_TMP_DIR, task_id)
        if not os.path.isdir(task_dir):
            os.makedirs(task_dir, exist_ok=True)

        engine = get_engine()
        model = get_model()
        sampling_rate = model.sampling_rate

        # Resolve role
        role_name = req.role
        pt_path = None
        if role_name and role_name not in ("跟随全局默认角色", "Follow Global Role"):
            pt_path = _resolve_pt_path(role_name) or _resolve_pt_path(role_name + ".pt")
        if not pt_path:
            # Check task global role
            t_info = _TASK_STORE.get(task_id, {})
            pt_path = _resolve_pt_path(t_info.get("global_role"))

        if not pt_path or not os.path.isfile(pt_path):
            available = [f for f in os.listdir(VOICE_DIR) if f.endswith(".pt")]
            if available:
                pt_path = os.path.join(VOICE_DIR, available[0])

        if not pt_path or not os.path.isfile(pt_path):
            raise HTTPException(status_code=400, detail="No role .pt available for retry.")

        prompt = engine.get_role_prompt(pt_path)
        lang_param = None if req.lang in (None, "Auto", "auto") else req.lang

        def _do_gen():
            kw: Dict[str, Any] = {
                "text": req.text.strip(),
                "language": lang_param,
                "voice_clone_prompt": prompt,
            }
            if req.speed != 1.0:
                kw["speed"] = float(req.speed)
            if req.instruct and req.instruct.strip():
                kw["instruct"] = req.instruct.strip()
            auds = model.generate(**kw)
            w = auds[0]
            if w.ndim > 1:
                w = w.squeeze()
            return w

        try:
            waveform = await asyncio.to_thread(_do_gen)
            dur = len(waveform) / sampling_rate
            int16_wave = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)

            item = SentenceItem(
                index=req.index,
                text=req.text.strip(),
                raw_text=req.text.strip(),
                sample_rate=sampling_rate,
                waveform=waveform,
                duration=dur,
                status="done",
                role_name=os.path.basename(pt_path),
            )
            filename = f"{item.safe_filename_prefix}.wav"
            filepath = os.path.join(task_dir, filename)
            sf.write(filepath, int16_wave, sampling_rate)

            # Update store
            task_info = _TASK_STORE.setdefault(task_id, {"items": []})
            items = task_info.get("items", [])
            found = False
            for i, it in enumerate(items):
                if it.index == req.index:
                    items[i] = item
                    found = True
                    break
            if not found:
                items.append(item)
                items.sort(key=lambda x: x.index)
            task_info["items"] = items

            # Re-pack ZIP
            zip_filename = os.path.join(CACHE_TMP_DIR, f"sentence_studio_{task_id}.zip")
            pack_sentence_audio_zip(items, zip_filename, include_manifest=False)
            task_info["zip_path"] = zip_filename

            # Persist to task_state.json so the result survives restarts
            state = _load_task_state(task_id) or {"meta": {}, "items": []}
            _upsert_entry(state.setdefault("items", []), {
                "index": req.index,
                "text": req.text.strip(),
                "filename": filename,
                "duration": round(dur, 2),
                "role_name": os.path.basename(pt_path),
                "sample_rate": sampling_rate,
            })
            if not state.get("meta"):
                state["meta"] = {
                    "global_role": task_info.get("global_role"),
                    "speed": task_info.get("speed", req.speed),
                    "instruct": task_info.get("instruct"),
                    "lang": task_info.get("lang", req.lang),
                }
            _save_task_state(task_id, state["items"], state["meta"])

            return {
                "index": req.index,
                "status": "done",
                "duration": round(dur, 2),
                "audio_url": f"/api/audio/{task_id}/{filename}?t={int(time.time())}",
                "download_url": f"/api/download_zip/{task_id}",
            }
        except Exception as e:
            logger.error(f"Failed to regenerate sentence #{req.index}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # 7. Upload an existing .pt voice file
    @app.post("/api/upload_voice")
    async def upload_voice(file: UploadFile = File(...)):
        """Save an uploaded .pt voice prompt into voices/."""
        if not file.filename or not file.filename.lower().endswith(".pt"):
            raise HTTPException(status_code=400, detail="仅支持上传 .pt 角色文件。")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="上传文件为空。")
        dest = _unique_voice_path(_clean_role_name(file.filename))
        with open(dest, "wb") as f:
            f.write(content)
        try:
            VoiceClonePrompt.load(dest)
        except Exception as e:
            os.remove(dest)
            raise HTTPException(status_code=400, detail=f"无效的 .pt 角色文件: {e}")
        logger.info(f"Voice uploaded: {os.path.basename(dest)}")
        return _get_voice_info(os.path.basename(dest))

    # 8. Create & save a new voice from uploaded reference audio
    @app.post("/api/create_voice")
    async def create_voice(
        file: UploadFile = File(...),
        name: Optional[str] = Form(None),
        ref_text: Optional[str] = Form(None),
    ):
        """Clone a reusable .pt voice prompt from a reference audio upload."""
        suffix = os.path.splitext(file.filename or "ref.wav")[1].lower()
        if suffix not in (".wav", ".mp3", ".flac", ".ogg", ".opus"):
            raise HTTPException(
                status_code=400,
                detail=f"不支持的音频格式 {suffix}，请使用 wav / mp3 / flac / ogg。",
            )
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="上传音频为空。")
        tmp_path = os.path.join(
            CACHE_TMP_DIR, f"ref_upload_{uuid.uuid4().hex[:8]}{suffix}"
        )
        with open(tmp_path, "wb") as f:
            f.write(content)
        try:
            model = get_model()
            # ASR auto-transcription (when ref_text is empty) may load Whisper on
            # first use, which is slow — run off the event loop.
            prompt = await asyncio.to_thread(
                model.create_voice_clone_prompt,
                tmp_path,
                (ref_text or "").strip() or None,
            )
            dest = _unique_voice_path(
                _clean_role_name(name or os.path.splitext(file.filename or "")[0])
            )
            prompt.save(dest)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Voice creation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"音色创建失败: {e}")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        logger.info(f"Voice created: {os.path.basename(dest)}")
        return _get_voice_info(os.path.basename(dest))

    # 9. Voice design attribute catalog (kept in sync with the Gradio demo)
    @app.get("/api/design_categories")
    def design_categories():
        """Speaker attribute groups for the voice design tab."""
        from omnivoice.cli.i18n import CATEGORIES

        cats = []
        for cat in CATEGORIES:
            cats.append(
                {
                    "key": cat["key"],
                    "label": cat["label"]["zh"],
                    "info": (cat.get("info") or {}).get("zh"),
                    "options": [
                        {"label": o["zh"], "value": o["value"]} for o in cat["options"]
                    ],
                }
            )
        return {"categories": cats}

    # 10. Supported target languages (600+, mirrors the Gradio demo list)
    @app.get("/api/languages")
    def list_languages():
        langs = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)
        return {"languages": langs}

    # 11. Whisper ASR transcription of a reference audio upload
    @app.post("/api/asr_transcribe")
    async def asr_transcribe(file: UploadFile = File(...)):
        suffix = os.path.splitext(file.filename or "ref.wav")[1].lower()
        if suffix not in (".wav", ".mp3", ".flac", ".ogg", ".opus"):
            raise HTTPException(
                status_code=400,
                detail=f"不支持的音频格式 {suffix}，请使用 wav / mp3 / flac / ogg。",
            )
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="上传音频为空。")
        tmp_path = os.path.join(
            CACHE_TMP_DIR, f"asr_{uuid.uuid4().hex[:8]}{suffix}"
        )
        with open(tmp_path, "wb") as f:
            f.write(content)
        try:
            model = get_model()
            # transcribe() raises if the Whisper pipe was never loaded
            # (from_pretrained runs with load_asr=False) — lazy-load it here,
            # mirroring create_voice_clone_prompt's on-the-fly behaviour.
            if model._asr_pipe is None:
                await asyncio.to_thread(model.load_asr_model)
            text = await asyncio.to_thread(model.transcribe, tmp_path)
            return {"text": (text or "").strip()}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"ASR transcribe failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"转写失败: {e}")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # 12. Voice clone: generate from uploaded reference audio
    @app.post("/api/clone_generate")
    async def clone_generate(
        file: UploadFile = File(...),
        text: str = Form(...),
        ref_text: Optional[str] = Form(None),
        lang: Optional[str] = Form(None),
        speed: float = Form(1.0),
        instruct: Optional[str] = Form(None),
        class_temperature: float = Form(0.35),
    ):
        """Clone-synthesize target text with a temporary reference audio prompt."""
        suffix = os.path.splitext(file.filename or "ref.wav")[1].lower()
        if suffix not in (".wav", ".mp3", ".flac", ".ogg", ".opus"):
            raise HTTPException(
                status_code=400,
                detail=f"不支持的音频格式 {suffix}，请使用 wav / mp3 / flac / ogg。",
            )
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="上传音频为空。")
        tmp_path = os.path.join(
            CACHE_TMP_DIR, f"ref_upload_{uuid.uuid4().hex[:8]}{suffix}"
        )
        with open(tmp_path, "wb") as f:
            f.write(content)
        try:
            model = get_model()
            prompt = await asyncio.to_thread(
                model.create_voice_clone_prompt,
                tmp_path,
                (ref_text or "").strip() or None,
            )
            waveform = await asyncio.to_thread(
                _generate_once,
                model,
                text,
                lang,
                speed,
                class_temperature,
                instruct,
                prompt,
            )
            return _save_generated_wave(waveform, model.sampling_rate)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Clone generation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"克隆合成失败: {e}")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # 13. Voice design: generate from attribute tags (no reference audio)
    @app.post("/api/design_generate")
    async def design_generate(req: DesignGenerateRequest):
        """Design-synthesize text from speaker attribute tags."""
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="合成文本不能为空。")
        instruct = ", ".join(a.strip() for a in req.attributes if a.strip()) or None
        try:
            model = get_model()
            waveform = await asyncio.to_thread(
                _generate_once,
                model,
                req.text,
                req.lang,
                req.speed,
                req.class_temperature,
                instruct,
            )
            return _save_generated_wave(waveform, model.sampling_rate)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Design generation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"声音设计合成失败: {e}")

    # 14. Mount Static Web Frontend
    if os.path.isdir(WEB_DIR):
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

    return app


app = create_app()
