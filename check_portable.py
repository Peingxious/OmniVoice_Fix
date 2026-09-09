#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OmniVoice 可移植性自检脚本（仅用 Python 标准库，无需 torch/venv 即可运行）
用法：直接双击 check_portable.bat，或 python check_portable.py
功能：
  1. 检查 .venv 虚拟环境是否就绪
  2. 检查 HuggingFace 模型缓存（主模型 + Whisper ASR）是否完整
  3. 若 torch 已安装，检测 CUDA 是否可用
  4. 缺什么就打印对应的下载地址/命令（含国内 HF 镜像）
"""
import os
import sys

# 脚本所在目录（即仓库根目录，保证从 U 盘/任意盘符都能用）
ROOT = os.path.dirname(os.path.abspath(__file__))
HF_HOME = os.path.join(ROOT, ".cache", "huggingface")
HUB = os.path.join(HF_HOME, "hub")

OK = "  [OK]"
MISS = "  [缺]"
WARN = "  [!]"

lines = []


def section(title):
    lines.append("")
    lines.append("=" * 60)
    lines.append(title)
    lines.append("=" * 60)


def check_file(path):
    return os.path.isfile(path)


def check_dir(path):
    return os.path.isdir(path)


# ---------------------------------------------------------------------------
section("1. Python 虚拟环境 (.venv)")
# ---------------------------------------------------------------------------
venv_py = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
venv_ok = check_file(venv_py)
if venv_ok:
    lines.append(f"{OK} .venv 已存在: {venv_py}")
else:
    lines.append(f"{MISS} 未找到 .venv\\Scripts\\python.exe")
    lines.append("        → 需要重建 Python 环境（联网，约 1-2 分钟）:")
    lines.append("          方式A（推荐，需先装 uv）:  uv sync")
    lines.append("          方式B:                    pip install -e .")
    lines.append("          uv 安装:  pip install uv  或  https://github.com/astral-sh/uv/releases")
    lines.append("          Python 本体(3.10~3.12):   https://www.python.org/downloads/")

# ---------------------------------------------------------------------------
section("2. 主模型权重  k2-fsa/OmniVoice")
# ---------------------------------------------------------------------------
main_repo = os.path.join(HUB, "models--k2-fsa--OmniVoice")
main_snap = None
if check_dir(main_repo):
    snaps = os.path.join(main_repo, "snapshots")
    if check_dir(snaps):
        subs = [d for d in os.listdir(snaps)
                if check_dir(os.path.join(snaps, d)) and not d.startswith(".")]
        if subs:
            main_snap = os.path.join(snaps, subs[0])
            lines.append(f"{OK} 主模型缓存存在: .../OmniVoice/snapshots/{subs[0][:12]}...")

need_main_files = [
    ("model.safetensors", "主模型权重"),
    (os.path.join("audio_tokenizer", "model.safetensors"), "音频分词器权重"),
    ("config.json", "模型配置"),
]
if main_snap:
    for rel, desc in need_main_files:
        p = os.path.join(main_snap, rel)
        if check_file(p):
            size = os.path.getsize(p)
            lines.append(f"{OK} {desc}: {rel}  ({size/1024/1024:.1f} MB)")
        else:
            lines.append(f"{MISS} {desc} 缺失: {rel}")
else:
    lines.append(f"{MISS} 主模型未下载")
    for rel, desc in need_main_files:
        lines.append(f"{MISS}   {desc}: {rel}")

if not main_snap:
    lines.append("        → 下载主模型（联网，约数 GB）:")
    lines.append("           先设国内镜像再下载:")
    lines.append("             set HF_ENDPOINT=https://hf-mirror.com")
    lines.append("           命令行: .venv\\Scripts\\python.exe -m huggingface_hub.commands.huggingface_cli download k2-fsa/OmniVoice")
    lines.append("           或：直接运行 start_demo.bat，首次会自动联网下载")

# ---------------------------------------------------------------------------
section("3. ASR 模型  openai/whisper-large-v3-turbo  (参考音频自动转写用)")
# ---------------------------------------------------------------------------
whisper_repo = os.path.join(HUB, "models--openai--whisper-large-v3-turbo")
whisper_ok = False
if check_dir(whisper_repo):
    snaps = os.path.join(whisper_repo, "snapshots")
    subs = [d for d in os.listdir(snaps)
            if check_dir(os.path.join(snaps, d)) and not d.startswith(".")] if check_dir(snaps) else []
    if subs and check_file(os.path.join(snaps, subs[0], "model.safetensors")):
        whisper_ok = True
        lines.append(f"{OK} Whisper 缓存存在: .../whisper-large-v3-turbo/snapshots/{subs[0][:12]}...")
    else:
        lines.append(f"{MISS} Whisper 权重文件 model.safetensors 缺失")
else:
    lines.append(f"{MISS} Whisper 模型未下载（不影响克隆/设计，仅影响'不手动填参考文本'的自动转写）")

if not whisper_ok:
    lines.append("        → 下载 ASR 模型（联网，约 1.5 GB）:")
    lines.append("           set HF_ENDPOINT=https://hf-mirror.com")
    lines.append("           .venv\\Scripts\\python.exe -m huggingface_hub.commands.huggingface_cli download openai/whisper-large-v3-turbo")

# ---------------------------------------------------------------------------
section("4. GPU / CUDA 可用性（仅当 torch 已安装时检测）")
# ---------------------------------------------------------------------------
cuda_ok = None
if venv_ok:
    try:
        import subprocess
        code = (
            "import torch, sys;"
            "print('CUDA_AVAILABLE=' + str(torch.cuda.is_available()));"
            "print('CUDA_VERSION=' + (torch.version.cuda or 'None'));"
            "print('DEVICE_COUNT=' + str(torch.cuda.device_count()))"
        )
        r = subprocess.run([venv_py, "-c", code], capture_output=True, text=True, timeout=60)
        out = r.stdout
        for line in out.splitlines():
            if line.startswith("CUDA_AVAILABLE="):
                cuda_ok = line.split("=", 1)[1].strip() == "True"
            if line.startswith("CUDA_VERSION="):
                lines.append(f"        torch CUDA 编译版本: {line.split('=',1)[1].strip()}")
            if line.startswith("DEVICE_COUNT="):
                lines.append(f"        可见显卡数: {line.split('=',1)[1].strip()}")
        if cuda_ok:
            lines.append(f"{OK} CUDA 可用，可使用 GPU 加速推理")
        else:
            lines.append(f"{WARN} torch 已装但 CUDA 不可用（将退化为 CPU，速度很慢 / 可能 OOM）")
            lines.append("        → 目标机需有 NVIDIA 显卡并安装匹配驱动；")
            lines.append("          若环境是从他机复制的，请在该机重新 uv sync 以匹配其 CUDA。")
    except Exception as e:  # noqa
        lines.append(f"{WARN} 未检测到 torch（环境可能不完整），跳过 CUDA 检测")
else:
    lines.append(f"{WARN} .venv 缺失，跳过 CUDA 检测（先重建环境）")

# ---------------------------------------------------------------------------
section("5. 结论")
# ---------------------------------------------------------------------------
all_model_ok = main_snap is not None and whisper_ok
if venv_ok and main_snap and cuda_ok:
    lines.append("  [通过] 环境、模型、CUDA 全部就绪，可直接双击 start_demo.bat / start_studio_web.bat 使用。")
elif venv_ok and main_snap and not cuda_ok:
    lines.append("  [警告] 环境+模型就绪，但 CUDA 不可用，将走 CPU（很慢）。如需 GPU 请重装 torch。")
elif not venv_ok:
    lines.append("  [失败] 缺 Python 环境，请先按第 1 节下载/重建 .venv，再重跑本脚本。")
elif not main_snap:
    lines.append("  [失败] 缺模型权重，请按第 2 节下载 k2-fsa/OmniVoice（联网，或把源机的 .cache 拷过来）。")
else:
    lines.append("  [警告] 部分就绪，请按上面标记 [缺]/[!] 的项补齐。")

lines.append("")
lines.append("说明：仓库代码本身不含模型与 .venv（均被 .gitignore 忽略）。")
lines.append("完整可移植 = 复制'代码 + .cache + .venv'；")
lines.append("最稳妥可移植 = 复制'代码 + .cache'，目标机重跑 uv sync 重建 .venv。")
lines.append("国内下载慢/超时，务必先 set HF_ENDPOINT=https://hf-mirror.com")

print("\n".join(lines))

# 给一个退出码，方便批处理判断
sys.exit(0 if (venv_ok and main_snap) else 1)
