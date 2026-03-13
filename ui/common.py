"""ui/common.py — 共享常量、全局引擎实例、角色数据函数、工具函数"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from core.audio_utils import AudioFormat, audio_to_bytes, normalize_audio
from core.batch_processor import BatchProcessor
from core.tts_engine import TTSEngine
from core.app_logger import log_event, EVENT_SYNTHESIZE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REF_AUDIO_DIR = DATA_DIR / "reference_audio"
OUTPUT_DIR = BASE_DIR / "output"
CHARACTERS_FILE = DATA_DIR / "characters.json"
DESIGN_CHARACTERS_FILE = DATA_DIR / "design_characters.json"

FFMPEG_BIN = BASE_DIR / "ffmpeg" / "ffmpeg.exe"

for _d in (DATA_DIR, REF_AUDIO_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 全局引擎（单例）
# ---------------------------------------------------------------------------
engine = TTSEngine()
batch_processor = BatchProcessor(engine)


# ---------------------------------------------------------------------------
# 角色数据
# ---------------------------------------------------------------------------

def load_characters() -> dict:
    if CHARACTERS_FILE.exists():
        return json.loads(CHARACTERS_FILE.read_text(encoding="utf-8"))
    return {}


def save_characters(characters: dict):
    CHARACTERS_FILE.write_text(
        json.dumps(characters, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def character_display_rows(characters: dict) -> list[list]:
    rows = []
    for name, cfg in characters.items():
        ref_path = cfg.get("ref_audio_path", "")
        ref_text = cfg.get("ref_text", "")
        rows.append([name, ref_path, ref_text, cfg.get("description", "")])
    return rows


def load_design_characters() -> dict:
    if DESIGN_CHARACTERS_FILE.exists():
        return json.loads(DESIGN_CHARACTERS_FILE.read_text(encoding="utf-8"))
    return {}


def save_design_characters(characters: dict):
    DESIGN_CHARACTERS_FILE.write_text(
        json.dumps(characters, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def design_character_display_rows(characters: dict) -> list[list]:
    rows = []
    for name, cfg in characters.items():
        rows.append([name, cfg.get("instruct", ""), cfg.get("description", "")])
    return rows


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def to_gradio_audio(audio_bytes: bytes, fmt: str, name_hint: str = "") -> str:
    """写入 output/日期 子目录，返回路径供 Gradio Audio 组件播放。"""
    now = datetime.datetime.now()
    date_dir = OUTPUT_DIR / now.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    ts = now.strftime("%Y%m%d_%H%M%S")
    if name_hint:
        safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name_hint)
        filename = f"{safe}_{ts}.{fmt}"
    else:
        filename = f"synth_{ts}.{fmt}"
    path = date_dir / filename
    path.write_bytes(audio_bytes)
    return str(path)


def synth_to_file(
    text: str,
    fmt: str,
    voice_name: Optional[str] = None,
    ref_audio=None,
    ref_text: Optional[str] = None,
    name_hint: str = "",
    char_name: str = "",
) -> tuple[Optional[str], str]:
    """通用合成 → (audio_path | None, status)"""
    try:
        audio, sr = engine.synthesize(
            text=text,
            voice_name=voice_name,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        audio = normalize_audio(audio)
        out_fmt: AudioFormat = "mp3" if fmt == "MP3" else "wav"
        path = to_gradio_audio(audio_to_bytes(audio, sr, out_fmt), out_fmt, name_hint)
        duration = len(audio) / sr
        rel_path = Path(path).relative_to(BASE_DIR).as_posix()
        char_info = f" 角色={char_name}" if char_name else ""
        if ref_audio is not None:
            log_event(EVENT_SYNTHESIZE, f"类型=克隆合成{char_info} 文本={text[:50]} 参考文字={ref_text or ''} 格式={fmt} 时长={duration:.1f}s 文件={rel_path}")
        else:
            log_event(EVENT_SYNTHESIZE, f"类型=预设合成{char_info} 文本={text[:50]} 音色={voice_name or '默认'} 格式={fmt} 时长={duration:.1f}s 文件={rel_path}")
        return path, f"合成成功（{len(audio)/sr:.1f} 秒）\n已保存：{rel_path}"
    except Exception as e:
        logger.exception("合成失败")
        return None, f"合成失败：{e}"
