"""
core/audio_utils.py

音频工具：numpy array → WAV/MP3 bytes，保存文件，峰值归一化。
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

AudioFormat = Literal["wav", "mp3"]


def array_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def wav_bytes_to_mp3_bytes(wav_bytes: bytes, bitrate: str = "192k") -> bytes:
    try:
        from pydub import AudioSegment  # type: ignore
    except ImportError as e:
        raise RuntimeError("MP3 输出需要 pydub：pip install pydub（同时需要系统安装 ffmpeg）") from e

    segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    buf = io.BytesIO()
    segment.export(buf, format="mp3", bitrate=bitrate)
    return buf.getvalue()


def audio_to_bytes(
    audio: np.ndarray,
    sample_rate: int,
    output_format: AudioFormat = "wav",
) -> bytes:
    wav_bytes = array_to_wav_bytes(audio, sample_rate)
    if output_format == "mp3":
        return wav_bytes_to_mp3_bytes(wav_bytes)
    return wav_bytes


def save_audio(
    audio: np.ndarray,
    sample_rate: int,
    path: Path | str,
    output_format: AudioFormat = "wav",
) -> Path:
    path = Path(path).with_suffix(f".{output_format}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio_to_bytes(audio, sample_rate, output_format))
    logger.debug("已保存：%s", path)
    return path


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = np.abs(audio).max()
    if peak < 1e-6:
        return audio
    return (audio / peak * target_peak).astype(np.float32)
