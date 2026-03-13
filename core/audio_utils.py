"""
core/audio_utils.py

音频工具函数：
  - numpy array → WAV bytes
  - WAV bytes → MP3 bytes（依赖 pydub + ffmpeg）
  - 保存到文件
  - 音量归一化
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
    """将 float32 numpy 数组转为 WAV 格式字节流。"""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def wav_bytes_to_mp3_bytes(wav_bytes: bytes, bitrate: str = "192k") -> bytes:
    """将 WAV 字节流转换为 MP3 字节流（需要 pydub 和 ffmpeg）。"""
    try:
        from pydub import AudioSegment  # type: ignore
    except ImportError as e:
        raise RuntimeError("请安装 pydub：pip install pydub") from e

    wav_buf = io.BytesIO(wav_bytes)
    segment = AudioSegment.from_wav(wav_buf)
    mp3_buf = io.BytesIO()
    segment.export(mp3_buf, format="mp3", bitrate=bitrate)
    return mp3_buf.getvalue()


def audio_to_bytes(
    audio: np.ndarray,
    sample_rate: int,
    output_format: AudioFormat = "wav",
) -> bytes:
    """将音频数组转为目标格式字节流。"""
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
    """将音频数组保存到文件，自动创建父目录。返回实际保存路径。"""
    path = Path(path)
    # 强制使用正确后缀
    path = path.with_suffix(f".{output_format}")
    path.parent.mkdir(parents=True, exist_ok=True)

    audio_bytes = audio_to_bytes(audio, sample_rate, output_format)
    path.write_bytes(audio_bytes)
    logger.debug("音频已保存：%s", path)
    return path


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """峰值归一化，将音频幅度缩放到 target_peak。"""
    peak = np.abs(audio).max()
    if peak < 1e-6:
        return audio
    return (audio / peak * target_peak).astype(np.float32)
