"""
core/tts_engine.py

Qwen3-TTS 模型封装，支持：
  - Base 模型预设音色合成
  - CustomVoice 模型参考音频克隆
  - 延迟加载，节省显存
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

# 预设音色列表（Qwen3-TTS CustomVoice 模型内置说话人）
PRESET_VOICES: list[str] = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]

BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CUSTOM_VOICE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# 模型本地缓存目录（相对于本文件两级上的项目根）
_PROJECT_ROOT = Path(__file__).parent.parent
MODEL_CACHE_DIR = _PROJECT_ROOT / "models"


class TTSEngine:
    """Qwen3-TTS 推理引擎（延迟加载两个模型）。"""

    def __init__(self, device: Optional[str] = None, use_flash_attn: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_flash_attn = use_flash_attn
        self._base_model = None
        self._custom_model = None
        logger.info("TTSEngine 初始化，device=%s", self.device)

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def _load_model(self, model_id: str):
        """加载并返回 Qwen3TTSModel 实例。"""
        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "请先安装 qwen-tts：pip install -U qwen-tts"
            ) from e

        kwargs: dict = {
            "device_map": self.device,
            "dtype": torch.bfloat16,
            "cache_dir": str(MODEL_CACHE_DIR),
        }
        if self.use_flash_attn:
            kwargs["attn_implementation"] = "flash_attention_2"

        logger.info("正在加载模型：%s", model_id)
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
        logger.info("模型加载完成：%s", model_id)
        return model

    def _get_base_model(self):
        if self._base_model is None:
            self._base_model = self._load_model(BASE_MODEL_ID)
        return self._base_model

    def _get_custom_model(self):
        if self._custom_model is None:
            self._custom_model = self._load_model(CUSTOM_VOICE_MODEL_ID)
        return self._custom_model

    # ------------------------------------------------------------------
    # 合成接口
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        voice_name: Optional[str] = None,
        ref_audio: Optional[Union[str, Path, tuple]] = None,
        ref_text: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """合成语音，返回 (audio_array, sample_rate)。

        参数:
            text: 待合成文本
            voice_name: 预设音色名称（无 ref_audio 时生效）
            ref_audio: 参考音频（本地路径 / URL / (numpy_array, sr) 元组）
            ref_text: 参考音频对应文字（提升克隆质量）

        返回:
            (np.ndarray float32, sample_rate int)
        """
        if not text or not text.strip():
            raise ValueError("合成文本不能为空")

        if ref_audio is not None:
            return self._synthesize_clone(text, ref_audio, ref_text)
        else:
            return self._synthesize_base(text, voice_name)

    def _synthesize_base(
        self, text: str, voice_name: Optional[str]
    ) -> tuple[np.ndarray, int]:
        model = self._get_custom_model()
        speaker = voice_name if (voice_name and voice_name in PRESET_VOICES) else PRESET_VOICES[0]
        audio_array, sr = model.generate_custom_voice(
            text=text,
            language="Auto",
            speaker=speaker,
        )
        return np.array(audio_array, dtype=np.float32), int(sr)

    def _synthesize_clone(
        self,
        text: str,
        ref_audio: Union[str, Path, tuple],
        ref_text: Optional[str],
    ) -> tuple[np.ndarray, int]:
        model = self._get_custom_model()
        kwargs: dict = {
            "text": text,
            "language": "Auto",
            "ref_audio": ref_audio,
            "ref_text": ref_text or "",
        }
        audio_array, sr = model.generate_voice_clone(**kwargs)
        return np.array(audio_array, dtype=np.float32), int(sr)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def get_preset_voices() -> list[str]:
        return list(PRESET_VOICES)

    def unload(self):
        """释放显存。"""
        self._base_model = None
        self._custom_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("模型已卸载，显存已释放")
