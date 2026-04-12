"""
core/voxcpm_model.py

VoxCPM2 统一 TTS 模型：使用 openbmb/VoxCPM2 替代 Qwen3-TTS。
支持基础合成、参考音频克隆、音色设计三种模式。
"""

from __future__ import annotations

import logging
from typing import Optional

import gc

import numpy as np

import config  # noqa: F401  确保 HF 环境变量尽早生效
from core.app_logger import log_event, EVENT_MODEL_LOAD, EVENT_MODEL_UNLOAD

logger = logging.getLogger(__name__)

VOXCPM2_MODEL_ID = "openbmb/VoxCPM2"

# 预设音色名称 → 音色描述映射（用于向后兼容原有角色配置）
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

# 音色描述文本
VOICE_DESCRIPTIONS: dict[str, str] = {
    "Vivian": "A young woman, clear and bright voice",
    "Serena": "A young woman, gentle and sweet voice",
    "Uncle_Fu": "A middle-aged man, warm and deep voice",
    "Dylan": "A young man, clear and energetic voice",
    "Eric": "A young man, calm and steady voice",
    "Ryan": "A young man, cheerful and confident voice",
    "Aiden": "A young man, gentle and polite voice",
    "Ono_Anna": "A young woman, soft and elegant voice",
    "Sohee": "A young woman, lively and cute voice",
}


class VoxCPMModel:
    """VoxCPM2 统一 TTS 模型（延迟加载）。"""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available() else "cpu")
        self._model = None
        self._model_manager = None
        self._sample_rate = 48000  # VoxCPM2 默认采样率 48kHz

    def set_model_manager(self, manager):
        """注入 ModelManager 引用。"""
        self._model_manager = manager

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """加载 VoxCPM2 模型。"""
        if self._model is not None:
            return
        if self._model_manager is not None:
            self._model_manager.request_load("tts_preset")
        try:
            from voxcpm import VoxCPM  # type: ignore
        except ImportError as e:
            raise RuntimeError("请安装 voxcpm：pip install voxcpm") from e

        logger.info("正在加载 VoxCPM2 模型：%s", VOXCPM2_MODEL_ID)
        self._model = VoxCPM.from_pretrained(
            VOXCPM2_MODEL_ID,
            load_denoiser=False,
        )
        # 获取采样率
        try:
            self._sample_rate = self._model.tts_model.sample_rate
        except Exception:
            self._sample_rate = 48000
        logger.info("VoxCPM2 模型加载完成，采样率 %dHz", self._sample_rate)
        log_event(EVENT_MODEL_LOAD, f"模型=VoxCPM2 ({VOXCPM2_MODEL_ID})", user="system")

    def unload(self):
        """卸载模型释放显存。"""
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("VoxCPM2 模型已卸载")
            log_event(EVENT_MODEL_UNLOAD, "VoxCPM2 模型已卸载", user="system")

    def get_preset_voices(self) -> list[str]:
        """返回预设音色名称列表。"""
        return list(PRESET_VOICES)

    # ------------------------------------------------------------------
    # 基础合成（指定音色名称或使用音色描述）
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        voice_name: Optional[str] = None,
        voice_description: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """预设人声合成。
        如果提供了 voice_name 且在 PRESET_VOICES 中，使用对应的音色描述；
        否则使用 voice_description 或直接基础合成。
        """
        self.load()

        # 确定音色描述
        desc = None
        if voice_name and voice_name in VOICE_DESCRIPTIONS:
            desc = VOICE_DESCRIPTIONS[voice_name]
        elif voice_description:
            desc = voice_description

        if desc:
            # 带音色描述的合成
            wav = self._model.generate(
                text=f"({desc}){text}",
                cfg_value=2.0,
                inference_timesteps=10,
            )
        else:
            # 基础合成（无特定音色）
            wav = self._model.generate(
                text=text,
                cfg_value=2.0,
                inference_timesteps=10,
            )

        return np.array(wav, dtype=np.float32), int(self._sample_rate)

    # ------------------------------------------------------------------
    # 参考音频克隆
    # ------------------------------------------------------------------

    def synthesize_clone(
        self,
        text: str,
        ref_audio: str,
        ref_text: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """参考音频克隆人声。"""
        self.load()

        if ref_text and ref_text.strip():
            # Ultimate cloning：参考音频 + 文字
            wav = self._model.generate(
                text=text,
                prompt_wav_path=str(ref_audio),
                prompt_text=ref_text.strip(),
                reference_wav_path=str(ref_audio),
                cfg_value=2.0,
                inference_timesteps=10,
            )
        else:
            # 基础克隆：仅参考音频
            wav = self._model.generate(
                text=text,
                reference_wav_path=str(ref_audio),
                cfg_value=2.0,
                inference_timesteps=10,
            )

        return np.array(wav, dtype=np.float32), int(self._sample_rate)

    # ------------------------------------------------------------------
    # 音色设计
    # ------------------------------------------------------------------

    def synthesize_design(
        self,
        text: str,
        instruct: str,
    ) -> tuple[np.ndarray, int]:
        """音色设计合成。"""
        self.load()

        wav = self._model.generate(
            text=f"({instruct}){text}",
            cfg_value=2.0,
            inference_timesteps=10,
        )

        return np.array(wav, dtype=np.float32), int(self._sample_rate)
