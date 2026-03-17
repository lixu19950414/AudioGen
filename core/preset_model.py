"""
core/preset_model.py

CustomVoice 预设音色模型：使用内置说话人合成语音。
模型 ID: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

import config  # noqa: F401  确保 HF 环境变量尽早生效
from core.app_logger import log_event, EVENT_MODEL_LOAD, EVENT_MODEL_UNLOAD

logger = logging.getLogger(__name__)

CUSTOM_VOICE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# CustomVoice 模型内置说话人（可通过 model.get_supported_speakers() 动态获取）
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


class PresetModel:
    """CustomVoice 预设音色模型（延迟加载）。"""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._dynamic_speakers: Optional[list[str]] = None
        self._model_manager = None

    def set_model_manager(self, manager):
        """注入 ModelManager 引用。"""
        self._model_manager = manager

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """加载 CustomVoice 模型。"""
        if self._model is not None:
            return
        if self._model_manager is not None:
            self._model_manager.request_load("tts_preset")
        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore
        except ImportError as e:
            raise RuntimeError("请安装 qwen-tts：pip install -U qwen-tts") from e

        logger.info("正在加载 CustomVoice 模型：%s", CUSTOM_VOICE_MODEL_ID)
        self._model = Qwen3TTSModel.from_pretrained(
            CUSTOM_VOICE_MODEL_ID,
            device_map=self.device,
            dtype=torch.bfloat16,
        )
        # 尝试从模型动态获取说话人列表
        try:
            speakers = self._model.get_supported_speakers()
            if speakers:
                self._dynamic_speakers = [s.title() for s in speakers]
                logger.info("动态说话人列表：%s", self._dynamic_speakers)
        except Exception:
            pass
        logger.info("CustomVoice 模型加载完成")
        log_event(EVENT_MODEL_LOAD, f"模型=CustomVoice ({CUSTOM_VOICE_MODEL_ID})", user="system")

    def unload(self):
        """卸载模型释放显存。"""
        if self._model is not None:
            self._model = None
            self._dynamic_speakers = None
            logger.info("CustomVoice 模型已卸载")
            log_event(EVENT_MODEL_UNLOAD, "CustomVoice 模型已卸载", user="system")

    def get_preset_voices(self) -> list[str]:
        """返回预设说话人列表。"""
        if self._dynamic_speakers is not None:
            return self._dynamic_speakers
        return list(PRESET_VOICES)

    def synthesize(self, text: str, voice_name: Optional[str] = None) -> tuple[np.ndarray, int]:
        """预设人声合成。"""
        self.load()
        presets = self.get_preset_voices()
        speaker = voice_name if (voice_name and voice_name in presets) else presets[0]
        audio_list, sr = self._model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language="Auto",
            non_streaming_mode=True,
        )
        return np.array(audio_list[0], dtype=np.float32), int(sr)
