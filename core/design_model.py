"""
core/design_model.py

VoiceDesign 音色设计模型：通过自然语言描述生成音色。
模型 ID: Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
"""

from __future__ import annotations

import logging
from typing import Optional

import gc

import numpy as np
import torch

import config  # noqa: F401  确保 HF 环境变量尽早生效
from core.app_logger import log_event, EVENT_MODEL_LOAD, EVENT_MODEL_UNLOAD

logger = logging.getLogger(__name__)

VOICE_DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


class DesignModel:
    """VoiceDesign 音色设计模型（延迟加载）。"""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._model_manager = None

    def set_model_manager(self, manager):
        """注入 ModelManager 引用。"""
        self._model_manager = manager

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """加载 VoiceDesign 模型。"""
        if self._model is not None:
            return
        if self._model_manager is not None:
            self._model_manager.request_load("tts_design")
        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore
        except ImportError as e:
            raise RuntimeError("请安装 qwen-tts：pip install -U qwen-tts") from e

        logger.info("正在加载 VoiceDesign 模型：%s", VOICE_DESIGN_MODEL_ID)
        self._model = Qwen3TTSModel.from_pretrained(
            VOICE_DESIGN_MODEL_ID,
            device_map=self.device,
            dtype=torch.bfloat16,
        )
        logger.info("VoiceDesign 模型加载完成")
        log_event(EVENT_MODEL_LOAD, f"模型=VoiceDesign ({VOICE_DESIGN_MODEL_ID})", user="system")

    def unload(self):
        """卸载模型释放显存。"""
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("VoiceDesign 模型已卸载")
            log_event(EVENT_MODEL_UNLOAD, "VoiceDesign 模型已卸载", user="system")

    def synthesize(self, text: str, instruct: str) -> tuple[np.ndarray, int]:
        """音色设计合成。"""
        self.load()
        audio_list, sr = self._model.generate_voice_design(
            text=text,
            instruct=instruct,
            language="Auto",
            non_streaming_mode=True,
        )
        return np.array(audio_list[0], dtype=np.float32), int(sr)
