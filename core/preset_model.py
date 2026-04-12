"""
core/preset_model.py

预设音色模型：基于 VoxCPM2 的预设人声合成。
向后兼容原 PresetModel API。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import config  # noqa: F401  确保 HF 环境变量尽早生效
from core.voxcpm_model import (
    VoxCPMModel,
    PRESET_VOICES,
    VOICE_DESCRIPTIONS,
)

__all__ = ["PresetModel", "PRESET_VOICES", "VOICE_DESCRIPTIONS"]


class PresetModel:
    """VoxCPM2 预设音色模型（延迟加载）。"""

    def __init__(self, device: Optional[str] = None):
        self._model = VoxCPMModel(device)

    def set_model_manager(self, manager):
        self._model.set_model_manager(manager)

    def is_loaded(self) -> bool:
        return self._model.is_loaded()

    def load(self):
        self._model.load()

    def unload(self):
        self._model.unload()

    def get_preset_voices(self) -> list[str]:
        """返回预设说话人列表。"""
        return list(PRESET_VOICES)

    def synthesize(self, text: str, voice_name: Optional[str] = None) -> tuple[np.ndarray, int]:
        """预设人声合成。"""
        return self._model.synthesize(text=text, voice_name=voice_name)
