"""
core/design_model.py

音色设计模型：基于 VoxCPM2 的音色设计合成。
向后兼容原 DesignModel API。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import config  # noqa: F401  确保 HF 环境变量尽早生效
from core.voxcpm_model import VoxCPMModel


class DesignModel:
    """VoxCPM2 音色设计模型（延迟加载）。"""

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

    def synthesize(self, text: str, instruct: str) -> tuple[np.ndarray, int]:
        """音色设计合成。"""
        return self._model.synthesize_design(text=text, instruct=instruct)
