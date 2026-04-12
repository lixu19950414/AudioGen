"""
core/clone_model.py

参考音频克隆模型：基于 VoxCPM2 的克隆人声合成。
向后兼容原 CloneModel API。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

import config  # noqa: F401  确保 HF 环境变量尽早生效
from core.voxcpm_model import VoxCPMModel


class CloneModel:
    """VoxCPM2 参考音频克隆模型（延迟加载）。"""

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

    def synthesize(
        self,
        text: str,
        ref_audio: Union[str, Path],
        ref_text: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """参考音频克隆人声。"""
        return self._model.synthesize_clone(text=text, ref_audio=str(ref_audio), ref_text=ref_text)
