"""
core/sfx_model.py

Stable Audio Open 1.0 音效合成模型。
使用 diffusers 的 StableAudioPipeline 延迟加载模型。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import gc

import numpy as np
import torch

from core.app_logger import log_event, EVENT_MODEL_LOAD, EVENT_MODEL_UNLOAD

logger = logging.getLogger(__name__)

SFX_MODEL_ID = "stabilityai/stable-audio-open-1.0"


class SfxModel:
    """Stable Audio Open 1.0 音效合成模型（延迟加载）。"""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = None
        self._model_manager = None
        logger.info("SfxModel 初始化，device=%s", self.device)

    def set_model_manager(self, manager):
        """注入 ModelManager 引用。"""
        self._model_manager = manager

    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load(self):
        """延迟加载 StableAudioPipeline，加载前通过 ModelManager 卸载其他模型。"""
        if self._pipe is not None:
            return
        if self._model_manager is not None:
            self._model_manager.request_load("sfx")

        try:
            from diffusers import StableAudioPipeline
        except ImportError as e:
            raise RuntimeError(
                "请安装 diffusers：pip install diffusers>=0.30.0"
            ) from e

        logger.info("正在加载音效模型：%s", SFX_MODEL_ID)
        self._pipe = StableAudioPipeline.from_pretrained(
            SFX_MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=os.environ.get("HF_HUB_CACHE"),
        ).to(self.device)
        logger.info("音效模型加载完成")
        log_event(EVENT_MODEL_LOAD, f"模型=SFX ({SFX_MODEL_ID})", user="system")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        duration: float = 10.0,
        steps: int = 100,
        guidance_scale: float = 7.0,
    ) -> tuple[np.ndarray, int]:
        """生成音效，返回 (np.ndarray, sample_rate)。"""
        if not prompt or not prompt.strip():
            raise ValueError("音效描述不能为空")

        self.load()
        output = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=steps,
            audio_end_in_s=duration,
            guidance_scale=guidance_scale,
        )
        audio = output.audios[0]  # tensor or ndarray, shape: (channels, samples)
        sr = self._pipe.vae.config.sampling_rate

        # 确保转为 numpy
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()

        # 转为单声道 (samples,) 以兼容项目的 audio_utils
        if audio.ndim == 2:
            audio = audio.mean(axis=0)

        return audio.astype(np.float32), int(sr)

    def unload(self):
        """释放显存。"""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("音效模型已卸载，显存已释放")
            log_event(EVENT_MODEL_UNLOAD, "音效模型已卸载", user="system")
