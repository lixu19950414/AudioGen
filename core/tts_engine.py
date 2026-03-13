"""
core/tts_engine.py

Qwen3-TTS 模型封装（CustomVoice 模型，支持三种合成模式）：
  - 预设音色合成   generate_custom_voice
  - 参考音频克隆   generate_voice_clone
  - 音色设计合成   generate_voice_design（自然语言描述音色）

所有方法返回 (List[np.ndarray], int)，统一取 result[0] 作为单条音频。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

# 强制使用 HTTP 镜像下载模型权重（优先读取环境变量，未设置则使用 hf-mirror.com）
os.environ.setdefault("HF_ENDPOINT", "http://hf-mirror.com")

logger = logging.getLogger(__name__)

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

# CustomVoice 模型：预设音色合成
CUSTOM_VOICE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
# Base 模型：参考音频克隆（generate_voice_clone 只支持 base 模型）
BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


class TTSEngine:
    """Qwen3-TTS 推理引擎（延迟加载，单例使用）。"""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None           # CustomVoice 模型
        self._base_model = None      # Base 模型（用于克隆）
        self._dynamic_speakers: Optional[list[str]] = None
        logger.info("TTSEngine 初始化，device=%s", self.device)

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def _get_model(self):
        """加载 CustomVoice 模型（预设音色合成）。"""
        if self._model is None:
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
        return self._model

    def _get_base_model(self):
        """加载 Base 模型（参考音频克隆）。"""
        if self._base_model is None:
            try:
                from qwen_tts import Qwen3TTSModel  # type: ignore
            except ImportError as e:
                raise RuntimeError("请安装 qwen-tts：pip install -U qwen-tts") from e

            logger.info("正在加载 Base 模型：%s", BASE_MODEL_ID)
            self._base_model = Qwen3TTSModel.from_pretrained(
                BASE_MODEL_ID,
                device_map=self.device,
                dtype=torch.bfloat16,
            )
            logger.info("Base 模型加载完成")
        return self._base_model

    def get_preset_voices(self) -> list[str]:
        """返回预设说话人列表（模型加载后尝试动态获取，否则用静态列表）。"""
        if self._dynamic_speakers is not None:
            return self._dynamic_speakers
        return list(PRESET_VOICES)

    # ------------------------------------------------------------------
    # 合成接口
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        voice_name: Optional[str] = None,
        ref_audio: Optional[Union[str, Path]] = None,
        ref_text: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """预设音色 / 参考克隆 两合一入口。

        - ref_audio 不为 None → 参考克隆
        - 否则 → 预设音色（voice_name 为空则用第一个预设）
        """
        if not text or not text.strip():
            raise ValueError("合成文本不能为空")
        if ref_audio is not None:
            return self._clone(text, ref_audio, ref_text)
        return self._preset(text, voice_name)


    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _preset(self, text: str, voice_name: Optional[str]) -> tuple[np.ndarray, int]:
        presets = self.get_preset_voices()
        speaker = voice_name if (voice_name and voice_name in presets) else presets[0]
        model = self._get_model()
        audio_list, sr = model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language="Auto",
            non_streaming_mode=True,
        )
        return np.array(audio_list[0], dtype=np.float32), int(sr)

    def _clone(
        self,
        text: str,
        ref_audio: Union[str, Path],
        ref_text: Optional[str],
    ) -> tuple[np.ndarray, int]:
        model = self._get_base_model()
        clone_kwargs = dict(
            text=text,
            ref_audio=str(ref_audio),
            language="Auto",
            non_streaming_mode=True,
        )
        if ref_text and ref_text.strip():
            clone_kwargs["ref_text"] = ref_text
        else:
            clone_kwargs["x_vector_only_mode"] = True
        audio_list, sr = model.generate_voice_clone(**clone_kwargs)
        return np.array(audio_list[0], dtype=np.float32), int(sr)

    # ------------------------------------------------------------------
    # 资源管理
    # ------------------------------------------------------------------

    def unload(self):
        """释放显存。"""
        self._model = None
        self._base_model = None
        self._dynamic_speakers = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("模型已卸载，显存已释放")
