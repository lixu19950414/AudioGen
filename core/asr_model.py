"""
core/asr_model.py

Whisper 语音识别模型（延迟加载）。
使用 openai/whisper-large-v3 模型自动识别音频中的文字。
"""

from __future__ import annotations

import logging
from typing import Optional

import gc

import torch

import config  # noqa: F401  确保 HF 环境变量尽早生效
from core.app_logger import log_event, EVENT_MODEL_LOAD, EVENT_MODEL_UNLOAD

logger = logging.getLogger(__name__)

WHISPER_MODEL_ID = "openai/whisper-large-v3"


class AsrModel:
    """Whisper 语音识别模型（延迟加载）。"""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = None
        self._model_manager = None
        logger.info("AsrModel 初始化，device=%s", self.device)

    def set_model_manager(self, manager):
        """注入 ModelManager 引用。"""
        self._model_manager = manager

    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load(self):
        """延迟加载 Whisper pipeline。"""
        if self._pipe is not None:
            return
        if self._model_manager is not None:
            self._model_manager.request_load("asr")

        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info("正在加载 Whisper 模型：%s", WHISPER_MODEL_ID)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL_ID,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.to(self.device)
        processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
        )
        logger.info("Whisper 模型加载完成")
        log_event(EVENT_MODEL_LOAD, f"模型=Whisper ({WHISPER_MODEL_ID})", user="system")

    def recognize(self, audio_path: str) -> str:
        """识别音频文件中的文字。"""
        self.load()
        result = self._pipe(audio_path, return_timestamps=False)
        return result["text"].strip()

    def unload(self):
        """释放模型显存。"""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Whisper 模型已卸载")
            log_event(EVENT_MODEL_UNLOAD, "Whisper 模型已卸载", user="system")
