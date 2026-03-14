"""
core/asr_engine.py

Whisper 语音识别引擎（延迟加载，单例使用）。
使用 openai/whisper-large-v3 模型自动识别音频中的文字。
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

import config  # noqa: F401  确保 HF 环境变量尽早生效
from core.app_logger import log_event, EVENT_MODEL_LOAD, EVENT_MODEL_UNLOAD

logger = logging.getLogger(__name__)

WHISPER_MODEL_ID = "openai/whisper-large-v3"


class ASREngine:
    """Whisper 语音识别引擎（延迟加载，单例使用）。"""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = None
        logger.info("ASREngine 初始化，device=%s", self.device)

    def _get_pipe(self):
        """延迟加载 Whisper pipeline。"""
        if self._pipe is None:
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
        return self._pipe

    def recognize(self, audio_path: str) -> str:
        """识别音频文件中的文字。"""
        pipe = self._get_pipe()
        result = pipe(audio_path, return_timestamps=False)
        return result["text"].strip()

    def unload(self):
        """释放模型显存。"""
        if self._pipe is not None:
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Whisper 模型已卸载")
            log_event(EVENT_MODEL_UNLOAD, "Whisper 模型已卸载", user="system")
