"""
core/music_model.py

ACE-Step 1.5 音乐合成模型。
通过 ACE-Step-1.5 子目录的 acestep 包加载 DiT + LM 进行音乐生成。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch

import config  # noqa: F401  确保 HF 环境变量和 sys.path 尽早生效
from core.app_logger import log_event, EVENT_MODEL_LOAD, EVENT_MODEL_UNLOAD

logger = logging.getLogger(__name__)

# ACE-Step 项目根目录
ACESTEP_ROOT = Path(__file__).resolve().parent.parent / "ACE-Step-1.5"


class MusicModel:
    """ACE-Step 1.5 音乐合成模型（延迟加载）。"""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dit_handler = None
        self._llm_handler = None
        self._model_manager = None
        logger.info("MusicModel 初始化，device=%s", self.device)

    def set_model_manager(self, manager):
        """注入 ModelManager 引用。"""
        self._model_manager = manager

    def is_loaded(self) -> bool:
        return self._dit_handler is not None

    def load(self):
        """延迟加载 DiT + LM handler。"""
        if self._dit_handler is not None:
            return
        if self._model_manager is not None:
            self._model_manager.request_load("music")

        try:
            from acestep.handler import AceStepHandler
            from acestep.llm_inference import LLMHandler
            from acestep.model_downloader import (
                get_checkpoints_dir,
                ensure_main_model,
                ensure_dit_model,
                ensure_lm_model,
            )
        except ImportError as e:
            raise RuntimeError(
                f"ACE-Step 未安装。请克隆仓库到 ACE-Step-1.5/ 并运行 uv sync {e}"
            ) from e

        project_root = str(ACESTEP_ROOT)
        checkpoints_dir = str(get_checkpoints_dir(str(ACESTEP_ROOT / "checkpoints")))

        # 设置环境变量，确保 ACE-Step 能找到项目根目录
        os.environ["ACESTEP_PROJECT_ROOT"] = project_root

        # 确保模型已下载
        logger.info("检查/下载 ACE-Step 模型...")
        ensure_main_model(checkpoints_dir=checkpoints_dir)

        dit_config = "acestep-v15-turbo"
        lm_model = "acestep-5Hz-lm-0.6B"

        ensure_dit_model(dit_config, checkpoints_dir=checkpoints_dir)
        ensure_lm_model(lm_model, checkpoints_dir=checkpoints_dir)

        # 初始化 DiT handler
        logger.info("正在加载 ACE-Step DiT 模型: %s", dit_config)
        dit_handler = AceStepHandler()
        init_status, enable_generate = dit_handler.initialize_service(
            project_root=project_root,
            config_path=dit_config,
            device=self.device,
        )
        logger.info("DiT 初始化状态: %s", init_status)
        if not enable_generate:
            raise RuntimeError(f"ACE-Step DiT 模型初始化失败: {init_status}")

        # 初始化 LLM handler
        logger.info("正在加载 ACE-Step LM 模型: %s", lm_model)
        llm_handler = LLMHandler()
        lm_status, lm_success = llm_handler.initialize(
            checkpoint_dir=checkpoints_dir,
            lm_model_path=lm_model,
            backend="pt",
            device=self.device,
        )
        logger.info("LM 初始化状态: %s (success=%s)", lm_status, lm_success)
        if not lm_success:
            raise RuntimeError(f"ACE-Step LM 模型初始化失败: {lm_status}")

        self._dit_handler = dit_handler
        self._llm_handler = llm_handler
        logger.info("ACE-Step 音乐模型加载完成")
        log_event(EVENT_MODEL_LOAD, f"模型=ACE-Step (DiT={dit_config}, LM={lm_model})", user="system")

    def generate(
        self,
        task_type: str = "text2music",
        caption: str = "",
        lyrics: str = "",
        instrumental: bool = False,
        vocal_language: str = "unknown",
        duration: float = 30.0,
        bpm: Optional[int] = None,
        keyscale: str = "",
        timesignature: str = "",
        reference_audio: Optional[str] = None,
        src_audio: Optional[str] = None,
        inference_steps: int = 8,
        guidance_scale: float = 7.0,
        seed: int = -1,
        repainting_start: float = 0.0,
        repainting_end: float = -1,
        repaint_strength: float = 0.5,
        repaint_mode: str = "balanced",
        audio_cover_strength: float = 1.0,
        cover_noise_strength: float = 0.0,
    ) -> dict:
        """
        统一生成接口。

        返回 GenerationResult 字典，包含 audios 列表、success 状态等。
        """
        if not caption and not lyrics:
            raise ValueError("音乐描述(caption)和歌词(lyrics)不能同时为空")

        self.load()

        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        params = GenerationParams(
            task_type=task_type,
            caption=caption,
            lyrics=lyrics,
            instrumental=instrumental,
            vocal_language=vocal_language,
            duration=duration,
            bpm=bpm,
            keyscale=keyscale,
            timesignature=timesignature,
            reference_audio=reference_audio,
            src_audio=src_audio,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            repainting_start=repainting_start,
            repainting_end=repainting_end,
            repaint_strength=repaint_strength,
            repaint_mode=repaint_mode,
            audio_cover_strength=audio_cover_strength,
            cover_noise_strength=cover_noise_strength,
        )

        gen_config = GenerationConfig(
            batch_size=1,
            audio_format="wav",
        )

        result = generate_music(
            dit_handler=self._dit_handler,
            llm_handler=self._llm_handler,
            params=params,
            config=gen_config,
        )

        return result

    def unload(self):
        """释放显存。"""
        if self._dit_handler is not None:
            self._dit_handler = None
            self._llm_handler = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("ACE-Step 音乐模型已卸载，显存已释放")
            log_event(EVENT_MODEL_UNLOAD, "ACE-Step 音乐模型已卸载", user="system")
