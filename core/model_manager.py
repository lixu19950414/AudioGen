"""
core/model_manager.py

全局模型管理器：统一管理所有模型引擎的加载/卸载生命周期。
同一时刻只允许一个模型占用 VRAM，加载新模型前自动卸载其他模型。
"""

from __future__ import annotations

import gc
import logging

import torch

from core.app_logger import log_event, EVENT_MODEL_UNLOAD

logger = logging.getLogger(__name__)


class ModelManager:
    """全局模型管理器（单例使用）。"""

    def __init__(self):
        self._engines: dict[str, object] = {}

    def register(self, name: str, engine):
        """注册引擎（引擎需实现 unload() 方法）。"""
        self._engines[name] = engine
        logger.debug("ModelManager 注册引擎：%s", name)

    def request_load(self, name: str):
        """请求加载某引擎：先卸载其他所有引擎释放 VRAM。"""
        unloaded = []
        for n, eng in self._engines.items():
            if n != name:
                eng.unload()
                unloaded.append(n)
        if unloaded:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if unloaded:
            logger.info("为加载 %s，已卸载引擎：%s", name, ", ".join(unloaded))

    def unload_all(self):
        """卸载所有引擎。"""
        for eng in self._engines.values():
            eng.unload()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("所有引擎已卸载")
        log_event(EVENT_MODEL_UNLOAD, "所有模型已卸载", user="system")

    def unload(self, name: str):
        """卸载指定引擎。"""
        if name in self._engines:
            self._engines[name].unload()
