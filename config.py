"""
config.py

全局配置：HuggingFace 环境变量等。
应在其他模块之前导入，确保环境变量尽早生效。
"""

import os
from pathlib import Path

# 模型缓存目录：项目根目录下的 models/（替代默认的 ~/.cache/huggingface/hub/）
_MODELS_DIR = str(Path(__file__).resolve().parent / "models")
os.environ["HF_HUB_CACHE"] = _MODELS_DIR

# 强制使用 HTTP 镜像下载模型权重（优先读取环境变量，未设置则使用 hf-mirror.com）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 禁用 hf_xet 下载协议，避免兼容性问题
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Gradio 登录账号列表 [(用户名, 密码), ...]
AUTH_USERS = [
    ("admin", "123456"),
    ("admin2", "123456"),
]
