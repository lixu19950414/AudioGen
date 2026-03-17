"""
config.py

全局配置：HuggingFace 环境变量等。
应在其他模块之前导入，确保环境变量尽早生效。
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 将项目内的 ffmpeg 目录加入 PATH，确保音频转换可用
_FFMPEG_DIR = str(BASE_DIR / "ffmpeg")
if _FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

# ACE-Step-1.5 子目录加入 sys.path，使 import acestep 可用
_ACESTEP_DIR = BASE_DIR / "ACE-Step-1.5"
if _ACESTEP_DIR.exists() and str(_ACESTEP_DIR) not in sys.path:
    sys.path.insert(0, str(_ACESTEP_DIR))

# 模型缓存目录：项目根目录下的 models/（替代默认的 ~/.cache/huggingface/hub/）
_MODELS_DIR = str(BASE_DIR / "models")
os.environ["HF_HUB_CACHE"] = _MODELS_DIR

# Demucs 等 torch.hub 模型也缓存到 models/ 目录
os.environ.setdefault("TORCH_HOME", _MODELS_DIR)

# 强制使用 HTTP 镜像下载模型权重（优先读取环境变量，未设置则使用 hf-mirror.com）
# os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 禁用 hf_xet 下载协议，避免兼容性问题
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Gradio 登录账号列表 [(用户名, 密码), ...]
AUTH_USERS = [
    ("admin", "123456"),
    ("admin2", "123456"),
]
