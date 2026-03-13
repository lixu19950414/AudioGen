"""
config.py

全局配置：HuggingFace 环境变量等。
应在其他模块之前导入，确保环境变量尽早生效。
"""

import os

# 强制使用 HTTP 镜像下载模型权重（优先读取环境变量，未设置则使用 hf-mirror.com）
os.environ.setdefault("HF_ENDPOINT", "http://hf-mirror.com")

# 禁用 hf_xet 下载协议，避免兼容性问题
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Gradio 登录账号列表 [(用户名, 密码), ...]
AUTH_USERS = [
    ("admin", "123456"),
]
