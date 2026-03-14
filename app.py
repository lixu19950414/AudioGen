"""
app.py — DPAudio 游戏语音合成工具
Gradio 前端，共 7 个 Tab：
  1. 预设音色合成
  2. 自定义音色设计（自然语言描述音色）
  3. 模仿音频设计
  4. 角色管理
  5. 批量处理
  6. 工具
  7. 音频浏览
"""

from __future__ import annotations

import config  # noqa: F401  确保 HF 环境变量尽早生效
from config import AUTH_USERS

import logging

import gradio as gr

from ui.common import (
    character_display_rows,
    design_character_display_rows,
    load_characters,
    load_design_characters,
)
from ui.tab_single_synth import tab_single_synth
from ui.tab_voice_design import tab_voice_design
from ui.tab_clone import tab_clone
from ui.tab_character_manager import tab_character_manager
from ui.tab_batch import tab_batch
from ui.tab_tools import tab_tools
from ui.tab_audio_browser import tab_audio_browser
from ui.tab_batch_download import tab_batch_download
from core.app_logger import log_event, EVENT_LOGIN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AudioGen — 游戏语音合成工具") as demo:
        gr.Markdown(
            "# AudioGen — 游戏语音合成工具\n"
        )
        synth_audio_out, synth_send_btn = tab_single_synth()
        design_char_dd, design_save_btn, design_audio_out, design_send_btn = tab_voice_design()
        clone_char_dd, clone_save_btn, ref_audio_in = tab_clone()
        char_table, design_table, delete_clone_btn, delete_design_btn = tab_character_manager()
        tab_batch()
        tool_audio_out, tool_send_btn = tab_tools()
        browser_audio_out, browser_send_btn = tab_audio_browser()
        tab_batch_download()

        # 跨 Tab 自动刷新：保存角色 → 刷新管理表格
        clone_save_btn.click(
            fn=lambda: character_display_rows(load_characters()),
            outputs=[char_table],
        )
        design_save_btn.click(
            fn=lambda: design_character_display_rows(load_design_characters()),
            outputs=[design_table],
        )
        # 跨 Tab 自动刷新：删除角色 → 刷新合成 Tab 下拉
        delete_clone_btn.click(
            fn=lambda: gr.update(choices=["（不使用角色）"] + list(load_characters().keys())),
            outputs=[clone_char_dd],
        )
        delete_design_btn.click(
            fn=lambda: gr.update(choices=["（不使用角色）"] + list(load_design_characters().keys())),
            outputs=[design_char_dd],
        )

        # 发送到模仿音频设计
        synth_send_btn.click(fn=lambda a: a, inputs=[synth_audio_out], outputs=[ref_audio_in])
        design_send_btn.click(fn=lambda a: a, inputs=[design_audio_out], outputs=[ref_audio_in])
        tool_send_btn.click(fn=lambda a: a, inputs=[tool_audio_out], outputs=[ref_audio_in])
        browser_send_btn.click(fn=lambda a: a, inputs=[browser_audio_out], outputs=[ref_audio_in])

        # 页面加载时记录登录日志
        def on_page_load(request: gr.Request):
            username = request.username or "unknown"
            client_ip = request.client.host if request.client else "unknown"
            log_event(EVENT_LOGIN, f"用户={username} IP={client_ip}", user=username)

        demo.load(fn=on_page_load)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        auth=AUTH_USERS,
    )
