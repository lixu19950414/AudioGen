"""
app.py — DPAudio 游戏语音合成工具
Gradio 前端，共 11 个 Tab：
  1. 预设人声
  2. 自定义人声（自然语言描述音色）
  3. 克隆人声
  4. 角色管理
  5. 批量处理人声
  6. 工具
  7. 音频浏览
  8. 批量下载
  9. 音效合成
  10. 批量处理音效
  11. 音乐合成
  12. 批量处理音乐
"""

from __future__ import annotations

import config  # noqa: F401  确保 HF 环境变量尽早生效
from config import AUTH_USERS

import logging
import subprocess
import time

import gradio as gr

from ui.common import (
    character_display_rows,
    design_character_display_rows,
    load_characters,
    load_design_characters,
    model_manager,
    task_queue,
)
from ui.tab_single_synth import tab_single_synth
from ui.tab_voice_design import tab_voice_design
from ui.tab_clone import tab_clone
from ui.tab_character_manager import tab_character_manager
from ui.tab_batch import tab_batch
from ui.tab_tools import tab_tools
from ui.tab_audio_browser import tab_audio_browser
from ui.tab_batch_download import tab_batch_download
from ui.tab_sfx import tab_sfx
from ui.tab_batch_sfx import tab_batch_sfx
from ui.tab_music import tab_music
from ui.tab_batch_music import tab_batch_music
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

        # ---- 任务队列状态面板 ----
        with gr.Accordion("任务队列状态", open=True):
            queue_status_md = gr.Markdown("当前无任务")
            queue_timer = gr.Timer(5)

        with gr.Tabs():
            # ---- 语音合成 ----
            synth_audio_out, synth_send_btn = tab_single_synth()
            design_char_dd, design_save_btn, design_audio_out, design_send_btn = tab_voice_design()
            clone_char_dd, clone_save_btn, ref_audio_in = tab_clone()
            char_table, design_table, delete_clone_btn, delete_design_btn = tab_character_manager()
            tab_batch()
            with gr.Tab("┃", interactive=False): pass
            # ---- 音效与音乐 ----
            tab_sfx()
            tab_batch_sfx()
            tab_music()
            tab_batch_music()
            with gr.Tab("┃ ", interactive=False): pass
            # ---- 工具 ----
            browser_audio_out, browser_send_btn = tab_audio_browser()
            tab_batch_download()
            tool_audio_out, tool_send_btn, sep_vocals_out, sep_send_btn = tab_tools()

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

        # 发送到克隆人声
        synth_send_btn.click(fn=lambda a: a, inputs=[synth_audio_out], outputs=[ref_audio_in])
        design_send_btn.click(fn=lambda a: a, inputs=[design_audio_out], outputs=[ref_audio_in])
        tool_send_btn.click(fn=lambda a: a, inputs=[tool_audio_out], outputs=[ref_audio_in])
        sep_send_btn.click(fn=lambda a: a, inputs=[sep_vocals_out], outputs=[ref_audio_in])
        browser_send_btn.click(fn=lambda a: a, inputs=[browser_audio_out], outputs=[ref_audio_in])

        # 页面加载时记录登录日志 + 刷新角色下拉菜单
        def on_page_load(request: gr.Request):
            username = request.username or "unknown"
            client_ip = request.client.host if request.client else "unknown"
            log_event(EVENT_LOGIN, f"用户={username} IP={client_ip}", user=username)
            clone_chars = load_characters()
            clone_choices = ["（不使用角色）"] + [
                name for name, cfg in clone_chars.items()
                if cfg.get("voice_type") == "clone"
            ]
            design_choices = ["（不使用角色）"] + list(load_design_characters().keys())
            return gr.update(choices=clone_choices), gr.update(choices=design_choices)

        demo.load(
            fn=on_page_load,
            outputs=[clone_char_dd, design_char_dd],
        )

        # ---- 任务队列状态轮询 ----
        def _get_gpu_vram_info() -> str:
            """通过 nvidia-smi 获取显存使用情况（MB）。"""
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode != 0:
                    return "显存信息: 获取失败"
                lines = []
                for row in result.stdout.strip().splitlines():
                    parts = [p.strip() for p in row.split(",")]
                    if len(parts) == 4:
                        idx, name, used, total = parts
                        lines.append(f"GPU{idx} ({name}): {used} MB / {total} MB")
                return "**显存**: " + "　|　".join(lines) if lines else "显存信息: 解析失败"
            except Exception:
                return "显存信息: nvidia-smi 不可用"

        def refresh_queue_status():
            status = task_queue.get_status()
            lines = []
            # 显存使用情况
            lines.append(_get_gpu_vram_info())
            lines.append("")
            current = status["current"]
            if current:
                elapsed = time.time() - (current.started_at or current.submitted_at)
                lines.append(f"**正在执行**: {current.description}　"
                             f"(用户: {current.username}, 已运行 {elapsed:.0f}s)")
            else:
                lines.append("**当前空闲**")
            waiting = status["waiting"]
            if waiting:
                lines.append(f"\n**等待队列** ({len(waiting)} 个任务):")
                for i, entry in enumerate(waiting, 1):
                    wait_time = time.time() - entry.submitted_at
                    lines.append(f"{i}. {entry.description} "
                                 f"(用户: {entry.username}, 等待 {wait_time:.0f}s)")
            return "\n".join(lines)

        queue_timer.tick(fn=refresh_queue_status, outputs=[queue_status_md])

    return demo


def cleanup():
    """服务器关闭时清理所有模型，释放 VRAM。"""
    logger.info("服务器关闭，开始清理模型...")
    model_manager.unload_all()
    logger.info("清理完成")


if __name__ == "__main__":
    import atexit

    atexit.register(cleanup)
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        auth=AUTH_USERS,
    )
