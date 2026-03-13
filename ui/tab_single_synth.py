"""Tab 1 — 预设音色合成"""

from __future__ import annotations

import gradio as gr

from core.tts_engine import PRESET_VOICES
from ui.common import synth_to_file


def tab_single_synth():
    voice_choices = ["（默认）"] + PRESET_VOICES

    with gr.Tab("预设音色合成"):
        gr.Markdown("### 预设音色文本合成")
        with gr.Row():
            with gr.Column(scale=2):
                text_in = gr.Textbox(label="合成文本", placeholder="请输入要合成的文字…", lines=5)
                voice_dd = gr.Dropdown(choices=voice_choices, value="（默认）", label="音色选择")
                fmt_radio = gr.Radio(choices=["WAV", "MP3"], value="WAV", label="输出格式")
                synth_btn = gr.Button("生成语音", variant="primary")
            with gr.Column(scale=2):
                audio_out = gr.Audio(label="合成结果", type="filepath", interactive=False)
                status_out = gr.Textbox(label="状态", interactive=False)
                send_to_clone_btn = gr.Button("发送到模仿音频设计")

        def on_synth(text, voice, fmt):
            if not text.strip():
                return None, "文本不能为空"
            voice_name = voice if voice and voice != "（默认）" else None
            hint = voice if voice and voice != "（默认）" else ""
            return synth_to_file(text, fmt, voice_name, name_hint=hint)

        synth_btn.click(
            fn=on_synth,
            inputs=[text_in, voice_dd, fmt_radio],
            outputs=[audio_out, status_out],
        )

    return audio_out, send_to_clone_btn
