"""Tab — 音效合成（Stable Audio Open 1.0）"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from core.audio_utils import AudioFormat, audio_to_bytes, normalize_audio
from core.app_logger import log_event, EVENT_SYNTHESIZE
from ui.common import task_queue, to_gradio_audio, BASE_DIR, sfx_model

logger = logging.getLogger(__name__)


def tab_sfx():
    with gr.Tab("音效合成"):
        gr.Markdown("### 音效合成（Stable Audio Open 1.0）\n"
                     "输入英文描述生成音效，如 footsteps on gravel、thunder and rain 等。")
        with gr.Row():
            with gr.Column(scale=2):
                prompt_in = gr.Textbox(
                    label="音效描述（英文）",
                    placeholder="e.g. footsteps on a wooden floor, sword clashing...",
                    lines=3,
                )
                negative_in = gr.Textbox(
                    label="负面提示词（可选）",
                    placeholder="e.g. music, speech, low quality",
                    lines=2,
                )
                with gr.Row():
                    duration_slider = gr.Slider(
                        minimum=1, maximum=47, value=10, step=1,
                        label="时长（秒）",
                    )
                    steps_slider = gr.Slider(
                        minimum=50, maximum=200, value=100, step=10,
                        label="推理步数",
                    )
                    guidance_slider = gr.Slider(
                        minimum=1, maximum=15, value=7, step=0.5,
                        label="引导系数",
                    )
                fmt_radio = gr.Radio(choices=["WAV", "MP3"], value="WAV", label="输出格式")
                gen_btn = gr.Button("生成音效", variant="primary")

            with gr.Column(scale=2):
                audio_out = gr.Audio(label="生成结果", type="filepath", interactive=False)
                status_out = gr.Textbox(label="状态", interactive=False)
                send_to_clone_btn = gr.Button("发送到模仿音频设计")

        def on_generate(prompt, negative, duration, steps, guidance, fmt, request: gr.Request):
            if not prompt or not prompt.strip():
                return None, "音效描述不能为空"

            user = (request.username or "unknown") if request else "-"

            try:
                def do_generate():
                    return sfx_model.generate(
                        prompt=prompt.strip(),
                        negative_prompt=negative.strip() if negative else "",
                        duration=duration,
                        steps=int(steps),
                        guidance_scale=guidance,
                    )

                audio, sr = task_queue.submit(
                    user, "sfx", f"音效合成: {prompt[:30]}", do_generate
                )
                audio = normalize_audio(audio)
                out_fmt: AudioFormat = "mp3" if fmt == "MP3" else "wav"
                path = to_gradio_audio(
                    audio_to_bytes(audio, sr, out_fmt), out_fmt, name_hint="sfx"
                )
                duration_s = len(audio) / sr
                rel_path = Path(path).relative_to(BASE_DIR).as_posix()
                log_event(
                    EVENT_SYNTHESIZE,
                    f"类型=音效合成 描述={prompt[:50]} 时长={duration_s:.1f}s 文件={rel_path}",
                    user=user,
                )
                return path, f"生成成功（{duration_s:.1f} 秒）\n已保存：{rel_path}"
            except Exception as e:
                logger.exception("音效合成失败")
                return None, f"音效合成失败：{e}"

        gen_btn.click(
            fn=on_generate,
            inputs=[prompt_in, negative_in, duration_slider, steps_slider, guidance_slider, fmt_radio],
            outputs=[audio_out, status_out],
            concurrency_limit=10,
        )

    return audio_out, send_to_clone_btn
