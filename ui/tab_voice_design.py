"""Tab 2 — 自定义音色设计"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from core.audio_utils import AudioFormat, audio_to_bytes
from core.audio_utils import normalize_audio
from ui.common import (
    BASE_DIR,
    engine,
    task_queue,
    load_design_characters,
    save_design_characters,
    to_gradio_audio,
)
from core.app_logger import log_event, EVENT_SYNTHESIZE, EVENT_CHAR_ADD


def tab_voice_design():
    with gr.Tab("自定义音色设计"):
        gr.Markdown('通过自然语言描述生成音色，如"一个温柔的年轻女性声音，语速适中"')
        with gr.Row():
            with gr.Column(scale=1):
                def _design_char_choices():
                    return ["（不使用角色）"] + list(load_design_characters().keys())

                design_char_dd = gr.Dropdown(
                    choices=_design_char_choices(),
                    value="（不使用角色）",
                    label="从设计角色库选择",
                )
                design_instruct = gr.Textbox(
                    label="音色描述",
                    placeholder="例：一个低沉有磁性的中年男性声音",
                    lines=3,
                )
                design_text = gr.Textbox(
                    label="待合成文本",
                    placeholder="输入想要合成的文字...",
                    lines=5,
                )
                design_fmt = gr.Radio(choices=["WAV", "MP3"], value="WAV", label="输出格式")
                design_btn = gr.Button("生成语音", variant="primary")
            with gr.Column(scale=2):
                design_audio_out = gr.Audio(label="合成结果", type="filepath", interactive=False)
                design_status = gr.Textbox(label="状态", interactive=False)
                design_send_to_clone_btn = gr.Button("发送到模仿音频设计")

                gr.Markdown("---\n**保存为设计角色**")
                design_save_name = gr.Textbox(label="角色名", placeholder="例：温柔少女")
                design_save_desc = gr.Textbox(label="描述（可选）")
                design_save_btn = gr.Button("保存角色", variant="secondary")
                design_save_status = gr.Textbox(label="保存状态", interactive=False)

        def on_char_select(char_name):
            if not char_name or char_name == "（不使用角色）":
                return gr.update(value="")
            chars = load_design_characters()
            cfg = chars.get(char_name, {})
            return gr.update(value=cfg.get("instruct", ""))

        design_char_dd.change(
            fn=on_char_select,
            inputs=[design_char_dd],
            outputs=[design_instruct],
        )

        def on_voice_design(instruct, text, fmt, request: gr.Request):
            if not instruct.strip():
                return None, "请输入音色描述"
            if not text.strip():
                return None, "待合成文本不能为空"
            user = (request.username or "unknown") if request else "-"
            try:
                task_desc = f"音色设计: {text[:30]}"

                def do_design():
                    return engine.voice_design(text=text, instruct=instruct)

                audio, sr = task_queue.submit(user, "design", task_desc, do_design)
                audio = normalize_audio(audio)
                out_fmt: AudioFormat = "mp3" if fmt == "MP3" else "wav"
                path = to_gradio_audio(audio_to_bytes(audio, sr, out_fmt), out_fmt, name_hint="design")
                rel_path = Path(path).relative_to(BASE_DIR).as_posix()
                log_event(EVENT_SYNTHESIZE, f"类型=音色设计 文本={text[:50]} 音色描述={instruct[:50]} 格式={fmt} 时长={len(audio)/sr:.1f}s 文件={rel_path}", user=user)
                return path, f"合成成功（{len(audio)/sr:.1f} 秒）\n已保存：{rel_path}"
            except Exception as e:
                return None, f"合成失败：{e}"

        design_btn.click(
            fn=on_voice_design,
            inputs=[design_instruct, design_text, design_fmt],
            outputs=[design_audio_out, design_status],
            concurrency_limit=10,
        )

        def on_save_design_char(instruct, char_name, char_desc, request: gr.Request):
            char_name = char_name.strip()
            if not char_name:
                return "角色名不能为空", gr.update()
            if not instruct.strip():
                return "音色描述不能为空", gr.update()
            user = (request.username or "unknown") if request else "-"
            chars = load_design_characters()
            chars[char_name] = {
                "description": char_desc.strip(),
                "voice_type": "design",
                "instruct": instruct.strip(),
            }
            save_design_characters(chars)
            log_event(EVENT_CHAR_ADD, f"设计角色={char_name} 描述={instruct.strip()[:30]}", user=user)
            new_choices = ["（不使用角色）"] + list(chars.keys())
            return (
                f"已保存设计角色 [{char_name}]",
                gr.update(choices=new_choices),
            )

        design_save_btn.click(
            fn=on_save_design_char,
            inputs=[design_instruct, design_save_name, design_save_desc],
            outputs=[design_save_status, design_char_dd],
        )

    return design_char_dd, design_save_btn, design_audio_out, design_send_to_clone_btn
