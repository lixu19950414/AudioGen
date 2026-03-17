"""Tab 3 — 克隆合成"""

from __future__ import annotations

import shutil
from pathlib import Path

import gradio as gr

from ui.common import (
    BASE_DIR,
    REF_AUDIO_DIR,
    load_characters,
    save_characters,
    synth_to_file,
    asr_model,
    task_queue,
)
from core.app_logger import log_event, EVENT_CHAR_ADD


def tab_clone():
    with gr.Tab("克隆合成"):
        gr.Markdown(
            "### 上传参考音频，克隆音色合成新语音\n"
            "建议：3-15 秒，清晰无噪声；提供对应文字可显著提升克隆质量\n"
            "也可从角色库选择已有的克隆角色，直接使用其参考音频"
        )
        with gr.Row():
            with gr.Column():
                def _clone_char_choices():
                    chars = load_characters()
                    return ["（不使用角色）"] + [
                        name for name, cfg in chars.items()
                        if cfg.get("voice_type") == "clone"
                    ]

                clone_char_dd = gr.Dropdown(
                    choices=_clone_char_choices(),
                    value="（不使用角色）",
                    label="从角色库选择（仅克隆类型）",
                )
                ref_audio_in = gr.Audio(
                    label="参考音频（WAV / MP3）",
                    type="filepath",
                    sources=["upload"],
                )
                ref_text_in = gr.Textbox(
                    label="参考音频对应文字（可选）",
                    placeholder="例：好的，没问题。",
                )
                recognize_btn = gr.Button("识别文字", variant="secondary")
                clone_text_in = gr.Textbox(
                    label="待合成文本",
                    placeholder="请输入要用克隆音色朗读的文字…",
                    lines=4,
                )
                clone_fmt = gr.Radio(choices=["WAV", "MP3"], value="WAV", label="输出格式")
                clone_btn = gr.Button("克隆合成", variant="primary")

            with gr.Column():
                clone_audio_out = gr.Audio(label="克隆合成结果", type="filepath", interactive=False)
                clone_status = gr.Textbox(label="状态", interactive=False)

                gr.Markdown("---\n**保存为角色音色**")
                save_char_name = gr.Textbox(label="角色名", placeholder="例：反派Boss")
                save_char_desc = gr.Textbox(label="描述（可选）")
                save_to_char_btn = gr.Button("保存角色", variant="secondary")
                save_char_status = gr.Textbox(label="保存状态", interactive=False)

        def on_char_select(char_name):
            """选择角色后自动填充参考音频路径和对应文字。"""
            if not char_name or char_name == "（不使用角色）":
                return gr.update(), gr.update(value="")
            chars = load_characters()
            cfg = chars.get(char_name, {})
            ref_path = cfg.get("ref_audio_path", "")
            ref_text = cfg.get("ref_text", "")
            if ref_path:
                abs_path = (BASE_DIR / ref_path) if not Path(ref_path).is_absolute() else Path(ref_path)
                if abs_path.exists():
                    return gr.update(value=str(abs_path)), gr.update(value=ref_text)
            return gr.update(), gr.update(value=ref_text)

        clone_char_dd.change(
            fn=on_char_select,
            inputs=[clone_char_dd],
            outputs=[ref_audio_in, ref_text_in],
        )

        def on_recognize(ref_audio, request: gr.Request):
            """使用 Whisper 识别参考音频中的文字。"""
            if ref_audio is None:
                return gr.update(), "请先上传参考音频"
            user = (request.username or "unknown") if request else "-"
            try:
                def do_recognize():
                    return asr_model.recognize(ref_audio)

                text = task_queue.submit(user, "asr", f"语音识别: {Path(ref_audio).name}", do_recognize)
                return gr.update(value=text), f"识别完成：{text[:50]}..."
            except Exception as e:
                return gr.update(), f"识别失败：{e}"

        recognize_btn.click(
            fn=on_recognize,
            inputs=[ref_audio_in],
            outputs=[ref_text_in, clone_status],
            concurrency_limit=10,
            trigger_mode="multiple",
        )

        def on_clone(char_sel, ref_audio, ref_t, text, fmt, request: gr.Request):
            if ref_audio is None:
                return None, "请先上传参考音频"
            if not text.strip():
                return None, "待合成文本不能为空"
            hint = char_sel if char_sel and char_sel != "（不使用角色）" else "clone"
            char = char_sel if char_sel and char_sel != "（不使用角色）" else ""
            return synth_to_file(text, fmt, ref_audio=ref_audio, ref_text=ref_t.strip() or None, name_hint=hint, char_name=char, request=request)

        def on_save_to_char(ref_audio, ref_t, char_name, char_desc, request: gr.Request):
            char_name = char_name.strip()
            if not char_name:
                return "角色名不能为空", gr.update()
            if ref_audio is None:
                return "请先上传参考音频", gr.update()
            user = (request.username or "unknown") if request else "-"
            src = Path(ref_audio)
            dest = REF_AUDIO_DIR / f"{char_name}{src.suffix}"
            shutil.copy2(src, dest)
            chars = load_characters()
            chars[char_name] = {
                "description": char_desc.strip(),
                "voice_type": "clone",
                "ref_audio_path": str(dest.relative_to(BASE_DIR)),
                "ref_text": ref_t.strip() if ref_t else "",
            }
            save_characters(chars)
            log_event(EVENT_CHAR_ADD, f"克隆角色={char_name} 参考音频={dest.name}", user=user)
            new_choices = ["（不使用角色）"] + list(chars.keys())
            return (
                f"已保存角色 [{char_name}]，参考音频：{dest.name}",
                gr.update(choices=new_choices),
            )

        clone_btn.click(
            fn=on_clone,
            inputs=[clone_char_dd, ref_audio_in, ref_text_in, clone_text_in, clone_fmt],
            outputs=[clone_audio_out, clone_status],
            concurrency_limit=10,
            trigger_mode="multiple",
        )
        save_to_char_btn.click(
            fn=on_save_to_char,
            inputs=[ref_audio_in, ref_text_in, save_char_name, save_char_desc],
            outputs=[save_char_status, clone_char_dd],
        )

    return clone_char_dd, save_to_char_btn, ref_audio_in
