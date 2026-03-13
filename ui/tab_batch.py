"""Tab 5 — 批量处理"""

from __future__ import annotations

import gradio as gr

from core.audio_utils import AudioFormat
from core.batch_processor import load_table
from ui.common import (
    OUTPUT_DIR,
    batch_processor,
    load_characters,
)


def tab_batch():
    with gr.Tab("批量处理"):
        gr.Markdown(
            "### 上传 CSV / Excel，批量合成游戏台词\n"
            "CSV 格式示例：`text,character,filename`"
        )
        with gr.Row():
            with gr.Column():
                file_upload = gr.File(
                    label="上传 CSV / Excel 文件",
                    file_types=[".csv", ".xlsx", ".xls"],
                )
                preview_btn = gr.Button("预览列名")
                preview_out = gr.Textbox(label="列名预览", interactive=False, lines=2)
            with gr.Column():
                text_col_in = gr.Textbox(label="文本列名", value="text")
                char_col_in = gr.Textbox(label="角色列名（可选）", value="character")
                fname_col_in = gr.Textbox(label="文件名列名（可选）", value="filename")
                out_dir_in = gr.Textbox(label="输出目录", value=str(OUTPUT_DIR))
                batch_fmt = gr.Radio(choices=["WAV", "MP3"], value="WAV", label="输出格式")
                batch_btn = gr.Button("开始批量合成", variant="primary")

        batch_log = gr.Textbox(label="处理日志", lines=15, interactive=False, max_lines=30)

        def preview_cols(file):
            if file is None:
                return "请先上传文件"
            try:
                df = load_table(file.name)
                return "列名：" + "、".join(df.columns.tolist())
            except Exception as e:
                return f"读取失败：{e}"

        def run_batch(file, text_col, char_col, fname_col, out_dir, fmt):
            if file is None:
                return "请先上传文件"
            try:
                df = load_table(file.name)
            except Exception as e:
                return f"文件读取失败：{e}"

            chars = load_characters()
            log_lines: list[str] = []

            def progress_cb(cur, total, msg):
                log_lines.append(msg)

            out_fmt: AudioFormat = "mp3" if fmt == "MP3" else "wav"
            result = batch_processor.run(
                df=df,
                text_col=text_col or "text",
                output_dir=out_dir or str(OUTPUT_DIR),
                output_format=out_fmt,
                char_col=char_col.strip() or None,
                filename_col=fname_col.strip() or None,
                characters=chars,
                progress_cb=progress_cb,
            )
            log_lines.append(result.summary())
            return "\n".join(log_lines)

        preview_btn.click(fn=preview_cols, inputs=[file_upload], outputs=[preview_out])
        batch_btn.click(
            fn=run_batch,
            inputs=[file_upload, text_col_in, char_col_in, fname_col_in, out_dir_in, batch_fmt],
            outputs=[batch_log],
        )
