"""Tab 5 — 批量处理"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from core.audio_utils import AudioFormat
from core.batch_processor import load_table
from ui.common import (
    OUTPUT_DIR,
    batch_processor,
    load_characters,
    load_design_characters,
)

_TEMPLATE_COLUMNS = ["character", "text", "filename"]


def _generate_template() -> str:
    """生成 Excel 模板文件，返回路径。"""
    df = pd.DataFrame(columns=_TEMPLATE_COLUMNS)
    path = Path(tempfile.gettempdir()) / "批量合成模板.xlsx"
    df.to_excel(path, index=False)
    return str(path)


def tab_batch():
    with gr.Tab("批量处理"):
        gr.Markdown(
            "### 上传 Excel 批量合成游戏台词\n"
            "Excel 固定三列：`character`（角色名，必填）、`text`（台词文本，必填）、`filename`（输出文件名，可选）\n\n"
            "角色名需与已保存的角色完全匹配（支持克隆角色和设计角色）。"
        )
        with gr.Row():
            with gr.Column():
                template_btn = gr.Button("下载 Excel 模板")
                template_file = gr.File(label="模板文件", interactive=False, visible=False)
                file_upload = gr.File(
                    label="上传 CSV / Excel 文件",
                    file_types=[".csv", ".xlsx", ".xls"],
                )
            with gr.Column():
                out_dir_in = gr.Textbox(label="输出目录", value=str(OUTPUT_DIR))
                batch_fmt = gr.Radio(choices=["WAV", "MP3"], value="WAV", label="输出格式")
                batch_btn = gr.Button("开始批量合成", variant="primary")

        batch_log = gr.Textbox(label="处理日志", lines=15, interactive=False, max_lines=30)

        def download_template():
            path = _generate_template()
            return gr.update(value=path, visible=True)

        def run_batch(file, out_dir, fmt):
            if file is None:
                return "请先上传文件"
            try:
                df = load_table(file.name)
            except Exception as e:
                return f"文件读取失败：{e}"

            # 检查必要列
            missing = [c for c in ["character", "text"] if c not in df.columns]
            if missing:
                return f"文件缺少必要列：{', '.join(missing)}。需要的列：character, text, filename（可选）"

            chars = load_characters()
            design_chars = load_design_characters()
            log_lines: list[str] = []

            def progress_cb(cur, total, msg):
                log_lines.append(msg)

            out_fmt: AudioFormat = "mp3" if fmt == "MP3" else "wav"
            result = batch_processor.run(
                df=df,
                output_dir=out_dir or str(OUTPUT_DIR),
                output_format=out_fmt,
                characters=chars,
                design_characters=design_chars,
                progress_cb=progress_cb,
            )
            log_lines.append(result.summary())
            return "\n".join(log_lines)

        template_btn.click(fn=download_template, inputs=[], outputs=[template_file])
        batch_btn.click(
            fn=run_batch,
            inputs=[file_upload, out_dir_in, batch_fmt],
            outputs=[batch_log],
        )
