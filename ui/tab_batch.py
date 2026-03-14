"""Tab 5 — 批量处理"""

from __future__ import annotations

import datetime
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import pandas as pd

from core.audio_utils import AudioFormat
from core.batch_processor import load_table
from core.tts_engine import PRESET_VOICES
from core.app_logger import log_event, EVENT_BATCH
from ui.common import (
    BATCH_OUTPUT_DIR,
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
                batch_fmt = gr.Radio(choices=["WAV", "MP3"], value="WAV", label="输出格式")
                batch_btn = gr.Button("开始批量合成", variant="primary")

        batch_log = gr.Textbox(label="处理日志", lines=15, interactive=False, max_lines=30)
        batch_download = gr.File(label="下载压缩包", interactive=False)

        def download_template():
            path = _generate_template()
            return gr.update(value=path, visible=True)

        def run_batch(file, fmt, request: gr.Request):
            user = (request.username or "unknown") if request else "-"
            if file is None:
                return "请先上传文件", gr.update(value=None)
            try:
                df = load_table(file.name)
            except Exception as e:
                return f"文件读取失败：{e}", gr.update(value=None)

            # 检查必要列
            missing = [c for c in ["character", "text"] if c not in df.columns]
            if missing:
                return (
                    f"文件缺少必要列：{', '.join(missing)}。需要的列：character, text, filename（可选）",
                    gr.update(value=None),
                )

            chars = load_characters()
            design_chars = load_design_characters()

            # ---- 预检查：格式与角色可用性 ----
            errors: list[str] = []
            empty_text_rows: list[int] = []
            empty_char_rows: list[int] = []
            invalid_chars: dict[str, list[int]] = {}  # 角色名 → 出现行号

            for idx, row in df.iterrows():
                row_num = int(idx) + 1
                text = str(row.get("text", "")).strip()
                char_name = str(row.get("character", "")).strip()

                if not text:
                    empty_text_rows.append(row_num)
                if not char_name:
                    empty_char_rows.append(row_num)
                elif char_name not in chars and char_name not in design_chars and char_name not in PRESET_VOICES:
                    invalid_chars.setdefault(char_name, []).append(row_num)

            if empty_text_rows:
                errors.append(f"以下行文本为空：第 {', '.join(map(str, empty_text_rows))} 行")
            if empty_char_rows:
                errors.append(f"以下行角色为空：第 {', '.join(map(str, empty_char_rows))} 行")
            if invalid_chars:
                for name, rows in invalid_chars.items():
                    errors.append(f"角色「{name}」不存在（第 {', '.join(map(str, rows))} 行）")

            if errors:
                all_chars = sorted(set(list(chars.keys()) + list(design_chars.keys()) + PRESET_VOICES))
                return (
                    "文件预检查未通过，请修正后重试：\n\n" + "\n".join(errors) + "\n\n可用角色：" + "、".join(all_chars),
                    gr.update(value=None),
                )

            # ---- 预检查通过，创建时间戳+文件名子目录，开始合成 ----
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_stem = Path(file.orig_name).stem if hasattr(file, "orig_name") else Path(file.name).stem
            safe_stem = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in file_stem)
            folder_name = f"{timestamp}_{safe_stem}"
            out_dir = BATCH_OUTPUT_DIR / folder_name
            out_dir.mkdir(parents=True, exist_ok=True)

            log_event(EVENT_BATCH, f"开始 文件={file_stem} 总数={len(df)} 格式={fmt}", user=user)

            log_lines: list[str] = []

            def progress_cb(cur, total, msg):
                log_lines.append(msg)

            out_fmt: AudioFormat = "mp3" if fmt == "MP3" else "wav"
            result = batch_processor.run(
                df=df,
                output_dir=str(out_dir),
                output_format=out_fmt,
                characters=chars,
                design_characters=design_chars,
                progress_cb=progress_cb,
            )
            log_lines.append(result.summary())
            log_event(EVENT_BATCH, f"完成 文件={file_stem} 成功={result.success} 失败={result.failed} 总数={result.total}", user=user)

            # ---- 压缩输出目录 ----
            if result.success > 0:
                zip_path = BATCH_OUTPUT_DIR / f"{folder_name}.zip"
                with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in out_dir.iterdir():
                        if f.is_file():
                            zf.write(f, f.name)
                log_lines.append(f"\n压缩包已生成：{zip_path.name}")
                return "\n".join(log_lines), gr.update(value=str(zip_path))

            return "\n".join(log_lines), gr.update(value=None)

        template_btn.click(fn=download_template, inputs=[], outputs=[template_file])
        batch_btn.click(
            fn=run_batch,
            inputs=[file_upload, batch_fmt],
            outputs=[batch_log, batch_download],
        )
