"""Tab — 批量音乐合成"""

from __future__ import annotations

import datetime
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import pandas as pd

from core.batch_processor import load_table
from core.app_logger import log_event, EVENT_BATCH
from core.music_model import DIT_MODELS, LM_MODELS, DEFAULT_DIT_MODEL, DEFAULT_LM_MODEL
from ui.common import (
    BATCH_OUTPUT_DIR,
    batch_processor,
    task_queue,
)

_MUSIC_TEMPLATE_COLUMNS = ["caption", "lyrics", "instrumental", "duration", "filename"]

VOCAL_LANGUAGES = [
    "unknown", "zh", "en", "ja", "ko", "fr", "de", "es", "pt", "ru",
    "it", "ar", "hi", "th", "vi", "id", "tr", "pl", "nl", "sv",
]


def _generate_music_template() -> str:
    """生成批量音乐合成 Excel 模板文件，返回路径。"""
    df = pd.DataFrame(columns=_MUSIC_TEMPLATE_COLUMNS)
    path = Path(tempfile.gettempdir()) / "批量音乐合成模板.xlsx"
    df.to_excel(path, index=False)
    return str(path)


def tab_batch_music():
    with gr.Tab("批量处理音乐"):
        gr.Markdown(
            "### 上传 Excel 批量合成音乐\n"
            "Excel 列：`caption`（音乐描述）、`lyrics`（歌词）— 至少填一个；"
            "`instrumental`（TRUE/FALSE，可选）、`duration`（时长秒数，可选）、`filename`（输出文件名，可选）"
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
                with gr.Row():
                    dit_model_dd = gr.Dropdown(
                        choices=[(v, k) for k, v in DIT_MODELS.items()],
                        value=DEFAULT_DIT_MODEL,
                        label="DiT 模型",
                    )
                    lm_model_dd = gr.Dropdown(
                        choices=[(v, k) for k, v in LM_MODELS.items()],
                        value=DEFAULT_LM_MODEL,
                        label="LM 模型",
                    )
                with gr.Row():
                    default_duration = gr.Slider(
                        minimum=10, maximum=600, value=30, step=5,
                        label="默认时长（秒）",
                    )
                    steps_slider = gr.Slider(
                        minimum=1, maximum=50, value=8, step=1,
                        label="推理步数",
                    )
                    guidance_slider = gr.Slider(
                        minimum=1, maximum=15, value=7, step=0.5,
                        label="引导系数",
                    )
                with gr.Row():
                    vocal_lang_dd = gr.Dropdown(
                        choices=VOCAL_LANGUAGES,
                        value="unknown",
                        label="默认人声语言",
                    )
                    instrumental_cb = gr.Checkbox(
                        label="默认纯器乐（无人声）", value=False,
                    )
                batch_btn = gr.Button("开始批量合成", variant="primary")

        batch_log = gr.Textbox(label="处理日志", lines=15, interactive=False, max_lines=30)
        batch_download = gr.File(label="下载压缩包", interactive=False)

        def download_template():
            path = _generate_music_template()
            return gr.update(value=path, visible=True)

        def run_batch_music(file, duration, steps, guidance, vocal_lang, instrumental, dit_model, lm_model, request: gr.Request):
            user = (request.username or "unknown") if request else "-"
            if file is None:
                return "请先上传文件", gr.update(value=None)
            try:
                df = load_table(file.name)
            except Exception as e:
                return f"文件读取失败：{e}", gr.update(value=None)

            # 检查必要列
            has_caption = "caption" in df.columns
            has_lyrics = "lyrics" in df.columns
            if not has_caption and not has_lyrics:
                return "文件缺少必要列：至少需要 caption 或 lyrics 列", gr.update(value=None)

            # 预检查
            errors: list[str] = []
            empty_rows: list[int] = []
            for idx, row in df.iterrows():
                row_num = int(idx) + 1
                caption = str(row.get("caption", "")).strip()
                lyrics = str(row.get("lyrics", "")).strip()
                if not caption and not lyrics:
                    empty_rows.append(row_num)

            if empty_rows:
                errors.append(f"以下行 caption 和 lyrics 均为空：第 {', '.join(map(str, empty_rows))} 行")

            if errors:
                return (
                    "文件预检查未通过，请修正后重试：\n\n" + "\n".join(errors),
                    gr.update(value=None),
                )

            # 创建输出目录
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_stem = Path(file.orig_name).stem if hasattr(file, "orig_name") else Path(file.name).stem
            safe_stem = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in file_stem)
            folder_name = f"{timestamp}_{safe_stem}_music"
            out_dir = BATCH_OUTPUT_DIR / folder_name
            out_dir.mkdir(parents=True, exist_ok=True)

            log_event(EVENT_BATCH, f"开始音乐批量 文件={file_stem} 总数={len(df)}", user=user)

            log_lines: list[str] = []

            def progress_cb(cur, total, msg):
                log_lines.append(msg)

            task_desc = f"批量音乐: {file_stem} ({len(df)} 条)"

            def do_batch():
                return batch_processor.run_music(
                    df=df,
                    output_dir=str(out_dir),
                    default_duration=duration,
                    inference_steps=int(steps),
                    guidance_scale=guidance,
                    vocal_language=vocal_lang,
                    default_instrumental=instrumental,
                    dit_config=dit_model,
                    lm_model=lm_model,
                    progress_cb=progress_cb,
                )

            result = task_queue.submit(user, "batch", task_desc, do_batch)
            log_lines.append(result.summary())
            log_event(EVENT_BATCH, f"完成音乐批量 文件={file_stem} 成功={result.success} 失败={result.failed} 总数={result.total}", user=user)

            # 压缩输出目录
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
            fn=run_batch_music,
            inputs=[file_upload, default_duration, steps_slider, guidance_slider, vocal_lang_dd, instrumental_cb, dit_model_dd, lm_model_dd],
            outputs=[batch_log, batch_download],
            concurrency_limit=10,
        )
