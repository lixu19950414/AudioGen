"""Tab 8 — 批量处理下载"""

from __future__ import annotations

import datetime
from pathlib import Path

import gradio as gr

from ui.common import BATCH_OUTPUT_DIR
from core.app_logger import log_event, EVENT_DOWNLOAD


def _scan_zip_files() -> list[list]:
    """扫描 BATCH_OUTPUT_DIR 下的 .zip 文件，按修改时间倒序返回表格行。"""
    files = [
        f for f in BATCH_OUTPUT_DIR.rglob("*.zip")
        if f.is_file()
    ]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    rows = []
    for f in files:
        stat = f.stat()
        size_kb = stat.st_size / 1024
        size_str = f"{size_kb / 1024:.1f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"
        mtime_str = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        rel = f.relative_to(BATCH_OUTPUT_DIR).as_posix()
        rows.append([rel, size_str, mtime_str])
    return rows


def tab_batch_download():
    with gr.Tab("批量处理下载"):
        gr.Markdown("### 下载批量处理生成的 ZIP 文件")

        batch_table = gr.Dataframe(
            headers=["文件名", "大小", "修改时间"],
            datatype=["str", "str", "str"],
            label="ZIP 文件列表",
            interactive=False,
            value=_scan_zip_files(),
            column_widths=["auto", "120px", "200px"],
        )

        refresh_btn = gr.Button("刷新列表")

        download_file = gr.File(label="下载文件")

        refresh_btn.click(fn=_scan_zip_files, inputs=[], outputs=[batch_table])

        def on_select_download(evt: gr.SelectData, table_data, request: gr.Request):
            row_idx = evt.index[0]
            if table_data is None or row_idx < 0 or row_idx >= len(table_data):
                return None
            filename = table_data.iloc[row_idx, 0] if hasattr(table_data, "iloc") else table_data[row_idx][0]
            filepath = BATCH_OUTPUT_DIR / filename
            if filepath.exists():
                user = (request.username or "unknown") if request else "-"
                log_event(EVENT_DOWNLOAD, f"下载批量文件 {filename}", user=user)
                return str(filepath)
            return None

        batch_table.select(
            fn=on_select_download,
            inputs=[batch_table],
            outputs=[download_file],
        )
