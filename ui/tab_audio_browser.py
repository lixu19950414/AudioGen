"""Tab 7 — 音频浏览"""

from __future__ import annotations

import datetime
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import gradio as gr

from ui.common import OUTPUT_DIR
from core.app_logger import log_event, EVENT_DOWNLOAD


def _scan_audio_files() -> list[list]:
    """递归扫描 OUTPUT_DIR 下的 .wav/.mp3 文件，按修改时间倒序返回表格行。"""
    exts = (".wav", ".mp3")
    files = [
        f for f in OUTPUT_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in exts
    ]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    rows = []
    for f in files:
        rel = f.relative_to(OUTPUT_DIR).as_posix()
        rows.append([False, rel])
    return rows


def _audio_detail(filepath: Path) -> str:
    """返回音频文件的详情文本。"""
    stat = filepath.stat()
    size_kb = stat.st_size / 1024
    size_str = f"{size_kb / 1024:.1f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"
    mtime_str = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"文件名：{filepath.name}\n"
        f"大小：{size_str}\n"
        f"修改时间：{mtime_str}\n"
        f"路径：{filepath}"
    )


def _batch_download(table_data, selected_only: bool = True, user: str = "-") -> Optional[str]:
    """将选中（或全部）音频文件打包为 ZIP 下载。"""
    if table_data is None or len(table_data) == 0:
        return None
    files = []
    for i in range(len(table_data)):
        if hasattr(table_data, "iloc"):
            checked = table_data.iloc[i, 0]
            rel = table_data.iloc[i, 1]
        else:
            checked = table_data[i][0]
            rel = table_data[i][1]
        if selected_only and not checked:
            continue
        fp = OUTPUT_DIR / rel
        if fp.exists():
            files.append((fp, rel))
    if not files:
        return None
    # 记录下载日志（所有文件名放在一条日志中）
    mode = "选中" if selected_only else "全部"
    names = ", ".join(rel for _, rel in files)
    log_event(EVENT_DOWNLOAD, f"下载文件({mode}) 文件数={len(files)} [{names}]", user=user)
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False, dir=str(OUTPUT_DIR))
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp, rel in files:
            zf.write(fp, rel)
    return tmp.name


def tab_audio_browser():
    with gr.Tab("音频浏览") as audio_browser_tab:
        gr.Markdown("### 浏览已生成的音频文件")
        with gr.Row():
            with gr.Column(scale=2):
                browser_table = gr.Dataframe(
                    headers=["选中", "文件名"],
                    datatype=["bool", "str"],
                    label="音频文件列表",
                    interactive=True,
                    value=_scan_audio_files(),
                    column_widths=["80px", "auto"],
                )
            with gr.Column(scale=1):
                browser_audio_out = gr.Audio(label="播放音频", type="filepath", interactive=False)
                browser_send_to_clone_btn = gr.Button("发送到模仿音频设计")
                browser_detail = gr.Textbox(label="音频详情", interactive=False, lines=4)
                with gr.Row():
                    refresh_btn = gr.Button("刷新列表")
                with gr.Row():
                    batch_download_selected_btn = gr.Button("下载选中文件", variant="primary")
                    batch_download_all_btn = gr.Button("下载全部")
                batch_download_file = gr.File(label="下载 ZIP")

        refresh_btn.click(fn=_scan_audio_files, inputs=[], outputs=[browser_table])
        audio_browser_tab.select(fn=_scan_audio_files, inputs=[], outputs=[browser_table])

        def on_batch_download_selected(table_data, request: gr.Request):
            user = (request.username or "unknown") if request else "-"
            zip_path = _batch_download(table_data, selected_only=True, user=user)
            return zip_path

        def on_batch_download_all(table_data, request: gr.Request):
            user = (request.username or "unknown") if request else "-"
            zip_path = _batch_download(table_data, selected_only=False, user=user)
            return zip_path

        batch_download_selected_btn.click(
            fn=on_batch_download_selected,
            inputs=[browser_table],
            outputs=[batch_download_file],
        )
        batch_download_all_btn.click(
            fn=on_batch_download_all,
            inputs=[browser_table],
            outputs=[batch_download_file],
        )

        def on_select(evt: gr.SelectData, table_data):
            row_idx = evt.index[0]
            if row_idx < 0 or row_idx >= len(table_data):
                return None, ""
            filename = table_data.iloc[row_idx, 1] if hasattr(table_data, "iloc") else table_data[row_idx][1]
            filepath = OUTPUT_DIR / filename
            if filepath.exists():
                return str(filepath), _audio_detail(filepath)
            return None, ""

        browser_table.select(
            fn=on_select,
            inputs=[browser_table],
            outputs=[browser_audio_out, browser_detail],
        )

    return browser_audio_out, browser_send_to_clone_btn
