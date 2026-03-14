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


def _scan_audio_files(keyword: str = "", fmt_filter: str = "全部") -> list[list]:
    """递归扫描 OUTPUT_DIR 下的 .wav/.mp3 文件，按修改时间倒序返回表格行。"""
    exts = (".wav", ".mp3")
    if fmt_filter == "WAV":
        exts = (".wav",)
    elif fmt_filter == "MP3":
        exts = (".mp3",)
    files = [
        f for f in OUTPUT_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in exts
    ]
    if keyword and keyword.strip():
        kw = keyword.strip().lower()
        files = [f for f in files if kw in f.name.lower()]
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


def _batch_download(table_data, selected_only: bool = True) -> Optional[str]:
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
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False, dir=str(OUTPUT_DIR))
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp, rel in files:
            zf.write(fp, rel)
    return tmp.name


def tab_audio_browser():
    with gr.Tab("音频浏览"):
        gr.Markdown("### 浏览已生成的音频文件")
        with gr.Row():
            with gr.Column(scale=1):
                browser_table = gr.Dataframe(
                    headers=["选中", "文件名"],
                    datatype=["bool", "str"],
                    label="音频文件列表",
                    interactive=True,
                    value=_scan_audio_files(),
                    column_widths=["60px", "auto"],
                )
            with gr.Column(scale=1):
                with gr.Row():
                    filter_keyword = gr.Textbox(
                        label="文件名搜索",
                        placeholder="输入关键词过滤…",
                        scale=3,
                    )
                    filter_fmt = gr.Radio(
                        choices=["全部", "WAV", "MP3"],
                        value="全部",
                        label="格式",
                        scale=1,
                    )
                with gr.Row():
                    refresh_btn = gr.Button("刷新列表")
                    select_all_btn = gr.Button("全选")
                    deselect_all_btn = gr.Button("取消全选")
                with gr.Row():
                    batch_download_selected_btn = gr.Button("下载选中文件", variant="primary")
                    batch_download_all_btn = gr.Button("下载全部")
                batch_download_file = gr.File(label="下载 ZIP", visible=False)
                browser_audio_out = gr.Audio(label="播放音频", type="filepath", interactive=False)
                browser_send_to_clone_btn = gr.Button("发送到模仿音频设计")
                browser_detail = gr.Textbox(label="音频详情", interactive=False, lines=4)

        refresh_btn.click(fn=_scan_audio_files, inputs=[filter_keyword, filter_fmt], outputs=[browser_table])
        filter_keyword.change(fn=_scan_audio_files, inputs=[filter_keyword, filter_fmt], outputs=[browser_table])
        filter_fmt.change(fn=_scan_audio_files, inputs=[filter_keyword, filter_fmt], outputs=[browser_table])

        def _toggle_all(table_data, checked: bool):
            if table_data is None or len(table_data) == 0:
                return table_data
            rows = []
            for i in range(len(table_data)):
                if hasattr(table_data, "iloc"):
                    rel = table_data.iloc[i, 1]
                else:
                    rel = table_data[i][1]
                rows.append([checked, rel])
            return rows

        select_all_btn.click(
            fn=lambda td: _toggle_all(td, True),
            inputs=[browser_table],
            outputs=[browser_table],
        )
        deselect_all_btn.click(
            fn=lambda td: _toggle_all(td, False),
            inputs=[browser_table],
            outputs=[browser_table],
        )

        def on_batch_download_selected(table_data):
            zip_path = _batch_download(table_data, selected_only=True)
            if zip_path:
                count = sum(
                    1 for i in range(len(table_data))
                    if (table_data.iloc[i, 0] if hasattr(table_data, "iloc") else table_data[i][0])
                )
                log_event(EVENT_DOWNLOAD, f"下载选中文件 文件数={count}")
                return gr.update(value=zip_path, visible=True)
            return gr.update(value=None, visible=False)

        def on_batch_download_all(table_data):
            zip_path = _batch_download(table_data, selected_only=False)
            if zip_path:
                count = len(table_data) if table_data is not None else 0
                log_event(EVENT_DOWNLOAD, f"批量下载全部 文件数={count}")
                return gr.update(value=zip_path, visible=True)
            return gr.update(value=None, visible=False)

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
