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

_ALL_LABEL = "全部日期"


def _get_date_dirs() -> list[str]:
    """获取 OUTPUT_DIR 下的日期子目录名，倒序排列。"""
    if not OUTPUT_DIR.exists():
        return []
    dirs = sorted(
        [d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()],
        reverse=True,
    )
    return dirs


def _format_size(size_bytes: int) -> str:
    kb = size_bytes / 1024
    return f"{kb / 1024:.1f} MB" if kb >= 1024 else f"{kb:.1f} KB"


def _scan_audio_files(date_filter: str = _ALL_LABEL):
    """扫描音频文件，返回 (table_rows, checkbox_choices)。"""
    exts = (".wav", ".mp3")
    if not OUTPUT_DIR.exists():
        return [], []

    if date_filter and date_filter != _ALL_LABEL:
        search_dir = OUTPUT_DIR / date_filter
        if not search_dir.exists():
            return [], []
        files = [f for f in search_dir.rglob("*") if f.is_file() and f.suffix.lower() in exts]
    else:
        files = [f for f in OUTPUT_DIR.rglob("*") if f.is_file() and f.suffix.lower() in exts]

    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    rows = []
    choices = []
    for f in files:
        rel = f.relative_to(OUTPUT_DIR).as_posix()
        stat = f.stat()
        size_str = _format_size(stat.st_size)
        mtime_str = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        rows.append([rel, size_str, mtime_str])
        choices.append(rel)
    return rows, choices


def _audio_detail(filepath: Path) -> str:
    stat = filepath.stat()
    size_str = _format_size(stat.st_size)
    mtime_str = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"文件名：{filepath.name}\n"
        f"大小：{size_str}\n"
        f"修改时间：{mtime_str}\n"
        f"路径：{filepath}"
    )


def _batch_download(selected_files: list[str], user: str = "-") -> Optional[str]:
    """将选中的音频文件打包为 ZIP 下载。"""
    if not selected_files:
        return None
    files = []
    for rel in selected_files:
        fp = OUTPUT_DIR / rel
        if fp.exists():
            files.append((fp, rel))
    if not files:
        return None
    names = ", ".join(rel for _, rel in files)
    log_event(EVENT_DOWNLOAD, f"下载文件(选中) 文件数={len(files)} [{names}]", user=user)
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False, dir=str(OUTPUT_DIR))
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp, rel in files:
            zf.write(fp, rel)
    return tmp.name


def _download_all(date_filter: str, user: str = "-") -> Optional[str]:
    """下载当前筛选条件下的全部文件。"""
    _, choices = _scan_audio_files(date_filter)
    if not choices:
        return None
    return _batch_download(choices, user=user)


def tab_audio_browser():
    with gr.Tab("音频浏览") as audio_browser_tab:
        gr.Markdown("### 浏览已生成的音频文件")

        # --- 顶部：日期筛选 ---
        with gr.Row():
            date_dropdown = gr.Dropdown(
                label="按日期筛选",
                choices=[_ALL_LABEL] + _get_date_dirs(),
                value=_ALL_LABEL,
                interactive=True,
            )
            refresh_btn = gr.Button("刷新列表", scale=0)

        # --- 主体：左侧列表 + 右侧播放 ---
        init_rows, init_choices = _scan_audio_files()
        with gr.Row():
            with gr.Column(scale=2):
                browser_table = gr.Dataframe(
                    headers=["文件名", "大小", "时间"],
                    datatype=["str", "str", "str"],
                    label="点击行播放音频",
                    interactive=False,
                    value=init_rows,
                    column_widths=["auto", "100px", "180px"],
                )
            with gr.Column(scale=1):
                browser_audio_out = gr.Audio(label="播放音频", type="filepath", interactive=False)
                browser_send_to_clone_btn = gr.Button("发送到克隆人声")
                browser_detail = gr.Textbox(label="音频详情", interactive=False, lines=4)

        # --- 底部：批量选择与下载 ---
        with gr.Accordion("批量下载", open=False):
            file_checkbox = gr.CheckboxGroup(
                label="选择要下载的文件",
                choices=init_choices,
                value=[],
            )
            with gr.Row():
                select_all_btn = gr.Button("全选")
                deselect_all_btn = gr.Button("取消全选")
                batch_download_btn = gr.Button("下载选中文件", variant="primary")
                batch_download_all_btn = gr.Button("下载全部")
            batch_download_file = gr.File(label="下载 ZIP")

        # ===== 事件绑定 =====

        def on_refresh(date_filter):
            """刷新列表 + 更新日期下拉选项。"""
            dates = [_ALL_LABEL] + _get_date_dirs()
            if date_filter not in dates:
                date_filter = _ALL_LABEL
            rows, choices = _scan_audio_files(date_filter)
            return (
                gr.update(choices=dates, value=date_filter),
                rows,
                gr.update(choices=choices, value=[]),
            )

        refresh_btn.click(
            fn=on_refresh,
            inputs=[date_dropdown],
            outputs=[date_dropdown, browser_table, file_checkbox],
        )
        audio_browser_tab.select(
            fn=on_refresh,
            inputs=[date_dropdown],
            outputs=[date_dropdown, browser_table, file_checkbox],
        )

        def on_date_change(date_filter):
            rows, choices = _scan_audio_files(date_filter)
            return rows, gr.update(choices=choices, value=[])

        date_dropdown.change(
            fn=on_date_change,
            inputs=[date_dropdown],
            outputs=[browser_table, file_checkbox],
        )

        # 点击表格行 → 播放
        def on_select(evt: gr.SelectData, table_data):
            row_idx = evt.index[0]
            if row_idx < 0 or row_idx >= len(table_data):
                return None, ""
            filename = table_data.iloc[row_idx, 0] if hasattr(table_data, "iloc") else table_data[row_idx][0]
            filepath = OUTPUT_DIR / filename
            if filepath.exists():
                return str(filepath), _audio_detail(filepath)
            return None, ""

        browser_table.select(
            fn=on_select,
            inputs=[browser_table],
            outputs=[browser_audio_out, browser_detail],
        )

        # 全选 / 取消全选
        def on_select_all(date_filter):
            _, choices = _scan_audio_files(date_filter)
            return choices

        select_all_btn.click(
            fn=on_select_all,
            inputs=[date_dropdown],
            outputs=[file_checkbox],
        )
        deselect_all_btn.click(fn=lambda: [], outputs=[file_checkbox])

        # 下载
        def on_download_selected(selected, request: gr.Request):
            user = (request.username or "unknown") if request else "-"
            return _batch_download(selected, user=user)

        def on_download_all(date_filter, request: gr.Request):
            user = (request.username or "unknown") if request else "-"
            return _download_all(date_filter, user=user)

        batch_download_btn.click(
            fn=on_download_selected,
            inputs=[file_checkbox],
            outputs=[batch_download_file],
        )
        batch_download_all_btn.click(
            fn=on_download_all,
            inputs=[date_dropdown],
            outputs=[batch_download_file],
        )

    return browser_audio_out, browser_send_to_clone_btn
