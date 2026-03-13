"""Tab 6 — 工具"""

from __future__ import annotations

import datetime
import logging
import subprocess
import tempfile
from typing import Optional

import gradio as gr

from ui.common import FFMPEG_BIN
from core.app_logger import read_logs, ALL_EVENT_TYPES

logger = logging.getLogger(__name__)


def _parse_time(t: str) -> float:
    """解析 HH:MM:SS 或纯秒数字符串，返回浮点秒数。"""
    t = t.strip()
    if ":" in t:
        parts = [float(p) for p in t.split(":")]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
    return float(t)


def tab_tools():
    with gr.Tab("工具"):
        with gr.Tabs():
            # ---------------------------------------------------------------
            # 工具 1 — 提取视频音频
            # ---------------------------------------------------------------
            with gr.Tab("提取视频音频"):
                gr.Markdown(
                    "### 从视频中提取指定时间段的音频\n"
                    "> 不填时间则提取全段。"
                )
                with gr.Row():
                    with gr.Column():
                        tool_video_in = gr.Video(label="上传视频")
                        with gr.Row():
                            tool_start_in = gr.Textbox(
                                label="开始时间（可选）",
                                placeholder="如 00:01:30 或 90（秒），留空从头开始",
                            )
                            tool_end_in = gr.Textbox(
                                label="结束时间（可选）",
                                placeholder="如 00:02:00 或 120（秒），留空到结尾",
                            )
                        tool_fmt = gr.Radio(choices=["WAV", "MP3"], value="WAV", label="输出格式")
                        tool_extract_btn = gr.Button("开始提取", variant="primary")

                    with gr.Column():
                        tool_audio_out = gr.Audio(label="提取结果", type="filepath", interactive=False)
                        tool_status_out = gr.Textbox(label="状态", interactive=False)
                        tool_send_to_clone_btn = gr.Button("发送到模仿音频设计")

                def on_extract_audio(video, start_t, end_t, fmt):
                    if video is None:
                        return None, "请先上传视频文件"

                    start_sec: Optional[float] = None
                    end_sec: Optional[float] = None
                    try:
                        if start_t and start_t.strip():
                            start_sec = _parse_time(start_t)
                        if end_t and end_t.strip():
                            end_sec = _parse_time(end_t)
                    except ValueError:
                        return None, "时间格式错误，请使用 HH:MM:SS 或纯秒数（如 90）"

                    if start_sec is not None and end_sec is not None and end_sec <= start_sec:
                        return None, "结束时间必须大于开始时间"

                    video_path = video if isinstance(video, str) else video.get("name", "")
                    suffix = ".wav" if fmt == "WAV" else ".mp3"
                    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    out_tmp.close()
                    output_path = out_tmp.name

                    try:
                        cmd = [str(FFMPEG_BIN), "-y", "-i", video_path]
                        if start_sec is not None:
                            cmd += ["-ss", str(start_sec)]
                        if end_sec is not None:
                            cmd += ["-to", str(end_sec)]
                        cmd += ["-vn"]
                        if fmt == "WAV":
                            cmd += ["-acodec", "pcm_s16le"]
                        else:
                            cmd += ["-acodec", "libmp3lame", "-q:a", "2"]
                        cmd.append(output_path)

                        result = subprocess.run(
                            cmd, capture_output=True, text=True, timeout=300
                        )
                        if result.returncode != 0:
                            return None, f"ffmpeg 错误：\n{result.stderr[-800:]}"

                        s_label = f"{start_sec:.2f}s" if start_sec is not None else "开头"
                        e_label = f"{end_sec:.2f}s" if end_sec is not None else "结尾"
                        return output_path, f"提取成功！{s_label} ~ {e_label}"

                    except FileNotFoundError:
                        return None, f"未找到 ffmpeg：{FFMPEG_BIN}"
                    except subprocess.TimeoutExpired:
                        return None, "处理超时（超过 5 分钟）"
                    except Exception as e:
                        logger.exception("提取音频失败")
                        return None, f"处理失败：{e}"

                tool_extract_btn.click(
                    fn=on_extract_audio,
                    inputs=[tool_video_in, tool_start_in, tool_end_in, tool_fmt],
                    outputs=[tool_audio_out, tool_status_out],
                )

            # ---------------------------------------------------------------
            # 工具 2 — 日志查看
            # ---------------------------------------------------------------
            with gr.Tab("日志查看"):
                gr.Markdown("### 查看系统操作日志")
                with gr.Row():
                    log_date = gr.Textbox(
                        label="日期（YYYY-MM-DD）",
                        value=datetime.date.today().isoformat(),
                        placeholder="如 2026-03-14",
                    )
                    log_type = gr.Dropdown(
                        choices=["全部"] + ALL_EVENT_TYPES,
                        value="全部",
                        label="事件类型",
                    )
                    log_refresh_btn = gr.Button("查询", variant="primary")

                log_output = gr.Textbox(
                    label="日志内容",
                    interactive=False,
                    lines=20,
                    max_lines=30,
                )

                def on_query_logs(date_str, event_type):
                    et = event_type if event_type != "全部" else None
                    return read_logs(date=date_str.strip() or None, event_type=et)

                log_refresh_btn.click(
                    fn=on_query_logs,
                    inputs=[log_date, log_type],
                    outputs=[log_output],
                )

    return tool_audio_out, tool_send_to_clone_btn
