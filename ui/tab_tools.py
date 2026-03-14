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
            # 工具 2 — 人声分离
            # ---------------------------------------------------------------
            with gr.Tab("人声分离"):
                gr.Markdown(
                    "### 使用 Demucs 分离人声与伴奏\n"
                    "> 上传音频文件，自动分离出人声和伴奏。分离后的人声可发送到模仿音频设计进行音色克隆。"
                )
                with gr.Row():
                    with gr.Column():
                        sep_audio_in = gr.Audio(label="上传音频", type="filepath")
                        sep_model = gr.Dropdown(
                            choices=["htdemucs", "htdemucs_ft", "mdx_extra"],
                            value="htdemucs",
                            label="分离模型",
                            info="htdemucs: 速度快；htdemucs_ft: 质量更高；mdx_extra: 适合音乐",
                        )
                        sep_btn = gr.Button("开始分离", variant="primary")

                    with gr.Column():
                        sep_vocals_out = gr.Audio(label="人声", type="filepath", interactive=False)
                        sep_accomp_out = gr.Audio(label="伴奏", type="filepath", interactive=False)
                        sep_status_out = gr.Textbox(label="状态", interactive=False)
                        sep_send_to_clone_btn = gr.Button("发送到模仿音频设计")

                def on_separate_vocals(audio_path, model_name):
                    if audio_path is None:
                        return None, None, "请先上传音频文件"

                    try:
                        import torch
                        import torchaudio
                        from demucs.pretrained import get_model
                        from demucs.apply import apply_model

                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        model = get_model(model_name)
                        model.to(device)
                        sample_rate = model.samplerate

                        wav, sr = torchaudio.load(audio_path)
                        if sr != sample_rate:
                            wav = torchaudio.functional.resample(wav, sr, sample_rate)
                        # apply_model 期望 (batch, channels, samples)
                        wav = wav.unsqueeze(0).to(device)
                        sources = apply_model(model, wav)  # (batch, n_sources, channels, samples)
                        sources = sources[0]  # (n_sources, channels, samples)

                        source_names = model.sources  # e.g. ['drums', 'bass', 'other', 'vocals']
                        vocals_idx = source_names.index("vocals")
                        vocals = sources[vocals_idx].cpu().numpy()
                        # 合并非人声轨道为伴奏
                        accomp_indices = [i for i, name in enumerate(source_names) if name != "vocals"]
                        accomp = sum(sources[i] for i in accomp_indices).cpu().numpy()

                        import soundfile as sf

                        vocals_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        vocals_tmp.close()
                        accomp_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        accomp_tmp.close()

                        # Demucs 输出形状为 (channels, samples)，需要转置为 (samples, channels)
                        sf.write(vocals_tmp.name, vocals.T, sample_rate)
                        sf.write(accomp_tmp.name, accomp.T, sample_rate)

                        return vocals_tmp.name, accomp_tmp.name, f"分离完成！模型: {model_name}"

                    except ImportError:
                        return None, None, "请先安装 demucs：pip install demucs"
                    except Exception as e:
                        logger.exception("人声分离失败")
                        return None, None, f"分离失败：{e}"

                sep_btn.click(
                    fn=on_separate_vocals,
                    inputs=[sep_audio_in, sep_model],
                    outputs=[sep_vocals_out, sep_accomp_out, sep_status_out],
                )

            # ---------------------------------------------------------------
            # 工具 3 — 日志查看
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

    return tool_audio_out, tool_send_to_clone_btn, sep_vocals_out, sep_send_to_clone_btn
