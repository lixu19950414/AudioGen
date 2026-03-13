"""
app.py — DPAudio 游戏语音合成工具
Gradio 前端，共 7 个 Tab：
  1. 预设音色合成
  2. 自定义音色设计（自然语言描述音色）
  3. 模仿音频设计
  4. 角色管理
  5. 批量处理
  6. 工具
  7. 音频浏览
"""

from __future__ import annotations

import config  # noqa: F401  确保 HF 环境变量尽早生效

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np

from core.audio_utils import AudioFormat, audio_to_bytes, normalize_audio, save_audio
from core.batch_processor import BatchProcessor, load_table
from core.tts_engine import TTSEngine, PRESET_VOICES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REF_AUDIO_DIR = DATA_DIR / "reference_audio"
OUTPUT_DIR = BASE_DIR / "output"
CHARACTERS_FILE = DATA_DIR / "characters.json"
DESIGN_CHARACTERS_FILE = DATA_DIR / "design_characters.json"

FFMPEG_BIN = BASE_DIR / "ffmpeg" / "ffmpeg.exe"

for _d in (DATA_DIR, REF_AUDIO_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 全局引擎（单例）
# ---------------------------------------------------------------------------
engine = TTSEngine()
batch_processor = BatchProcessor(engine)


# ---------------------------------------------------------------------------
# 角色数据
# ---------------------------------------------------------------------------

def load_characters() -> dict:
    if CHARACTERS_FILE.exists():
        return json.loads(CHARACTERS_FILE.read_text(encoding="utf-8"))
    return {}


def save_characters(characters: dict):
    CHARACTERS_FILE.write_text(
        json.dumps(characters, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def character_display_rows(characters: dict) -> list[list]:
    rows = []
    for name, cfg in characters.items():
        ref_path = cfg.get("ref_audio_path", "")
        ref_text = cfg.get("ref_text", "")
        rows.append([name, ref_path, ref_text, cfg.get("description", "")])
    return rows


# ---------------------------------------------------------------------------
# 设计角色数据（与克隆角色分开存储）
# ---------------------------------------------------------------------------

def load_design_characters() -> dict:
    if DESIGN_CHARACTERS_FILE.exists():
        return json.loads(DESIGN_CHARACTERS_FILE.read_text(encoding="utf-8"))
    return {}


def save_design_characters(characters: dict):
    DESIGN_CHARACTERS_FILE.write_text(
        json.dumps(characters, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def design_character_display_rows(characters: dict) -> list[list]:
    rows = []
    for name, cfg in characters.items():
        rows.append([name, cfg.get("instruct", ""), cfg.get("description", "")])
    return rows


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

import datetime


def to_gradio_audio(audio_bytes: bytes, fmt: str, name_hint: str = "") -> str:
    """写入 output 目录，返回路径供 Gradio Audio 组件播放。"""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if name_hint:
        # 清理文件名中不安全字符
        safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name_hint)
        filename = f"{safe}_{ts}.{fmt}"
    else:
        filename = f"synth_{ts}.{fmt}"
    path = OUTPUT_DIR / filename
    path.write_bytes(audio_bytes)
    return str(path)


def synth_to_file(
    text: str,
    fmt: str,
    voice_name: Optional[str] = None,
    ref_audio=None,
    ref_text: Optional[str] = None,
    name_hint: str = "",
) -> tuple[Optional[str], str]:
    """通用合成 → (audio_path | None, status)"""
    try:
        audio, sr = engine.synthesize(
            text=text,
            voice_name=voice_name,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        audio = normalize_audio(audio)
        out_fmt: AudioFormat = "mp3" if fmt == "MP3" else "wav"
        path = to_gradio_audio(audio_to_bytes(audio, sr, out_fmt), out_fmt, name_hint)
        return path, f"合成成功（{len(audio)/sr:.1f} 秒）\n已保存：{path}"
    except Exception as e:
        logger.exception("合成失败")
        return None, f"合成失败：{e}"


# ===========================================================================
# Tab 1 — 预设音色合成
# ===========================================================================

def tab_single_synth():
    voice_choices = ["（默认）"] + PRESET_VOICES

    with gr.Tab("预设音色合成"):
        gr.Markdown("### 预设音色文本合成")
        with gr.Row():
            with gr.Column(scale=2):
                text_in = gr.Textbox(label="合成文本", placeholder="请输入要合成的文字…", lines=5)
                voice_dd = gr.Dropdown(choices=voice_choices, value="（默认）", label="音色选择")
                fmt_radio = gr.Radio(choices=["WAV", "MP3"], value="WAV", label="输出格式")
                synth_btn = gr.Button("生成语音", variant="primary")
            with gr.Column(scale=2):
                audio_out = gr.Audio(label="合成结果", type="filepath", interactive=False)
                status_out = gr.Textbox(label="状态", interactive=False)
                send_to_clone_btn = gr.Button("发送到模仿音频设计")

        def on_synth(text, voice, fmt):
            if not text.strip():
                return None, "文本不能为空"
            voice_name = voice if voice and voice != "（默认）" else None
            hint = voice if voice and voice != "（默认）" else ""
            return synth_to_file(text, fmt, voice_name, name_hint=hint)

        synth_btn.click(
            fn=on_synth,
            inputs=[text_in, voice_dd, fmt_radio],
            outputs=[audio_out, status_out],
        )

    return audio_out, send_to_clone_btn

# ===========================================================================
# Tab 2 — 自定义音色设计
# ===========================================================================

def tab_voice_design():
    with gr.Tab("自定义音色设计"):
        gr.Markdown('通过自然语言描述生成音色，如"一个温柔的年轻女性声音，语速适中"')
        with gr.Row():
            with gr.Column(scale=1):
                # 从设计角色库选择
                def _design_char_choices():
                    return ["（不使用角色）"] + list(load_design_characters().keys())

                design_char_dd = gr.Dropdown(
                    choices=_design_char_choices(),
                    value="（不使用角色）",
                    label="从设计角色库选择",
                )
                design_instruct = gr.Textbox(
                    label="音色描述",
                    placeholder="例：一个低沉有磁性的中年男性声音",
                    lines=3,
                )
                design_text = gr.Textbox(
                    label="待合成文本",
                    placeholder="输入想要合成的文字...",
                    lines=5,
                )
                design_fmt = gr.Radio(choices=["WAV", "MP3"], value="WAV", label="输出格式")
                design_btn = gr.Button("生成语音", variant="primary")
            with gr.Column(scale=2):
                design_audio_out = gr.Audio(label="合成结果", type="filepath", interactive=False)
                design_status = gr.Textbox(label="状态", interactive=False)
                design_send_to_clone_btn = gr.Button("发送到模仿音频设计")

                gr.Markdown("---\n**保存为设计角色**")
                design_save_name = gr.Textbox(label="角色名", placeholder="例：温柔少女")
                design_save_desc = gr.Textbox(label="描述（可选）")
                design_save_btn = gr.Button("保存角色", variant="secondary")
                design_save_status = gr.Textbox(label="保存状态", interactive=False)

        def on_char_select(char_name):
            if not char_name or char_name == "（不使用角色）":
                return gr.update(value="")
            chars = load_design_characters()
            cfg = chars.get(char_name, {})
            return gr.update(value=cfg.get("instruct", ""))

        design_char_dd.change(
            fn=on_char_select,
            inputs=[design_char_dd],
            outputs=[design_instruct],
        )

        def on_voice_design(instruct, text, fmt):
            if not instruct.strip():
                return None, "请输入音色描述"
            if not text.strip():
                return None, "待合成文本不能为空"
            try:
                audio, sr = engine.voice_design(text=text, instruct=instruct)
                audio = normalize_audio(audio)
                out_fmt: AudioFormat = "mp3" if fmt == "MP3" else "wav"
                path = to_gradio_audio(audio_to_bytes(audio, sr, out_fmt), out_fmt, name_hint="design")
                return path, f"合成成功（{len(audio)/sr:.1f} 秒）\n已保存：{path}"
            except Exception as e:
                return None, f"合成失败：{e}"

        design_btn.click(
            fn=on_voice_design,
            inputs=[design_instruct, design_text, design_fmt],
            outputs=[design_audio_out, design_status],
        )

        def on_save_design_char(instruct, char_name, char_desc):
            char_name = char_name.strip()
            if not char_name:
                return "角色名不能为空", gr.update()
            if not instruct.strip():
                return "音色描述不能为空", gr.update()
            chars = load_design_characters()
            chars[char_name] = {
                "description": char_desc.strip(),
                "voice_type": "design",
                "instruct": instruct.strip(),
            }
            save_design_characters(chars)
            new_choices = ["（不使用角色）"] + list(chars.keys())
            return (
                f"已保存设计角色 [{char_name}]",
                gr.update(choices=new_choices),
            )

        design_save_btn.click(
            fn=on_save_design_char,
            inputs=[design_instruct, design_save_name, design_save_desc],
            outputs=[design_save_status, design_char_dd],
        )

    return design_char_dd, design_save_btn, design_audio_out, design_send_to_clone_btn


# ===========================================================================
# Tab 3 — 模仿音频设计
# ===========================================================================

def tab_clone():
    with gr.Tab("模仿音频设计"):
        gr.Markdown(
            "### 上传参考音频，克隆音色合成新语音\n"
            "建议：3-15 秒，清晰无噪声；提供对应文字可显著提升克隆质量\n"
            "也可从角色库选择已有的克隆角色，直接使用其参考音频"
        )
        with gr.Row():
            with gr.Column():
                # 从角色库选择已有克隆角色
                def _clone_char_choices():
                    chars = load_characters()
                    return ["（不使用角色）"] + [
                        name for name, cfg in chars.items()
                        if cfg.get("voice_type") == "clone"
                    ]

                clone_char_dd = gr.Dropdown(
                    choices=_clone_char_choices(),
                    value="（不使用角色）",
                    label="从角色库选择（仅克隆类型）",
                )
                ref_audio_in = gr.Audio(
                    label="参考音频（WAV / MP3）",
                    type="filepath",
                    sources=["upload"],
                )
                ref_text_in = gr.Textbox(
                    label="参考音频对应文字（可选）",
                    placeholder="例：好的，没问题。",
                )
                clone_text_in = gr.Textbox(
                    label="待合成文本",
                    placeholder="请输入要用克隆音色朗读的文字…",
                    lines=4,
                )
                clone_fmt = gr.Radio(choices=["WAV", "MP3"], value="WAV", label="输出格式")
                clone_btn = gr.Button("克隆合成", variant="primary")

            with gr.Column():
                clone_audio_out = gr.Audio(label="克隆合成结果", type="filepath", interactive=False)
                clone_status = gr.Textbox(label="状态", interactive=False)

                gr.Markdown("---\n**保存为角色音色**")
                save_char_name = gr.Textbox(label="角色名", placeholder="例：反派Boss")
                save_char_desc = gr.Textbox(label="描述（可选）")
                save_to_char_btn = gr.Button("保存角色", variant="secondary")
                save_char_status = gr.Textbox(label="保存状态", interactive=False)

        def on_char_select(char_name):
            """选择角色后自动填充参考音频路径和对应文字。"""
            if not char_name or char_name == "（不使用角色）":
                return gr.update(), gr.update(value="")
            chars = load_characters()
            cfg = chars.get(char_name, {})
            ref_path = cfg.get("ref_audio_path", "")
            ref_text = cfg.get("ref_text", "")
            if ref_path:
                abs_path = (BASE_DIR / ref_path) if not Path(ref_path).is_absolute() else Path(ref_path)
                if abs_path.exists():
                    return gr.update(value=str(abs_path)), gr.update(value=ref_text)
            return gr.update(), gr.update(value=ref_text)

        clone_char_dd.change(
            fn=on_char_select,
            inputs=[clone_char_dd],
            outputs=[ref_audio_in, ref_text_in],
        )

        def on_clone(char_sel, ref_audio, ref_t, text, fmt):
            if ref_audio is None:
                return None, "请先上传参考音频"
            if not text.strip():
                return None, "待合成文本不能为空"
            hint = char_sel if char_sel and char_sel != "（不使用角色）" else "clone"
            return synth_to_file(text, fmt, ref_audio=ref_audio, ref_text=ref_t.strip() or None, name_hint=hint)

        def on_save_to_char(ref_audio, ref_t, char_name, char_desc):
            char_name = char_name.strip()
            if not char_name:
                return "角色名不能为空", gr.update()
            if ref_audio is None:
                return "请先上传参考音频", gr.update()
            src = Path(ref_audio)
            dest = REF_AUDIO_DIR / f"{char_name}{src.suffix}"
            shutil.copy2(src, dest)
            chars = load_characters()
            chars[char_name] = {
                "description": char_desc.strip(),
                "voice_type": "clone",
                "ref_audio_path": str(dest.relative_to(BASE_DIR)),
                "ref_text": ref_t.strip() if ref_t else "",
            }
            save_characters(chars)
            new_choices = ["（不使用角色）"] + list(chars.keys())
            return (
                f"已保存角色 [{char_name}]，参考音频：{dest.name}",
                gr.update(choices=new_choices),
            )

        clone_btn.click(
            fn=on_clone,
            inputs=[clone_char_dd, ref_audio_in, ref_text_in, clone_text_in, clone_fmt],
            outputs=[clone_audio_out, clone_status],
        )
        save_to_char_btn.click(
            fn=on_save_to_char,
            inputs=[ref_audio_in, ref_text_in, save_char_name, save_char_desc],
            outputs=[save_char_status, clone_char_dd],
        )

    return clone_char_dd, save_to_char_btn, ref_audio_in


# ===========================================================================
# Tab 4 — 角色管理
# ===========================================================================

def tab_character_manager():
    with gr.Tab("角色管理"):
        with gr.Tabs():
            # --- 子 Tab 1：克隆角色管理 ---
            with gr.Tab("克隆角色"):
                char_table = gr.Dataframe(
                    headers=["角色名", "参考音频路径", "参考文字", "描述"],
                    datatype=["str", "str", "str", "str"],
                    label="克隆角色列表",
                    interactive=False,
                    value=character_display_rows(load_characters()),
                )
                with gr.Row():
                    refresh_btn = gr.Button("刷新")
                    delete_clone_name = gr.Textbox(label="角色名", placeholder="输入要删除的角色名", scale=2)
                    delete_clone_btn = gr.Button("删除", variant="stop")
                clone_op_status = gr.Textbox(label="操作状态", interactive=False)

            # --- 子 Tab 2：设计角色管理 ---
            with gr.Tab("设计角色"):
                design_table = gr.Dataframe(
                    headers=["角色名", "音色描述", "描述"],
                    datatype=["str", "str", "str"],
                    label="设计角色列表",
                    interactive=False,
                    value=design_character_display_rows(load_design_characters()),
                )
                with gr.Row():
                    design_refresh_btn = gr.Button("刷新")
                    delete_design_name = gr.Textbox(label="角色名", placeholder="输入要删除的角色名", scale=2)
                    delete_design_btn = gr.Button("删除", variant="stop")
                design_op_status = gr.Textbox(label="操作状态", interactive=False)

        def delete_clone_char(name):
            name = name.strip()
            if not name:
                return character_display_rows(load_characters()), "请输入角色名"
            chars = load_characters()
            if name in chars:
                del chars[name]
                save_characters(chars)
                return character_display_rows(chars), f"已删除克隆角色：{name}"
            return character_display_rows(chars), f"角色不存在：{name}"

        def delete_design_char(name):
            name = name.strip()
            if not name:
                return design_character_display_rows(load_design_characters()), "请输入角色名"
            chars = load_design_characters()
            if name in chars:
                del chars[name]
                save_design_characters(chars)
                return design_character_display_rows(chars), f"已删除设计角色：{name}"
            return design_character_display_rows(chars), f"角色不存在：{name}"

        refresh_btn.click(
            fn=lambda: character_display_rows(load_characters()),
            outputs=[char_table],
        )
        delete_clone_btn.click(
            fn=delete_clone_char,
            inputs=[delete_clone_name],
            outputs=[char_table, clone_op_status],
        )
        design_refresh_btn.click(
            fn=lambda: design_character_display_rows(load_design_characters()),
            outputs=[design_table],
        )
        delete_design_btn.click(
            fn=delete_design_char,
            inputs=[delete_design_name],
            outputs=[design_table, design_op_status],
        )

    return char_table, design_table, delete_clone_btn, delete_design_btn



# ===========================================================================
# Tab 5 — 批量处理
# ===========================================================================

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


# ===========================================================================
# Tab 6 — 工具
# ===========================================================================

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
                        cmd += ["-vn"]  # 不输出视频
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

    return tool_audio_out, tool_send_to_clone_btn


# ===========================================================================
# Tab 7 — 音频浏览
# ===========================================================================

def _scan_audio_files(keyword: str = "", fmt_filter: str = "全部") -> list[list]:
    """扫描 OUTPUT_DIR 下的 .wav/.mp3 文件，按修改时间倒序返回表格行。"""
    exts = (".wav", ".mp3")
    if fmt_filter == "WAV":
        exts = (".wav",)
    elif fmt_filter == "MP3":
        exts = (".mp3",)
    files = [
        f for f in OUTPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    ]
    if keyword and keyword.strip():
        kw = keyword.strip().lower()
        files = [f for f in files if kw in f.name.lower()]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    rows = []
    for f in files:
        mtime_str = datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        rows.append([f.name, mtime_str])
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


def tab_audio_browser():
    with gr.Tab("音频浏览"):
        gr.Markdown("### 浏览已生成的音频文件")
        with gr.Row():
            with gr.Column(scale=1):
                browser_table = gr.Dataframe(
                    headers=["文件名", "修改时间"],
                    datatype=["str", "str"],
                    label="音频文件列表",
                    interactive=False,
                    value=_scan_audio_files(),
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
                refresh_btn = gr.Button("刷新列表")
                browser_audio_out = gr.Audio(label="播放音频", type="filepath", interactive=False)
                browser_send_to_clone_btn = gr.Button("发送到模仿音频设计")
                browser_detail = gr.Textbox(label="音频详情", interactive=False, lines=4)

        refresh_btn.click(fn=_scan_audio_files, inputs=[filter_keyword, filter_fmt], outputs=[browser_table])
        filter_keyword.change(fn=_scan_audio_files, inputs=[filter_keyword, filter_fmt], outputs=[browser_table])
        filter_fmt.change(fn=_scan_audio_files, inputs=[filter_keyword, filter_fmt], outputs=[browser_table])

        def on_select(evt: gr.SelectData, table_data):
            row_idx = evt.index[0]
            if row_idx < 0 or row_idx >= len(table_data):
                return None, ""
            filename = table_data.iloc[row_idx, 0]
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


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AudioGen — 游戏语音合成工具") as demo:
        gr.Markdown(
            "# AudioGen — 游戏语音合成工具\n"
        )
        synth_audio_out, synth_send_btn = tab_single_synth()
        design_char_dd, design_save_btn, design_audio_out, design_send_btn = tab_voice_design()
        clone_char_dd, clone_save_btn, ref_audio_in = tab_clone()
        char_table, design_table, delete_clone_btn, delete_design_btn = tab_character_manager()
        tool_audio_out, tool_send_btn = tab_tools()
        browser_audio_out, browser_send_btn = tab_audio_browser()

        # 跨 Tab 自动刷新：保存角色 → 刷新管理表格
        clone_save_btn.click(
            fn=lambda: character_display_rows(load_characters()),
            outputs=[char_table],
        )
        design_save_btn.click(
            fn=lambda: design_character_display_rows(load_design_characters()),
            outputs=[design_table],
        )
        # 跨 Tab 自动刷新：删除角色 → 刷新合成 Tab 下拉
        delete_clone_btn.click(
            fn=lambda: gr.update(choices=["（不使用角色）"] + list(load_characters().keys())),
            outputs=[clone_char_dd],
        )
        delete_design_btn.click(
            fn=lambda: gr.update(choices=["（不使用角色）"] + list(load_design_characters().keys())),
            outputs=[design_char_dd],
        )

        # 发送到模仿音频设计
        synth_send_btn.click(fn=lambda a: a, inputs=[synth_audio_out], outputs=[ref_audio_in])
        design_send_btn.click(fn=lambda a: a, inputs=[design_audio_out], outputs=[ref_audio_in])
        tool_send_btn.click(fn=lambda a: a, inputs=[tool_audio_out], outputs=[ref_audio_in])
        browser_send_btn.click(fn=lambda a: a, inputs=[browser_audio_out], outputs=[ref_audio_in])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        auth=("admin", "123456"),
    )
