"""
app.py — DPAudio 游戏语音合成工具
Gradio 前端，共 5 个 Tab：
  1. 单条合成
  2. 音色设计（自然语言描述音色）
  3. 参考音频克隆
  4. 角色音色管理
  5. 批量处理
"""

from __future__ import annotations

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
        vtype = "预设音色" if cfg.get("voice_type") == "preset" else "参考克隆"
        detail = cfg.get("voice_name", "") or cfg.get("ref_audio_path", "")
        ref_text = cfg.get("ref_text", "") if cfg.get("voice_type") == "clone" else ""
        rows.append([name, vtype, detail, ref_text, cfg.get("description", "")])
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
# Tab 1 — 单条合成
# ===========================================================================

def tab_single_synth():
    voice_choices = ["（默认）"] + PRESET_VOICES

    with gr.Tab("单条合成"):
        gr.Markdown("### 预设音色文本合成")
        with gr.Row():
            with gr.Column(scale=2):
                text_in = gr.Textbox(label="合成文本", placeholder="请输入要合成的文字…", lines=5)
                voice_dd = gr.Dropdown(choices=voice_choices, value="（默认）", label="音色选择")
                # 追加角色列表到下拉
                char_dd = gr.Dropdown(
                    choices=["（不使用角色）"] + list(load_characters().keys()),
                    value="（不使用角色）",
                    label="或从角色库选择",
                )
                fmt_radio = gr.Radio(choices=["WAV", "MP3"], value="WAV", label="输出格式")
                synth_btn = gr.Button("生成语音", variant="primary")
            with gr.Column(scale=2):
                audio_out = gr.Audio(label="合成结果", type="filepath")
                status_out = gr.Textbox(label="状态", interactive=False)

        def on_synth(text, voice, char, fmt):
            if not text.strip():
                return None, "文本不能为空"
            chars = load_characters()
            voice_name: Optional[str] = None
            ref_audio = None
            ref_text: Optional[str] = None

            # 角色库优先
            if char and char != "（不使用角色）" and char in chars:
                cfg = chars[char]
                if cfg.get("voice_type") == "preset":
                    voice_name = cfg.get("voice_name")
                else:
                    rp = cfg.get("ref_audio_path", "")
                    if rp and Path(rp).exists():
                        ref_audio = rp
                        ref_text = cfg.get("ref_text")
            elif voice and voice != "（默认）":
                voice_name = voice

            hint = char if char and char != "（不使用角色）" else (voice if voice and voice != "（默认）" else "")
            return synth_to_file(text, fmt, voice_name, ref_audio, ref_text, name_hint=hint)

        synth_btn.click(
            fn=on_synth,
            inputs=[text_in, voice_dd, char_dd, fmt_radio],
            outputs=[audio_out, status_out],
        )

    return char_dd  # 供外部刷新


# ===========================================================================
# Tab 2 — 参考音频克隆
# ===========================================================================

def tab_clone():
    with gr.Tab("参考音频克隆"):
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
                clone_audio_out = gr.Audio(label="克隆合成结果", type="filepath")
                clone_status = gr.Textbox(label="状态", interactive=False)

                gr.Markdown("---\n**保存为角色音色**")
                save_char_name = gr.Textbox(label="角色名", placeholder="例：反派Boss")
                save_char_desc = gr.Textbox(label="描述（可选）")
                save_to_char_btn = gr.Button("保存到角色管理", variant="secondary")
                save_char_status = gr.Textbox(label="保存状态", interactive=False)

        def on_char_select(char_name):
            """选择角色后自动填充参考音频路径和对应文字。"""
            if not char_name or char_name == "（不使用角色）":
                return gr.update(), gr.update(value="")
            chars = load_characters()
            cfg = chars.get(char_name, {})
            ref_path = cfg.get("ref_audio_path", "")
            ref_text = cfg.get("ref_text", "")
            if ref_path and Path(ref_path).exists():
                return gr.update(value=ref_path), gr.update(value=ref_text)
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
                return "角色名不能为空"
            if ref_audio is None:
                return "请先上传参考音频"
            src = Path(ref_audio)
            dest = REF_AUDIO_DIR / f"{char_name}{src.suffix}"
            shutil.copy2(src, dest)
            chars = load_characters()
            chars[char_name] = {
                "description": char_desc.strip(),
                "voice_type": "clone",
                "ref_audio_path": str(dest),
                "ref_text": ref_t.strip() if ref_t else "",
            }
            save_characters(chars)
            return f"已保存角色 [{char_name}]，参考音频：{dest.name}"

        clone_btn.click(
            fn=on_clone,
            inputs=[clone_char_dd, ref_audio_in, ref_text_in, clone_text_in, clone_fmt],
            outputs=[clone_audio_out, clone_status],
        )
        save_to_char_btn.click(
            fn=on_save_to_char,
            inputs=[ref_audio_in, ref_text_in, save_char_name, save_char_desc],
            outputs=[save_char_status],
        )


# ===========================================================================
# Tab 4 — 角色音色管理
# ===========================================================================

def tab_character_manager():
    with gr.Tab("角色音色管理"):
        gr.Markdown("### 管理游戏角色与对应音色配置")

        char_table = gr.Dataframe(
            headers=["角色名", "音色类型", "音色详情", "参考文字", "描述"],
            datatype=["str", "str", "str", "str", "str"],
            label="角色列表",
            interactive=False,
            value=character_display_rows(load_characters()),
        )
        refresh_btn = gr.Button("刷新列表")

        with gr.Accordion("新增 / 编辑角色", open=False):
            char_name_in = gr.Textbox(label="角色名（唯一标识）", placeholder="例：勇者")
            char_desc_in = gr.Textbox(label="描述（可选）", placeholder="例：男主角，沉稳低沉")
            voice_type_radio = gr.Radio(
                choices=["预设音色", "参考克隆"],
                value="预设音色",
                label="音色类型",
            )
            preset_voice_dd = gr.Dropdown(
                choices=PRESET_VOICES,
                value=PRESET_VOICES[0],
                label="预设音色",
                visible=True,
            )
            ref_audio_path_in = gr.Textbox(
                label="参考音频路径（data/reference_audio/ 下）",
                placeholder="例：data/reference_audio/hero.wav",
                visible=False,
            )
            ref_text_in = gr.Textbox(
                label="参考音频对应文字（可选）",
                visible=False,
            )
            save_char_btn = gr.Button("保存角色", variant="primary")
            delete_char_btn = gr.Button("删除角色（按角色名）", variant="stop")
            char_op_status = gr.Textbox(label="操作状态", interactive=False)

        def toggle_voice_type(vtype):
            is_preset = vtype == "预设音色"
            return (
                gr.update(visible=is_preset),
                gr.update(visible=not is_preset),
                gr.update(visible=not is_preset),
            )

        def save_char(name, desc, vtype, preset_v, ref_path, ref_t):
            name = name.strip()
            if not name:
                return character_display_rows(load_characters()), "角色名不能为空"
            chars = load_characters()
            if vtype == "预设音色":
                chars[name] = {"description": desc, "voice_type": "preset", "voice_name": preset_v}
            else:
                chars[name] = {
                    "description": desc,
                    "voice_type": "clone",
                    "ref_audio_path": ref_path.strip(),
                    "ref_text": ref_t.strip(),
                }
            save_characters(chars)
            return character_display_rows(chars), f"已保存角色：{name}"

        def delete_char(name):
            name = name.strip()
            if not name:
                return character_display_rows(load_characters()), "请输入角色名"
            chars = load_characters()
            if name in chars:
                del chars[name]
                save_characters(chars)
                return character_display_rows(chars), f"已删除角色：{name}"
            return character_display_rows(chars), f"角色不存在：{name}"

        voice_type_radio.change(
            fn=toggle_voice_type,
            inputs=[voice_type_radio],
            outputs=[preset_voice_dd, ref_audio_path_in, ref_text_in],
        )
        save_char_btn.click(
            fn=save_char,
            inputs=[char_name_in, char_desc_in, voice_type_radio,
                    preset_voice_dd, ref_audio_path_in, ref_text_in],
            outputs=[char_table, char_op_status],
        )
        delete_char_btn.click(
            fn=delete_char,
            inputs=[char_name_in],
            outputs=[char_table, char_op_status],
        )
        refresh_btn.click(
            fn=lambda: character_display_rows(load_characters()),
            outputs=[char_table],
        )


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
                        tool_audio_out = gr.Audio(label="提取结果", type="filepath")
                        tool_status_out = gr.Textbox(label="状态", interactive=False)

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




def build_app() -> gr.Blocks:
    with gr.Blocks(title="DPAudio — 游戏语音合成工具") as demo:
        gr.Markdown(
            "# DPAudio — 游戏语音合成工具\n"
            "基于 **Qwen3-TTS** 本地模型 · 预设音色 / 参考克隆 / 角色管理 / 批量处理"
        )
        tab_single_synth()
        tab_clone()
        tab_character_manager()
        tab_batch()
        tab_tools()

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
