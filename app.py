"""
app.py — DPAudio 游戏语音合成工具
Gradio 前端，共 4 个 Tab：
  1. 单条合成
  2. 角色音色管理
  3. 批量处理
  4. 参考音频克隆
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np

from core.audio_utils import AudioFormat, audio_to_bytes, normalize_audio, save_audio
from core.batch_processor import BatchProcessor, load_table
from core.tts_engine import TTSEngine, PRESET_VOICES

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 全局路径
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REF_AUDIO_DIR = DATA_DIR / "reference_audio"
OUTPUT_DIR = BASE_DIR / "output"
CHARACTERS_FILE = DATA_DIR / "characters.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
REF_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 全局引擎（单例，延迟加载）
# ---------------------------------------------------------------------------
engine = TTSEngine()
batch_processor = BatchProcessor(engine)

# ---------------------------------------------------------------------------
# 角色数据持久化
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
        rows.append([name, vtype, detail, cfg.get("description", "")])
    return rows


# ---------------------------------------------------------------------------
# 工具：临时 WAV 文件（供 Gradio Audio 组件播放）
# ---------------------------------------------------------------------------

def bytes_to_gradio_audio(audio_bytes: bytes, fmt: str) -> str:
    suffix = f".{fmt}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name


# ===========================================================================
# Tab 1 — 单条合成
# ===========================================================================

def tab_single_synth(characters: dict):
    voice_choices = ["（默认）"] + PRESET_VOICES + list(characters.keys())

    with gr.Tab("单条合成"):
        gr.Markdown("### 输入文本，选择音色，生成语音")
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="合成文本",
                    placeholder="请输入要合成的文字…",
                    lines=5,
                )
                voice_dropdown = gr.Dropdown(
                    choices=voice_choices,
                    value="（默认）",
                    label="音色选择",
                )
                fmt_radio = gr.Radio(
                    choices=["WAV", "MP3"],
                    value="WAV",
                    label="输出格式",
                )
                synth_btn = gr.Button("生成语音", variant="primary")
            with gr.Column(scale=2):
                audio_out = gr.Audio(label="合成结果", type="filepath")
                status_text = gr.Textbox(label="状态", interactive=False)

        def on_synth(text, voice, fmt):
            if not text.strip():
                return None, "文本不能为空"
            try:
                chars = load_characters()
                voice_name: Optional[str] = None
                ref_audio = None
                ref_text: Optional[str] = None

                if voice and voice != "（默认）":
                    if voice in PRESET_VOICES:
                        voice_name = voice
                    elif voice in chars:
                        cfg = chars[voice]
                        if cfg.get("voice_type") == "preset":
                            voice_name = cfg.get("voice_name")
                        else:
                            rp = cfg.get("ref_audio_path", "")
                            if rp and Path(rp).exists():
                                ref_audio = rp
                                ref_text = cfg.get("ref_text")

                audio, sr = engine.synthesize(
                    text=text, voice_name=voice_name,
                    ref_audio=ref_audio, ref_text=ref_text,
                )
                audio = normalize_audio(audio)
                out_fmt: AudioFormat = "mp3" if fmt == "MP3" else "wav"
                audio_bytes = audio_to_bytes(audio, sr, out_fmt)
                tmp_path = bytes_to_gradio_audio(audio_bytes, out_fmt)
                return tmp_path, f"合成成功（{len(audio)/sr:.1f} 秒）"
            except Exception as e:
                logger.exception("单条合成失败")
                return None, f"合成失败：{e}"

        synth_btn.click(
            fn=on_synth,
            inputs=[text_input, voice_dropdown, fmt_radio],
            outputs=[audio_out, status_text],
        )

    return voice_dropdown  # 返回供外部刷新


# ===========================================================================
# Tab 2 — 角色音色管理
# ===========================================================================

def tab_character_manager():
    with gr.Tab("角色音色管理"):
        gr.Markdown("### 管理游戏角色与对应音色配置")

        char_table = gr.Dataframe(
            headers=["角色名", "音色类型", "音色详情", "描述"],
            datatype=["str", "str", "str", "str"],
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
                label="参考音频文件路径（data/reference_audio/ 下）",
                placeholder="例：data/reference_audio/hero.wav",
                visible=False,
            )
            ref_text_in = gr.Textbox(
                label="参考音频对应文字（可选，提升克隆质量）",
                visible=False,
            )
            save_char_btn = gr.Button("保存角色", variant="primary")
            delete_char_btn = gr.Button("删除角色（按角色名）", variant="stop")
            char_op_status = gr.Textbox(label="操作状态", interactive=False)

        # 动态显隐
        def toggle_voice_type(vtype):
            is_preset = vtype == "预设音色"
            return (
                gr.update(visible=is_preset),
                gr.update(visible=not is_preset),
                gr.update(visible=not is_preset),
            )

        voice_type_radio.change(
            fn=toggle_voice_type,
            inputs=[voice_type_radio],
            outputs=[preset_voice_dd, ref_audio_path_in, ref_text_in],
        )

        def save_char(name, desc, vtype, preset_v, ref_path, ref_t):
            name = name.strip()
            if not name:
                return character_display_rows(load_characters()), "角色名不能为空"
            chars = load_characters()
            if vtype == "预设音色":
                chars[name] = {
                    "description": desc,
                    "voice_type": "preset",
                    "voice_name": preset_v,
                }
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

        def refresh_table():
            return character_display_rows(load_characters())

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
        refresh_btn.click(fn=refresh_table, outputs=[char_table])


# ===========================================================================
# Tab 3 — 批量处理
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
                preview_btn = gr.Button("预览表头")
                preview_out = gr.Textbox(
                    label="列名预览", interactive=False, lines=3
                )

            with gr.Column():
                text_col_in = gr.Textbox(label="文本列名", value="text")
                char_col_in = gr.Textbox(label="角色列名（可选）", value="character")
                fname_col_in = gr.Textbox(label="文件名列名（可选）", value="filename")
                out_dir_in = gr.Textbox(
                    label="输出目录", value=str(OUTPUT_DIR)
                )
                batch_fmt_radio = gr.Radio(
                    choices=["WAV", "MP3"], value="WAV", label="输出格式"
                )
                batch_btn = gr.Button("开始批量合成", variant="primary")

        batch_log = gr.Textbox(
            label="处理日志", lines=15, interactive=False, max_lines=30
        )

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
            inputs=[file_upload, text_col_in, char_col_in,
                    fname_col_in, out_dir_in, batch_fmt_radio],
            outputs=[batch_log],
        )


# ===========================================================================
# Tab 4 — 参考音频克隆
# ===========================================================================

def tab_clone():
    with gr.Tab("参考音频克隆"):
        gr.Markdown(
            "### 上传参考音频，克隆音色合成新语音\n"
            "参考音频建议：3-15 秒，清晰无噪声，提供对应文字可显著提升质量"
        )

        with gr.Row():
            with gr.Column():
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
                clone_fmt_radio = gr.Radio(
                    choices=["WAV", "MP3"], value="WAV", label="输出格式"
                )
                clone_btn = gr.Button("克隆合成", variant="primary")

            with gr.Column():
                clone_audio_out = gr.Audio(label="克隆合成结果", type="filepath")
                clone_status = gr.Textbox(label="状态", interactive=False)

                gr.Markdown("---")
                gr.Markdown("**保存为角色音色**")
                save_char_name = gr.Textbox(
                    label="角色名", placeholder="例：反派Boss"
                )
                save_char_desc = gr.Textbox(label="描述（可选）")
                save_to_char_btn = gr.Button("保存到角色管理")
                save_char_status = gr.Textbox(label="保存状态", interactive=False)

        # 克隆合成
        def on_clone(ref_audio, ref_t, text, fmt):
            if ref_audio is None:
                return None, "请先上传参考音频"
            if not text.strip():
                return None, "待合成文本不能为空"
            try:
                audio, sr = engine.synthesize(
                    text=text,
                    ref_audio=ref_audio,
                    ref_text=ref_t.strip() or None,
                )
                audio = normalize_audio(audio)
                out_fmt: AudioFormat = "mp3" if fmt == "MP3" else "wav"
                audio_bytes = audio_to_bytes(audio, sr, out_fmt)
                tmp_path = bytes_to_gradio_audio(audio_bytes, out_fmt)
                return tmp_path, f"克隆合成成功（{len(audio)/sr:.1f} 秒）"
            except Exception as e:
                logger.exception("克隆合成失败")
                return None, f"克隆失败：{e}"

        # 保存参考音频到 data/reference_audio/ 并写入角色配置
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
            inputs=[ref_audio_in, ref_text_in, clone_text_in, clone_fmt_radio],
            outputs=[clone_audio_out, clone_status],
        )
        save_to_char_btn.click(
            fn=on_save_to_char,
            inputs=[ref_audio_in, ref_text_in, save_char_name, save_char_desc],
            outputs=[save_char_status],
        )


# ===========================================================================
# 主应用
# ===========================================================================

def build_app() -> gr.Blocks:
    with gr.Blocks(title="DPAudio — 游戏语音合成工具") as demo:
        gr.Markdown(
            "# DPAudio — 游戏语音合成工具\n"
            "基于 **Qwen3-TTS** 本地模型 · 支持预设音色 / 参考克隆 / 批量处理"
        )

        characters = load_characters()
        tab_single_synth(characters)
        tab_character_manager()
        tab_batch()
        tab_clone()

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
