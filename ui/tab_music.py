"""Tab — 音乐合成（ACE-Step 1.5）"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import gradio as gr

from core.app_logger import log_event, EVENT_SYNTHESIZE
from ui.common import task_queue, BASE_DIR, OUTPUT_DIR, music_model

logger = logging.getLogger(__name__)

# 人声语言选项
VOCAL_LANGUAGES = [
    "unknown", "zh", "en", "ja", "ko", "fr", "de", "es", "pt", "ru",
    "it", "ar", "hi", "th", "vi", "id", "tr", "pl", "nl", "sv",
]

# 调式选项
KEYSCALE_OPTIONS = [
    "", "C major", "C minor", "C# major", "C# minor",
    "D major", "D minor", "Eb major", "Eb minor",
    "E major", "E minor", "F major", "F minor",
    "F# major", "F# minor", "G major", "G minor",
    "Ab major", "Ab minor", "A major", "A minor",
    "Bb major", "Bb minor", "B major", "B minor",
]

# 拍号选项
TIME_SIGNATURE_OPTIONS = ["", "4/4", "3/4", "6/8", "2/4", "5/4", "7/8"]

# 重绘模式选项
REPAINT_MODES = ["balanced", "creative", "precise"]


def _save_music_output(result, name_hint: str = "music") -> str | None:
    """从 GenerationResult 中提取第一个音频，保存到 output/ 目录。"""
    if not result.success or not result.audios:
        return None

    audio_info = result.audios[0]
    audio_tensor = audio_info.get("tensor")
    if audio_tensor is None:
        return None

    import datetime
    import soundfile as sf
    import numpy as np

    sample_rate = audio_info.get("sample_rate", 48000)

    now = datetime.datetime.now()
    date_dir = OUTPUT_DIR / now.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    ts = now.strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name_hint)
    dest = date_dir / f"{safe}_{ts}.wav"

    # tensor 形状为 [channels, samples]，转为 [samples, channels]
    if hasattr(audio_tensor, "cpu"):
        audio_np = audio_tensor.cpu().numpy()
    else:
        audio_np = np.asarray(audio_tensor)
    if audio_np.ndim == 2:
        audio_np = audio_np.T  # [channels, samples] -> [samples, channels]

    sf.write(str(dest), audio_np, sample_rate)
    return str(dest)


def tab_music():
    with gr.Tab("音乐合成"):
        gr.Markdown("### 音乐合成（ACE-Step 1.5）\n"
                     "基于 ACE-Step 模型生成音乐，支持文本生成、翻唱和音频重绘。")

        with gr.Tabs():
            # ==============================================================
            # 子 Tab 1: 文本生成音乐
            # ==============================================================
            with gr.Tab("文本生成音乐"):
                with gr.Row():
                    with gr.Column(scale=2):
                        caption_in = gr.Textbox(
                            label="音乐描述",
                            placeholder="如：欢快的电子舞曲，重低音，充满活力",
                            lines=3,
                        )
                        lyrics_in = gr.Textbox(
                            label="歌词（可选，支持 [Verse] [Chorus] 等标记）",
                            placeholder="[Verse 1]\n歌词内容...\n[Chorus]\n副歌内容...",
                            lines=6,
                        )
                        with gr.Row():
                            instrumental_cb = gr.Checkbox(
                                label="纯器乐（无人声）", value=False,
                            )
                            vocal_lang_dd = gr.Dropdown(
                                choices=VOCAL_LANGUAGES,
                                value="unknown",
                                label="人声语言",
                            )
                        with gr.Row():
                            duration_slider = gr.Slider(
                                minimum=10, maximum=600, value=30, step=5,
                                label="时长（秒）",
                            )
                            bpm_in = gr.Number(
                                label="BPM（可选）", value=0, precision=0,
                                minimum=0, maximum=300,
                            )
                        with gr.Row():
                            keyscale_dd = gr.Dropdown(
                                choices=KEYSCALE_OPTIONS, value="",
                                label="调式（可选）",
                            )
                            timesig_dd = gr.Dropdown(
                                choices=TIME_SIGNATURE_OPTIONS, value="",
                                label="拍号（可选）",
                            )
                        with gr.Accordion("高级参数", open=False):
                            with gr.Row():
                                steps_slider = gr.Slider(
                                    minimum=1, maximum=50, value=8, step=1,
                                    label="推理步数",
                                )
                                guidance_slider = gr.Slider(
                                    minimum=1, maximum=15, value=7, step=0.5,
                                    label="引导系数",
                                )
                                seed_in = gr.Number(
                                    label="随机种子（-1 随机）", value=-1, precision=0,
                                )
                        gen_btn = gr.Button("生成音乐", variant="primary")

                    with gr.Column(scale=2):
                        t2m_audio_out = gr.Audio(label="生成结果", type="filepath", interactive=False)
                        t2m_status_out = gr.Textbox(label="状态", interactive=False)

                def on_text2music(caption, lyrics, instrumental, vocal_lang,
                                  duration, bpm, keyscale, timesig,
                                  steps, guidance, seed, request: gr.Request):
                    if not caption.strip() and not lyrics.strip():
                        return None, "音乐描述和歌词不能同时为空"

                    user = (request.username or "unknown") if request else "-"
                    try:
                        def do_generate():
                            return music_model.generate(
                                task_type="text2music",
                                caption=caption.strip(),
                                lyrics=lyrics.strip(),
                                instrumental=instrumental,
                                vocal_language=vocal_lang,
                                duration=duration,
                                bpm=int(bpm) if bpm and int(bpm) > 0 else None,
                                keyscale=keyscale,
                                timesignature=timesig,
                                inference_steps=int(steps),
                                guidance_scale=guidance,
                                seed=int(seed),
                            )

                        result = task_queue.submit(
                            user, "music", f"音乐合成: {caption[:30]}", do_generate
                        )

                        path = _save_music_output(result, name_hint="music")
                        if path:
                            rel_path = Path(path).relative_to(BASE_DIR).as_posix()
                            log_event(
                                EVENT_SYNTHESIZE,
                                f"类型=音乐合成 描述={caption[:50]} 时长={duration}s 文件={rel_path}",
                                user=user,
                            )
                            return path, f"生成成功\n已保存：{rel_path}"
                        else:
                            error_msg = result.error
                            return None, f"生成失败：{error_msg}"

                    except Exception as e:
                        logger.exception("音乐合成失败")
                        return None, f"音乐合成失败：{e}"

                gen_btn.click(
                    fn=on_text2music,
                    inputs=[caption_in, lyrics_in, instrumental_cb, vocal_lang_dd,
                            duration_slider, bpm_in, keyscale_dd, timesig_dd,
                            steps_slider, guidance_slider, seed_in],
                    outputs=[t2m_audio_out, t2m_status_out],
                    concurrency_limit=10,
                    trigger_mode="multiple",
                )

            # ==============================================================
            # 子 Tab 2: 翻唱/风格迁移
            # ==============================================================
            with gr.Tab("翻唱/风格迁移"):
                gr.Markdown(
                    "上传参考音频，基于其风格生成新音乐。可修改歌词实现翻唱效果。"
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        cover_ref_audio = gr.Audio(
                            label="参考音频（WAV / MP3）",
                            type="filepath",
                            sources=["upload"],
                        )
                        cover_caption = gr.Textbox(
                            label="音乐描述",
                            placeholder="描述目标音乐风格...",
                            lines=3,
                        )
                        cover_lyrics = gr.Textbox(
                            label="歌词（可选，留空则保持原曲歌词风格）",
                            placeholder="[Verse 1]\n新歌词...",
                            lines=5,
                        )
                        with gr.Row():
                            cover_instrumental = gr.Checkbox(
                                label="纯器乐", value=False,
                            )
                            cover_vocal_lang = gr.Dropdown(
                                choices=VOCAL_LANGUAGES, value="unknown",
                                label="人声语言",
                            )
                        with gr.Row():
                            cover_duration = gr.Slider(
                                minimum=10, maximum=600, value=30, step=5,
                                label="时长（秒）",
                            )
                            cover_strength = gr.Slider(
                                minimum=0, maximum=1, value=1.0, step=0.05,
                                label="翻唱强度",
                            )
                        with gr.Accordion("高级参数", open=False):
                            with gr.Row():
                                cover_steps = gr.Slider(
                                    minimum=1, maximum=50, value=8, step=1,
                                    label="推理步数",
                                )
                                cover_guidance = gr.Slider(
                                    minimum=1, maximum=15, value=7, step=0.5,
                                    label="引导系数",
                                )
                                cover_seed = gr.Number(
                                    label="随机种子", value=-1, precision=0,
                                )
                                cover_noise = gr.Slider(
                                    minimum=0, maximum=1, value=0, step=0.05,
                                    label="噪声强度",
                                )
                        cover_btn = gr.Button("生成翻唱", variant="primary")

                    with gr.Column(scale=2):
                        cover_audio_out = gr.Audio(label="翻唱结果", type="filepath", interactive=False)
                        cover_status_out = gr.Textbox(label="状态", interactive=False)

                def on_cover(ref_audio, caption, lyrics, instrumental, vocal_lang,
                             duration, strength, steps, guidance, seed, noise,
                             request: gr.Request):
                    if ref_audio is None:
                        return None, "请先上传参考音频"
                    if not caption.strip():
                        return None, "请输入音乐描述"

                    user = (request.username or "unknown") if request else "-"
                    try:
                        def do_generate():
                            return music_model.generate(
                                task_type="text2music",
                                caption=caption.strip(),
                                lyrics=lyrics.strip(),
                                instrumental=instrumental,
                                vocal_language=vocal_lang,
                                duration=duration,
                                reference_audio=ref_audio,
                                audio_cover_strength=strength,
                                cover_noise_strength=noise,
                                inference_steps=int(steps),
                                guidance_scale=guidance,
                                seed=int(seed),
                            )

                        result = task_queue.submit(
                            user, "music", f"翻唱: {caption[:30]}", do_generate
                        )

                        path = _save_music_output(result, name_hint="cover")
                        if path:
                            rel_path = Path(path).relative_to(BASE_DIR).as_posix()
                            log_event(
                                EVENT_SYNTHESIZE,
                                f"类型=翻唱 描述={caption[:50]} 时长={duration}s 文件={rel_path}",
                                user=user,
                            )
                            return path, f"翻唱生成成功\n已保存：{rel_path}"
                        else:
                            error_msg = result.error if hasattr(result, 'error') and result.error else "未知错误"
                            return None, f"翻唱生成失败：{error_msg}"

                    except Exception as e:
                        logger.exception("翻唱生成失败")
                        return None, f"翻唱生成失败：{e}"

                cover_btn.click(
                    fn=on_cover,
                    inputs=[cover_ref_audio, cover_caption, cover_lyrics,
                            cover_instrumental, cover_vocal_lang,
                            cover_duration, cover_strength,
                            cover_steps, cover_guidance, cover_seed, cover_noise],
                    outputs=[cover_audio_out, cover_status_out],
                    concurrency_limit=10,
                    trigger_mode="multiple",
                )

            # ==============================================================
            # 子 Tab 3: 重绘/编辑
            # ==============================================================
            with gr.Tab("重绘/编辑"):
                gr.Markdown(
                    "上传音频，对指定区间进行重绘编辑。可用于修改局部旋律或风格。"
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        repaint_src_audio = gr.Audio(
                            label="源音频（WAV / MP3）",
                            type="filepath",
                            sources=["upload"],
                        )
                        repaint_caption = gr.Textbox(
                            label="音乐描述（目标风格）",
                            placeholder="描述重绘后的音乐风格...",
                            lines=3,
                        )
                        repaint_lyrics = gr.Textbox(
                            label="歌词（可选）",
                            lines=4,
                        )
                        with gr.Row():
                            repaint_start = gr.Number(
                                label="重绘开始（秒）", value=0, precision=1, minimum=0,
                            )
                            repaint_end = gr.Number(
                                label="重绘结束（秒，-1 到末尾）", value=-1, precision=1,
                            )
                        with gr.Row():
                            repaint_strength_slider = gr.Slider(
                                minimum=0, maximum=1, value=0.5, step=0.05,
                                label="重绘强度",
                            )
                            repaint_mode_dd = gr.Dropdown(
                                choices=REPAINT_MODES, value="balanced",
                                label="重绘模式",
                            )
                        with gr.Accordion("高级参数", open=False):
                            with gr.Row():
                                repaint_steps = gr.Slider(
                                    minimum=1, maximum=50, value=8, step=1,
                                    label="推理步数",
                                )
                                repaint_guidance = gr.Slider(
                                    minimum=1, maximum=15, value=7, step=0.5,
                                    label="引导系数",
                                )
                                repaint_seed = gr.Number(
                                    label="随机种子", value=-1, precision=0,
                                )
                        repaint_btn = gr.Button("重绘音频", variant="primary")

                    with gr.Column(scale=2):
                        repaint_audio_out = gr.Audio(label="重绘结果", type="filepath", interactive=False)
                        repaint_status_out = gr.Textbox(label="状态", interactive=False)

                def on_repaint(src_audio, caption, lyrics,
                               start, end, strength, mode,
                               steps, guidance, seed,
                               request: gr.Request):
                    if src_audio is None:
                        return None, "请先上传源音频"
                    if not caption.strip():
                        return None, "请输入音乐描述"

                    user = (request.username or "unknown") if request else "-"
                    try:
                        def do_generate():
                            return music_model.generate(
                                task_type="repaint",
                                caption=caption.strip(),
                                lyrics=lyrics.strip() if lyrics else "",
                                src_audio=src_audio,
                                repainting_start=float(start),
                                repainting_end=float(end),
                                repaint_strength=strength,
                                repaint_mode=mode,
                                inference_steps=int(steps),
                                guidance_scale=guidance,
                                seed=int(seed),
                            )

                        result = task_queue.submit(
                            user, "music", f"音频重绘: {caption[:30]}", do_generate
                        )

                        path = _save_music_output(result, name_hint="repaint")
                        if path:
                            rel_path = Path(path).relative_to(BASE_DIR).as_posix()
                            log_event(
                                EVENT_SYNTHESIZE,
                                f"类型=音频重绘 描述={caption[:50]} 文件={rel_path}",
                                user=user,
                            )
                            return path, f"重绘成功\n已保存：{rel_path}"
                        else:
                            error_msg = result.error if hasattr(result, 'error') and result.error else "未知错误"
                            return None, f"重绘失败：{error_msg}"

                    except Exception as e:
                        logger.exception("音频重绘失败")
                        return None, f"音频重绘失败：{e}"

                repaint_btn.click(
                    fn=on_repaint,
                    inputs=[repaint_src_audio, repaint_caption, repaint_lyrics,
                            repaint_start, repaint_end, repaint_strength_slider,
                            repaint_mode_dd, repaint_steps, repaint_guidance,
                            repaint_seed],
                    outputs=[repaint_audio_out, repaint_status_out],
                    concurrency_limit=10,
                    trigger_mode="multiple",
                )
