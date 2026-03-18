"""
core/batch_processor.py

批量处理：TTS / 音效 / 音乐，读取 CSV/Excel，逐行合成，支持进度回调。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from core.audio_utils import AudioFormat, normalize_audio, save_audio
from core.preset_model import PresetModel, PRESET_VOICES
from core.clone_model import CloneModel
from core.design_model import DesignModel

if TYPE_CHECKING:
    from core.sfx_model import SfxModel
    from core.music_model import MusicModel

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    total: int = 0
    success: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    output_files: list[Path] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "批量处理完成",
            f"  总计：{self.total} 条",
            f"  成功：{self.success} 条",
            f"  失败：{self.failed} 条",
        ]
        if self.errors:
            lines.append("\n错误明细：")
            lines.extend(f"  {e}" for e in self.errors)
        return "\n".join(lines)


def load_table(file_path: str | Path) -> pd.DataFrame:
    """自动识别 CSV / Excel 并读取为 DataFrame。"""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    elif suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
    else:
        raise ValueError(f"不支持的文件格式：{suffix}（仅支持 .csv / .xlsx / .xls）")
    return df.fillna("")


class BatchProcessor:

    def __init__(
        self,
        preset_model: PresetModel,
        clone_model: CloneModel,
        design_model: DesignModel,
        sfx_model: Optional[SfxModel] = None,
        music_model: Optional[MusicModel] = None,
    ):
        self.preset_model = preset_model
        self.clone_model = clone_model
        self.design_model = design_model
        self.sfx_model = sfx_model
        self.music_model = music_model

    def run(
        self,
        df: pd.DataFrame,
        output_dir: str | Path,
        output_format: AudioFormat = "wav",
        characters: Optional[dict] = None,
        design_characters: Optional[dict] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> BatchResult:
        """逐行合成。固定列名：character（必填）、text（必填）、filename（可选）。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        characters = characters or {}
        design_characters = design_characters or {}
        result = BatchResult(total=len(df))

        for idx, row in df.iterrows():
            row_num = int(idx) + 1  # type: ignore
            text = str(row.get("text", "")).strip()
            char_name = str(row.get("character", "")).strip()

            if not text:
                msg = f"第 {row_num} 行：文本为空，跳过"
                logger.warning(msg)
                result.failed += 1
                result.errors.append(msg)
                if progress_cb:
                    progress_cb(row_num, result.total, msg)
                continue

            if not char_name:
                msg = f"第 {row_num} 行：角色为空，跳过"
                logger.warning(msg)
                result.failed += 1
                result.errors.append(msg)
                if progress_cb:
                    progress_cb(row_num, result.total, msg)
                continue

            # 输出文件名
            fname = str(row.get("filename", "")).strip()
            stem = fname if fname else f"{row_num:04d}"
            out_path = output_dir / f"{stem}.{output_format}"

            # 查找角色配置：克隆角色 → 设计角色 → 预设人声
            cfg = characters.get(char_name) or design_characters.get(char_name)
            if not cfg and char_name in PRESET_VOICES:
                cfg = {"voice_type": "preset", "voice_name": char_name}
            if not cfg:
                msg = f"第 {row_num} 行：角色「{char_name}」不存在，跳过"
                logger.warning(msg)
                result.failed += 1
                result.errors.append(msg)
                if progress_cb:
                    progress_cb(row_num, result.total, msg)
                continue

            try:
                if progress_cb:
                    progress_cb(row_num, result.total, f"正在合成 {row_num}/{result.total}：{text[:30]}…")

                voice_type = cfg.get("voice_type", "")

                if voice_type == "design":
                    instruct = cfg.get("instruct", "")
                    audio, sr = self.design_model.synthesize(text=text, instruct=instruct)
                elif voice_type == "clone":
                    rp = cfg.get("ref_audio_path", "")
                    ref_audio = None
                    ref_text = None
                    if rp:
                        abs_rp = (Path(__file__).resolve().parent.parent / rp) if not Path(rp).is_absolute() else Path(rp)
                        if abs_rp.exists():
                            ref_audio = str(abs_rp)
                            ref_text = cfg.get("ref_text")
                    audio, sr = self.clone_model.synthesize(text=text, ref_audio=ref_audio, ref_text=ref_text)
                else:
                    # preset
                    voice_name = cfg.get("voice_name")
                    audio, sr = self.preset_model.synthesize(text=text, voice_name=voice_name)

                audio = normalize_audio(audio)
                saved = save_audio(audio, sr, out_path, output_format)
                result.success += 1
                result.output_files.append(saved)

            except Exception as e:
                msg = f"第 {row_num} 行合成失败：{e}"
                logger.error(msg)
                result.failed += 1
                result.errors.append(msg)
                if progress_cb:
                    progress_cb(row_num, result.total, f"[失败] {msg}")

        if progress_cb:
            progress_cb(result.total, result.total, result.summary())
        return result

    # ------------------------------------------------------------------
    # 批量音效合成
    # ------------------------------------------------------------------

    def run_sfx(
        self,
        df: pd.DataFrame,
        output_dir: str | Path,
        output_format: AudioFormat = "wav",
        default_duration: float = 10.0,
        steps: int = 100,
        guidance_scale: float = 7.0,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> BatchResult:
        """逐行合成音效。列：prompt（必填）、negative_prompt / duration / filename（可选）。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = BatchResult(total=len(df))

        for idx, row in df.iterrows():
            row_num = int(idx) + 1
            prompt = str(row.get("prompt", "")).strip()

            if not prompt:
                msg = f"第 {row_num} 行：prompt 为空，跳过"
                logger.warning(msg)
                result.failed += 1
                result.errors.append(msg)
                if progress_cb:
                    progress_cb(row_num, result.total, msg)
                continue

            negative = str(row.get("negative_prompt", "")).strip()
            dur_str = str(row.get("duration", "")).strip()
            duration = float(dur_str) if dur_str else default_duration
            fname = str(row.get("filename", "")).strip()
            stem = fname if fname else f"{row_num:04d}"
            out_path = output_dir / f"{stem}.{output_format}"

            try:
                if progress_cb:
                    progress_cb(row_num, result.total, f"正在生成 {row_num}/{result.total}：{prompt[:40]}…")

                audio, sr = self.sfx_model.generate(
                    prompt=prompt,
                    negative_prompt=negative,
                    duration=duration,
                    steps=steps,
                    guidance_scale=guidance_scale,
                )
                audio = normalize_audio(audio)
                saved = save_audio(audio, sr, out_path, output_format)
                result.success += 1
                result.output_files.append(saved)

            except Exception as e:
                msg = f"第 {row_num} 行生成失败：{e}"
                logger.error(msg)
                result.failed += 1
                result.errors.append(msg)
                if progress_cb:
                    progress_cb(row_num, result.total, f"[失败] {msg}")

        if progress_cb:
            progress_cb(result.total, result.total, result.summary())
        return result

    # ------------------------------------------------------------------
    # 批量音乐合成
    # ------------------------------------------------------------------

    def run_music(
        self,
        df: pd.DataFrame,
        output_dir: str | Path,
        default_duration: float = 30.0,
        inference_steps: int = 8,
        guidance_scale: float = 7.0,
        vocal_language: str = "unknown",
        default_instrumental: bool = False,
        dit_config: str = "",
        lm_model: str = "",
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> BatchResult:
        """逐行合成音乐。列：caption / lyrics（至少一个）、instrumental / duration / filename（可选）。"""
        import soundfile as sf

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = BatchResult(total=len(df))

        for idx, row in df.iterrows():
            row_num = int(idx) + 1
            caption = str(row.get("caption", "")).strip()
            lyrics = str(row.get("lyrics", "")).strip()

            if not caption and not lyrics:
                msg = f"第 {row_num} 行：caption 和 lyrics 均为空，跳过"
                logger.warning(msg)
                result.failed += 1
                result.errors.append(msg)
                if progress_cb:
                    progress_cb(row_num, result.total, msg)
                continue

            inst_str = str(row.get("instrumental", "")).strip().upper()
            instrumental = inst_str == "TRUE" if inst_str else default_instrumental
            dur_str = str(row.get("duration", "")).strip()
            duration = float(dur_str) if dur_str else default_duration
            fname = str(row.get("filename", "")).strip()
            stem = fname if fname else f"{row_num:04d}"
            out_path = output_dir / f"{stem}.wav"

            try:
                if progress_cb:
                    progress_cb(row_num, result.total, f"正在生成 {row_num}/{result.total}：{(caption or lyrics)[:40]}…")

                kwargs = dict(
                    task_type="text2music",
                    caption=caption,
                    lyrics=lyrics,
                    instrumental=instrumental,
                    vocal_language=vocal_language,
                    duration=duration,
                    inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                )
                if dit_config:
                    kwargs["dit_config"] = dit_config
                if lm_model:
                    kwargs["lm_model"] = lm_model
                gen_result = self.music_model.generate(**kwargs)

                if not gen_result.success or not gen_result.audios:
                    raise RuntimeError(gen_result.error or "生成结果为空")

                audio_info = gen_result.audios[0]
                audio_tensor = audio_info.get("tensor")
                sample_rate = audio_info.get("sample_rate", 48000)

                if hasattr(audio_tensor, "cpu"):
                    audio_np = audio_tensor.cpu().numpy()
                else:
                    audio_np = np.asarray(audio_tensor)
                if audio_np.ndim == 2:
                    audio_np = audio_np.T  # [channels, samples] -> [samples, channels]

                sf.write(str(out_path), audio_np, sample_rate)
                result.success += 1
                result.output_files.append(out_path)

            except Exception as e:
                msg = f"第 {row_num} 行生成失败：{e}"
                logger.error(msg)
                result.failed += 1
                result.errors.append(msg)
                if progress_cb:
                    progress_cb(row_num, result.total, f"[失败] {msg}")

        if progress_cb:
            progress_cb(result.total, result.total, result.summary())
        return result
