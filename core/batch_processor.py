"""
core/batch_processor.py

批量处理逻辑：
  - 读取 CSV / Excel 文件
  - 遍历每行，调用 TTSEngine 生成音频
  - 支持进度回调和错误跳过
  - 输出汇总报告
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from core.audio_utils import AudioFormat, normalize_audio, save_audio
from core.tts_engine import TTSEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    total: int = 0
    success: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    output_files: list[Path] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"批量处理完成",
            f"  总计：{self.total} 条",
            f"  成功：{self.success} 条",
            f"  失败：{self.failed} 条",
        ]
        if self.errors:
            lines.append("\n错误明细：")
            lines.extend(f"  {e}" for e in self.errors)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 文件读取
# ---------------------------------------------------------------------------

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
    df = df.fillna("")
    return df


# ---------------------------------------------------------------------------
# 批量处理器
# ---------------------------------------------------------------------------

class BatchProcessor:
    """批量 TTS 处理器。"""

    def __init__(self, engine: TTSEngine):
        self.engine = engine

    def run(
        self,
        df: pd.DataFrame,
        text_col: str,
        output_dir: str | Path,
        output_format: AudioFormat = "wav",
        char_col: Optional[str] = None,
        filename_col: Optional[str] = None,
        characters: Optional[dict] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> BatchResult:
        """
        遍历 DataFrame 逐行合成。

        参数:
            df: 输入数据表
            text_col: 文本列名
            output_dir: 输出目录
            output_format: "wav" 或 "mp3"
            char_col: 角色列名（可选），用于从 characters 字典查找音色
            filename_col: 文件名列名（可选），若无则按行号命名
            characters: 角色配置字典 {name: {...}}
            progress_cb: 进度回调 (current, total, message)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        characters = characters or {}

        result = BatchResult(total=len(df))

        for idx, row in df.iterrows():
            row_num = int(idx) + 1  # type: ignore
            text = str(row.get(text_col, "")).strip()

            if not text:
                msg = f"第 {row_num} 行：文本为空，跳过"
                logger.warning(msg)
                result.failed += 1
                result.errors.append(msg)
                if progress_cb:
                    progress_cb(row_num, result.total, msg)
                continue

            # 文件名
            if filename_col and row.get(filename_col, "").strip():
                stem = str(row[filename_col]).strip()
            else:
                stem = f"{row_num:04d}"
            out_path = output_dir / f"{stem}.{output_format}"

            # 角色 → 音色参数
            voice_name: Optional[str] = None
            ref_audio = None
            ref_text: Optional[str] = None

            if char_col and row.get(char_col, "").strip():
                char_name = str(row[char_col]).strip()
                char_cfg = characters.get(char_name, {})
                voice_type = char_cfg.get("voice_type", "preset")
                if voice_type == "preset":
                    voice_name = char_cfg.get("voice_name")
                elif voice_type == "clone":
                    ref_audio_path = char_cfg.get("ref_audio_path", "")
                    if ref_audio_path and Path(ref_audio_path).exists():
                        ref_audio = ref_audio_path
                        ref_text = char_cfg.get("ref_text")

            try:
                if progress_cb:
                    progress_cb(row_num, result.total, f"正在合成第 {row_num}/{result.total} 条：{text[:30]}…")

                audio, sr = self.engine.synthesize(
                    text=text,
                    voice_name=voice_name,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
                audio = normalize_audio(audio)
                saved = save_audio(audio, sr, out_path, output_format)
                result.success += 1
                result.output_files.append(saved)
                logger.info("已生成：%s", saved)

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
