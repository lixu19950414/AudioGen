"""core/app_logger.py — 业务事件日志，按天轮转写入 logs/ 目录"""

from __future__ import annotations

import datetime
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# 专用 logger，不污染 root logger
_logger = logging.getLogger("app_event")
_logger.setLevel(logging.INFO)
_logger.propagate = False

_handler = TimedRotatingFileHandler(
    str(LOGS_DIR / "app.log"),
    when="midnight",
    backupCount=90,
    encoding="utf-8",
)
_handler.setFormatter(logging.Formatter("%(asctime)s\t%(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
_logger.addHandler(_handler)

# 事件类型常量
EVENT_LOGIN = "login"
EVENT_SYNTHESIZE = "synthesize"
EVENT_DOWNLOAD = "download"
EVENT_CHAR_ADD = "char_add"
EVENT_CHAR_REMOVE = "char_remove"

ALL_EVENT_TYPES = [EVENT_LOGIN, EVENT_SYNTHESIZE, EVENT_DOWNLOAD, EVENT_CHAR_ADD, EVENT_CHAR_REMOVE]


def log_event(event_type: str, detail: str = ""):
    """记录一条业务事件日志。格式: 时间\t事件类型\t详情"""
    _logger.info("%s\t%s", event_type, detail)


def read_logs(date: str | None = None, event_type: str | None = None, tail: int = 500) -> str:
    """读取日志内容，支持按日期和事件类型筛选。

    Args:
        date: 日期字符串 YYYY-MM-DD，None 表示今天
        event_type: 事件类型筛选，None 或空串表示全部
        tail: 最多返回最后 N 行
    """
    if not date:
        date = datetime.date.today().isoformat()

    # 主日志文件和轮转文件
    log_file = LOGS_DIR / "app.log"
    if not log_file.exists():
        return "暂无日志"

    lines: list[str] = []
    # 读取主日志文件
    try:
        for line in log_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            # 按日期筛选
            if not line.startswith(date):
                continue
            # 按事件类型筛选
            if event_type:
                parts = line.split("\t")
                if len(parts) >= 2 and parts[1] != event_type:
                    continue
            lines.append(line)
    except Exception:
        return "读取日志失败"

    # 也检查轮转的旧日志文件（文件名如 app.log.2026-03-13）
    rotated = LOGS_DIR / f"app.log.{date}"
    if rotated.exists():
        try:
            rotated_lines: list[str] = []
            for line in rotated.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                if event_type:
                    parts = line.split("\t")
                    if len(parts) >= 2 and parts[1] != event_type:
                        continue
                rotated_lines.append(line)
            lines = rotated_lines + lines
        except Exception:
            pass

    if not lines:
        return f"{date} 无匹配日志"

    # 取最后 tail 行，最新的在最后
    lines = lines[-tail:]
    return "\n".join(lines)
