"""
任务队列模块 - 保证同一时间只有一个模型推理任务在执行
"""

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class TaskEntry:
    task_id: str
    username: str
    task_type: str          # "preset" | "clone" | "design" | "batch"
    description: str
    callable: Callable
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "queued"  # "queued" | "running" | "done" | "error"
    result: Any = None
    error: Optional[Exception] = None
    done_event: threading.Event = field(default_factory=threading.Event)


class TaskQueue:
    """单 worker 线程任务队列，保证同一时间只执行一个推理任务"""

    TIMEOUT = 43200  # 12 小时

    def __init__(self):
        self._queue: queue.Queue[TaskEntry] = queue.Queue()
        self._lock = threading.Lock()
        self._current: Optional[TaskEntry] = None
        self._waiting: list[TaskEntry] = []
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def submit(self, username: str, task_type: str,
               description: str, fn: Callable) -> Any:
        """提交任务并阻塞等待结果。返回 fn() 的返回值，或抛出异常。"""
        entry = TaskEntry(
            task_id=uuid.uuid4().hex[:8],
            username=username,
            task_type=task_type,
            description=description,
            callable=fn,
        )
        with self._lock:
            self._waiting.append(entry)
        self._queue.put(entry)

        if not entry.done_event.wait(timeout=self.TIMEOUT):
            entry.status = "error"
            entry.error = TimeoutError("任务超时")
            with self._lock:
                if entry in self._waiting:
                    self._waiting.remove(entry)

        if entry.error:
            raise entry.error
        return entry.result

    def get_status(self) -> dict:
        """返回当前任务和等待列表的线程安全快照"""
        with self._lock:
            return {
                "current": self._current,
                "waiting": list(self._waiting),
            }

    def _run_worker(self):
        """后台 worker：逐个取任务执行"""
        while True:
            entry = self._queue.get()
            with self._lock:
                self._current = entry
                if entry in self._waiting:
                    self._waiting.remove(entry)
                entry.status = "running"
                entry.started_at = time.time()
            try:
                entry.result = entry.callable()
                entry.status = "done"
            except Exception as e:
                entry.error = e
                entry.status = "error"
            finally:
                entry.completed_at = time.time()
                with self._lock:
                    self._current = None
                entry.done_event.set()
