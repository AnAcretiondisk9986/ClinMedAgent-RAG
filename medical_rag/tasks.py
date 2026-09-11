"""后台任务：线程执行、阶段进度、日志缓冲与取消。

供本地网站（medical_rag.webapp）使用：导入/处理教材等耗时操作在线程里跑，
前端通过快照接口轮询进度和日志。任务只保存在内存里，进程退出即清空。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

MAX_LOG_LINES = 4000
MAX_TASKS = 40


class TaskCancelled(Exception):
    """任务被用户取消。"""


@dataclass
class StageState:
    key: str
    label: str
    status: str = "pending"  # pending | running | done | error | skipped
    done: int = 0
    total: int = 0
    message: str = ""
    started_at: float | None = None
    finished_at: float | None = None

    def ratio(self) -> float:
        if self.status == "done":
            return 1.0
        if self.status == "skipped":
            return 1.0
        if self.total > 0:
            return max(0.0, min(1.0, self.done / self.total))
        return 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "status": self.status,
            "done": self.done,
            "total": self.total,
            "message": self.message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class Task:
    """一个后台任务；所有可变状态都在锁内更新。"""

    def __init__(self, kind: str, label: str, meta: dict[str, Any] | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.label = label
        self.meta = meta or {}
        self.status = "pending"  # pending | running | done | error | cancelled
        self.error = ""
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.stages: list[StageState] = []
        self._stage_index: dict[str, StageState] = {}
        self._logs: list[tuple[int, str]] = []
        self._cursor = 0
        self._progress_done = 0
        self._progress_total = 0
        self._progress_message = ""
        self._cancel = threading.Event()
        self._process = None  # subprocess.Popen | None
        self._lock = threading.RLock()
        self._runner: Callable[["Task"], None] | None = None

    # ---------------------------------------------------------------- 日志
    def log(self, text: str) -> None:
        text = str(text).rstrip("\r\n")
        if not text:
            return
        with self._lock:
            self._cursor += 1
            stamp = time.strftime("%H:%M:%S")
            self._logs.append((self._cursor, f"{stamp}  {text}"))
            if len(self._logs) > MAX_LOG_LINES:
                del self._logs[: len(self._logs) - MAX_LOG_LINES]

    # ------------------------------------------------------------ 阶段进度
    def set_stages(self, stages: list[tuple[str, str]]) -> None:
        with self._lock:
            self.stages = [StageState(key=key, label=label) for key, label in stages]
            self._stage_index = {stage.key: stage for stage in self.stages}

    def start_stage(self, key: str, label: str | None = None, total: int = 0, message: str = "") -> StageState:
        with self._lock:
            stage = self._stage_index.get(key)
            if stage is None:
                stage = StageState(key=key, label=label or key)
                self.stages.append(stage)
                self._stage_index[key] = stage
            stage.status = "running"
            stage.started_at = time.time()
            stage.finished_at = None
            stage.done = 0
            stage.total = max(0, int(total))
            stage.message = message
            return stage

    def progress(self, done: int, total: int | None = None, message: str | None = None) -> None:
        with self._lock:
            running = next((s for s in self.stages if s.status == "running"), None)
            if running is not None:
                running.done = max(0, int(done))
                if total is not None:
                    running.total = max(0, int(total))
                if message is not None:
                    running.message = message
            else:
                self._progress_done = max(0, int(done))
                if total is not None:
                    self._progress_total = max(0, int(total))
                if message is not None:
                    self._progress_message = message

    def stage_message(self, message: str) -> None:
        with self._lock:
            running = next((s for s in self.stages if s.status == "running"), None)
            if running is not None:
                running.message = message

    def finish_stage(self, message: str = "") -> None:
        with self._lock:
            running = next((s for s in self.stages if s.status == "running"), None)
            if running is not None:
                running.status = "done"
                running.finished_at = time.time()
                if running.total == 0:
                    running.total = max(1, running.done)
                if running.done == 0:
                    running.done = running.total
                if message:
                    running.message = message

    def fail_stage(self, error: str) -> None:
        with self._lock:
            running = next((s for s in self.stages if s.status == "running"), None)
            if running is not None:
                running.status = "error"
                running.finished_at = time.time()
                running.message = error

    def skip_stage(self, key: str, label: str, message: str = "已完成，跳过") -> None:
        with self._lock:
            stage = self._stage_index.get(key)
            if stage is None:
                stage = StageState(key=key, label=label)
                self.stages.append(stage)
                self._stage_index[key] = stage
            stage.status = "skipped"
            stage.message = message
            stage.finished_at = time.time()

    # ------------------------------------------------------------ 整体进度
    def set_progress(self, done: int, total: int, message: str = "") -> None:
        self.progress(done, total, message)

    @property
    def percent(self) -> float:
        with self._lock:
            if self.stages:
                finished = sum(1 for s in self.stages if s.status in ("done", "skipped"))
                running = next((s for s in self.stages if s.status == "running"), None)
                fraction = running.ratio() if running else 0.0
                return round(min(100.0, (finished + fraction) / len(self.stages) * 100), 1)
            if self._progress_total > 0:
                return round(min(100.0, self._progress_done / self._progress_total * 100), 1)
            return 100.0 if self.status == "done" else 0.0

    # ------------------------------------------------------------ 手动任务
    # （上传这类由 HTTP 处理函数驱动的任务，不走 runner 线程）
    def mark_running(self) -> None:
        with self._lock:
            self.status = "running"
            self.started_at = time.time()

    def mark_done(self, message: str = "") -> None:
        with self._lock:
            self.status = "done"
            self.finished_at = time.time()
            if self._progress_total > 0:
                self._progress_done = self._progress_total
        if message:
            self.log(message)

    def mark_error(self, error: str) -> None:
        with self._lock:
            self.status = "error"
            self.error = error
            self.finished_at = time.time()
        self.log(f"任务失败：{error}")

    def mark_cancelled(self, message: str = "任务已取消") -> None:
        with self._lock:
            self.status = "cancelled"
            self.error = message
            self.finished_at = time.time()
        self.log(message)

    # ---------------------------------------------------------------- 取消
    def cancel(self) -> None:
        with self._lock:
            self._cancel.set()
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise TaskCancelled()

    def attach_process(self, process) -> None:
        with self._lock:
            self._process = process
        if self._cancel.is_set() and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    # ---------------------------------------------------------------- 快照
    def as_dict(self, include_logs_since: int | None = None) -> dict[str, Any]:
        with self._lock:
            running = next((s for s in self.stages if s.status == "running"), None)
            data: dict[str, Any] = {
                "id": self.id,
                "kind": self.kind,
                "label": self.label,
                "meta": self.meta,
                "status": self.status,
                "error": self.error,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "percent": self.percent,
                "stages": [stage.as_dict() for stage in self.stages],
                "current_stage": running.key if running else None,
                "progress_message": running.message if running else self._progress_message,
                "log_cursor": self._cursor,
            }
            if include_logs_since is not None:
                data["logs"] = [line for cursor, line in self._logs if cursor > include_logs_since]
            return data


class TaskManager:
    """创建、查询与取消任务。"""

    def __init__(self, max_tasks: int = MAX_TASKS):
        self._tasks: "OrderedDict[str, Task]" = OrderedDict()
        self._lock = threading.RLock()
        self._max_tasks = max_tasks

    def create(
        self,
        kind: str,
        label: str,
        runner: Callable[[Task], None],
        meta: dict[str, Any] | None = None,
    ) -> Task:
        task = Task(kind, label, meta)
        task._runner = runner
        with self._lock:
            self._tasks[task.id] = task
            while len(self._tasks) > self._max_tasks:
                oldest_id, oldest = next(iter(self._tasks.items()))
                if oldest.status in ("running", "pending"):
                    break
                self._tasks.pop(oldest_id)
        thread = threading.Thread(target=self._run, args=(task,), name=f"task-{task.id}", daemon=True)
        thread.start()
        return task

    def create_manual(
        self,
        kind: str,
        label: str,
        meta: dict[str, Any] | None = None,
    ) -> Task:
        """注册一个由调用方自己驱动的任务（如 HTTP 流式上传）。"""
        task = Task(kind, label, meta)
        with self._lock:
            self._tasks[task.id] = task
            while len(self._tasks) > self._max_tasks:
                oldest_id, oldest = next(iter(self._tasks.items()))
                if oldest.status in ("running", "pending"):
                    break
                self._tasks.pop(oldest_id)
        return task

    @staticmethod
    def _run(task: Task) -> None:
        task.status = "running"
        task.started_at = time.time()
        task.log(f"任务开始：{task.label}")
        try:
            runner = task._runner
            if runner is None:
                raise RuntimeError("任务没有执行体")
            runner(task)
        except TaskCancelled:
            task.status = "cancelled"
            task.error = "任务已取消"
            task.log("任务已取消")
        except Exception as exc:  # noqa: BLE001 - 需要把任何失败反馈给前端
            task.status = "error"
            task.error = f"{type(exc).__name__}: {exc}"
            task.log(f"任务失败：{task.error}")
        else:
            if task.status == "running":
                task.status = "done"
        finally:
            task.finished_at = time.time()
            task._process = None
            task.log(f"任务结束：{task.label}（{task.status}）")

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [task.as_dict() for task in reversed(self._tasks.values())]

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def snapshot(self, task_id: str, since: int = 0) -> dict[str, Any] | None:
        task = self.get(task_id)
        if task is None:
            return None
        return task.as_dict(include_logs_since=since)

    def cancel(self, task_id: str) -> bool:
        task = self.get(task_id)
        if task is None:
            return False
        if task.status in ("running", "pending"):
            task.cancel()
            return True
        return False
