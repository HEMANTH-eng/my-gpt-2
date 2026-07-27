from dataclasses import dataclass
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional

from utils.logger import get_logger

logger = get_logger("task_queue")


@dataclass
class AsyncTask:
    """Represents a background async task item."""

    task_id: str
    func: Callable[..., Any]
    args: tuple
    kwargs: dict
    status: str = "pending"  # 'pending', 'processing', 'completed', 'failed'
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = time.time()


class TaskQueue:
    """Thread-safe background task queue worker engine for async execution."""

    def __init__(self, num_workers: int = 2) -> None:
        self.work_queue: queue.Queue[AsyncTask] = queue.Queue()
        self.tasks: Dict[str, AsyncTask] = {}
        self.num_workers = num_workers
        self.workers: list = []
        self.running = False
        self.start_workers()

    def start_workers(self) -> None:
        """Starts background worker threads."""
        self.running = True
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, name=f"TaskWorker-{i+1}", daemon=True)
            t.start()
            self.workers.append(t)
        logger.info(f"TaskQueue started with {self.num_workers} background worker threads.")

    def enqueue(self, func: Callable[..., Any], *args, **kwargs) -> str:
        """Enqueues function task for background execution."""
        task_id = f"job_{int(time.time() * 1000)}_{len(self.tasks) + 1}"
        task = AsyncTask(task_id=task_id, func=func, args=args, kwargs=kwargs)
        self.tasks[task_id] = task
        self.work_queue.put(task)
        logger.info(f"Enqueued async task '{task_id}'. Queue size: {self.work_queue.qsize()}")
        return task_id

    def get_task(self, task_id: str) -> Optional[AsyncTask]:
        """Retrieves task state by task_id."""
        return self.tasks.get(task_id)

    def _worker_loop(self) -> None:
        """Worker loop picking up tasks from FIFO queue."""
        while self.running:
            try:
                task = self.work_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            task.status = "processing"
            logger.info(f"Worker [{threading.current_thread().name}] processing task '{task.task_id}'...")

            try:
                task.result = task.func(*task.args, **task.kwargs)
                task.status = "completed"
                logger.info(f"Task '{task.task_id}' completed successfully.")
            except Exception as e:
                task.error = str(e)
                task.status = "failed"
                logger.error(f"Task '{task.task_id}' failed: {e}")
            finally:
                self.work_queue.task_done()


# Global Task Queue Singleton
global_task_queue = TaskQueue()
