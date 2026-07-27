import time
import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.metrics import generate_prometheus_metrics, track_latency, track_request
from models.gpu_pool import GPUDevicePool
from utils.cache import ResponseCache
from utils.task_queue import TaskQueue


def test_response_cache_hit_miss_and_eviction():
    cache = ResponseCache(max_size=2, default_ttl=2)

    # Miss test
    assert cache.get("prompt 1") is None
    assert cache.hit_ratio == 0.0

    # Set & Hit test
    cache.set("prompt 1", "response 1")
    assert cache.get("prompt 1") == "response 1"
    assert cache.hits == 1

    # LRU Eviction test (max_size=2)
    cache.set("prompt 2", "response 2")
    cache.set("prompt 3", "response 3")
    # "prompt 1" should be evicted
    assert cache.get("prompt 1") is None
    assert cache.get("prompt 2") == "response 2"
    assert cache.get("prompt 3") == "response 3"


def test_gpu_device_pool():
    pool = GPUDevicePool()
    dev1 = pool.get_next_device()
    dev2 = pool.get_next_device()

    assert dev1 is not None
    assert dev2 is not None

    status = pool.get_pool_status()
    assert len(status) > 0
    assert "device" in status[0]
    assert "status" in status[0]


def test_task_queue_async_execution():
    tq = TaskQueue(num_workers=1)

    def sample_func(x, y):
        return x + y

    task_id = tq.enqueue(sample_func, 15, 25)
    assert task_id.startswith("job_")

    # Wait briefly for worker thread execution
    time.sleep(0.2)

    task = tq.get_task(task_id)
    assert task is not None
    assert task.status == "completed"
    assert task.result == 40


def test_prometheus_metrics_exporter():
    track_request("chat")
    track_latency(0.125)

    metrics_text = generate_prometheus_metrics()
    assert "mygpt_requests_total" in metrics_text
    assert "mygpt_inference_latency_seconds_average" in metrics_text
    assert "mygpt_cache_hit_ratio" in metrics_text

    client = TestClient(app)
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "mygpt_requests_total" in res.text
