import time
from typing import Dict
from models.gpu_pool import global_gpu_pool
from utils.cache import global_response_cache

REQUEST_COUNTER: Dict[str, int] = {
    "generate": 0,
    "chat": 0,
    "upload": 0,
    "vision": 0,
    "agents": 0,
    "tools": 0,
}

INFERENCE_LATENCY_SUM = 0.0
INFERENCE_LATENCY_COUNT = 0


def track_request(endpoint: str) -> None:
    """Increments Prometheus request counter for endpoint."""
    if endpoint in REQUEST_COUNTER:
        REQUEST_COUNTER[endpoint] += 1
    else:
        REQUEST_COUNTER[endpoint] = 1


def track_latency(duration_seconds: float) -> None:
    """Records inference latency duration."""
    global INFERENCE_LATENCY_SUM, INFERENCE_LATENCY_COUNT
    INFERENCE_LATENCY_SUM += duration_seconds
    INFERENCE_LATENCY_COUNT += 1


def generate_prometheus_metrics() -> str:
    """Generates Prometheus metric exposition text format."""
    lines = [
        "# HELP mygpt_requests_total Total number of HTTP API requests received",
        "# TYPE mygpt_requests_total counter",
    ]
    for endpoint, count in REQUEST_COUNTER.items():
        lines.append(f'mygpt_requests_total{{endpoint="{endpoint}"}} {count}')

    avg_latency = (INFERENCE_LATENCY_SUM / max(1, INFERENCE_LATENCY_COUNT))
    lines.extend([
        "# HELP mygpt_inference_latency_seconds_average Average inference latency in seconds",
        "# TYPE mygpt_inference_latency_seconds_average gauge",
        f'mygpt_inference_latency_seconds_average {avg_latency:.4f}',
        "# HELP mygpt_cache_hits_total Total response cache hits",
        "# TYPE mygpt_cache_hits_total counter",
        f'mygpt_cache_hits_total {global_response_cache.hits}',
        "# HELP mygpt_cache_hit_ratio Cache hit ratio percentage",
        "# TYPE mygpt_cache_hit_ratio gauge",
        f'mygpt_cache_hit_ratio {global_response_cache.hit_ratio:.2f}',
    ])

    gpu_status = global_gpu_pool.get_pool_status()
    lines.extend([
        "# HELP mygpt_gpu_memory_allocated_mb Allocated GPU hardware memory in megabytes",
        "# TYPE mygpt_gpu_memory_allocated_mb gauge",
    ])
    for item in gpu_status:
        lines.append(f'mygpt_gpu_memory_allocated_mb{{device="{item["device"]}"}} {item["allocated_mb"]}')

    return "\n".join(lines) + "\n"
