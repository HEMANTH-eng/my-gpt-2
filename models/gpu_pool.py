from typing import Any, Dict, List
import torch

from utils.logger import get_logger

logger = get_logger("gpu_pool")


class GPUDevicePool:
    """Multi-GPU Device Pool for load balancing inference across GPU worker servers."""

    def __init__(self) -> None:
        self.devices: List[str] = []
        self._detect_devices()
        self.current_idx = 0

    def _detect_devices(self) -> None:
        """Detects available CUDA GPU devices or falls back to CPU."""
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            self.devices = [f"cuda:{i}" for i in range(count)]
            logger.info(f"GPUDevicePool detected {count} CUDA GPU devices: {self.devices}")
        else:
            self.devices = ["cpu"]
            logger.info("GPUDevicePool detected 0 CUDA GPUs. Falling back to 'cpu'.")

    def get_next_device(self) -> str:
        """Retrieves next target CUDA device using round-robin scheduling."""
        if not self.devices:
            return "cpu"
        device = self.devices[self.current_idx % len(self.devices)]
        self.current_idx += 1
        return device

    def get_pool_status(self) -> List[Dict[str, Any]]:
        """Returns hardware memory usage and status for all devices in pool."""
        status = []
        for dev in self.devices:
            if dev.startswith("cuda") and torch.cuda.is_available():
                idx = int(dev.split(":")[-1])
                allocated_mb = torch.cuda.memory_allocated(idx) / (1024 * 1024)
                reserved_mb = torch.cuda.memory_reserved(idx) / (1024 * 1024)
                name = torch.cuda.get_device_name(idx)
                status.append({
                    "device": dev,
                    "name": name,
                    "allocated_mb": round(allocated_mb, 2),
                    "reserved_mb": round(reserved_mb, 2),
                    "status": "online",
                })
            else:
                status.append({
                    "device": "cpu",
                    "name": "CPU Host Processor",
                    "allocated_mb": 0.0,
                    "reserved_mb": 0.0,
                    "status": "online",
                })
        return status


# Global GPU Pool Singleton
global_gpu_pool = GPUDevicePool()
