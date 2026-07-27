from collections import OrderedDict
import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple

from utils.logger import get_logger

logger = get_logger("response_cache")


class ResponseCache:
    """High-speed response caching engine with SHA-256 key hashing and TTL expiration."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600) -> None:
        """Initializes ResponseCache.

        Args:
            max_size: Maximum number of cached items (LRU eviction).
            default_ttl: Expiration time in seconds (default 1 hour).
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _generate_key(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Computes deterministic SHA-256 hash key for prompt and parameters."""
        parameters = parameters or {}
        key_payload = {
            "prompt": prompt.strip(),
            "params": {k: str(v) for k, v in sorted(parameters.items())},
        }
        raw_bytes = json.dumps(key_payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def get(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Retrieves cached response if key exists and has not expired."""
        key = self._generate_key(prompt, parameters)
        if key in self.cache:
            value, expire_at = self.cache[key]
            if time.time() < expire_at:
                self.cache.move_to_end(key)
                self.hits += 1
                logger.info(f"Cache HIT for key hash '{key[:12]}...' (Hit ratio: {self.hit_ratio:.1f}%)")
                return value
            else:
                logger.info(f"Cache EXPIRED for key hash '{key[:12]}...'")
                del self.cache[key]

        self.misses += 1
        return None

    def set(self, prompt: str, value: Any, parameters: Optional[Dict[str, Any]] = None, ttl: Optional[int] = None) -> None:
        """Stores value in cache with TTL expiration and LRU eviction."""
        key = self._generate_key(prompt, parameters)
        ttl = ttl or self.default_ttl
        expire_at = time.time() + ttl

        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = (value, expire_at)

        if len(self.cache) > self.max_size:
            evicted_key, _ = self.cache.popitem(last=False)
            logger.info(f"Cache LRU evicted oldest key hash '{evicted_key[:12]}...'")

    def clear(self) -> None:
        """Clears all cached entries and resets statistics."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_ratio(self) -> float:
        """Returns cache hit percentage (0.0 to 100.0%)."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100.0


# Global singleton instance
global_response_cache = ResponseCache()
