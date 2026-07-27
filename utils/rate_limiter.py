from collections import defaultdict
import time
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("rate_limiter")


class SlidingWindowRateLimiter:
    """Sliding-window rate limiter per client IP address or API Key."""

    def __init__(self, default_limit: int = 60, window_seconds: int = 60) -> None:
        """Initializes RateLimiter.

        Args:
            default_limit: Maximum requests allowed per window (default 60 req/min).
            window_seconds: Window size in seconds (default 60s).
        """
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_id: str, limit: Optional[int] = None) -> bool:
        """Checks if request from client_id is permitted under rate limit.

        Args:
            client_id: Client identifier (IP address, API Key prefix, or User ID).
            limit: Optional custom rate limit for client.

        Returns:
            True if permitted, False if rate limit exceeded.
        """
        now = time.time()
        max_requests = limit or self.default_limit
        cutoff = now - self.window_seconds

        # Prune timestamps older than window
        timestamps = [t for t in self.requests[client_id] if t > cutoff]
        self.requests[client_id] = timestamps

        if len(timestamps) < max_requests:
            self.requests[client_id].append(now)
            return True
        else:
            logger.warning(f"Rate limit EXCEEDED for client '{client_id}' ({len(timestamps)}/{max_requests} req/min)")
            return False

    def clear(self) -> None:
        """Clears all rate limit state tracking."""
        self.requests.clear()


# Global Singleton Rate Limiter
global_rate_limiter = SlidingWindowRateLimiter()
