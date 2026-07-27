import json
from pathlib import Path
import time
from typing import Any, Dict, Optional
from utils.logger import get_logger

logger = get_logger("audit")

AUDIT_LOG_PATH = Path("audit.log")


class AuditLogger:
    """Structured JSON Security Audit Logger."""

    def __init__(self, log_path: Path = AUDIT_LOG_PATH) -> None:
        self.log_path = log_path

    def log_event(
        self,
        action: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        resource: Optional[str] = None,
        status: str = "success",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Appends a structured JSON security audit record to audit.log.

        Args:
            action: Action performed (e.g. 'user_login', 'api_key_create', 'agent_execute').
            user_id: Optional user identifier.
            ip_address: Optional client IP address.
            resource: Optional target resource URI.
            status: Event outcome ('success', 'denied', 'failed').
            details: Optional payload metadata dictionary.

        Returns:
            The written audit log dictionary.
        """
        audit_record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": action,
            "user_id": user_id or "anonymous",
            "ip_address": ip_address or "127.0.0.1",
            "resource": resource or "api",
            "status": status,
            "details": details or {},
        }

        log_entry_str = json.dumps(audit_record)
        logger.info(f"AUDIT LOG [{status.upper()}]: {action} by user '{audit_record['user_id']}'")

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(log_entry_str + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log entry: {e}")

        return audit_record


# Global Singleton Audit Logger
global_audit_logger = AuditLogger()
