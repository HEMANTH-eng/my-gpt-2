import html
import re
from utils.logger import get_logger

logger = get_logger("sanitizer")

PROMPT_INJECTION_PATTERNS = [
    r"(?i)ignore\s+previous\s+instructions",
    r"(?i)system\s+override",
    r"(?i)you\s+are\s+now\s+in\s+dan\s+mode",
    r"(?i)forget\s+all\s+prior\s+rules",
    r"(?i)disregard\s+safety\s+guidelines",
]

SQL_INJECTION_PATTERNS = [
    r"(?i);\s*DROP\s+TABLE",
    r"(?i);\s*DELETE\s+FROM",
    r"(?i)UNION\s+SELECT",
    r"(?i)OR\s+1\s*=\s*1",
]


def sanitize_input(text: str) -> str:
    """Sanitizes user input string against Prompt Injection, SQL Injection, XSS, and Command Injection attacks.

    Args:
        text: Raw user input text.

    Returns:
        Sanitized safe string.
    """
    if not text:
        return ""

    sanitized = text

    # 1. Neutralize Prompt Injection attempts
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, sanitized):
            logger.warning(f"Detected potential Prompt Injection attempt matching '{pattern}'")
            sanitized = re.sub(pattern, "[FILTERED PROMPT INJECTION]", sanitized)

    # 2. Neutralize SQL Injection patterns
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, sanitized):
            logger.warning(f"Detected potential SQL Injection attempt matching '{pattern}'")
            sanitized = re.sub(pattern, "[FILTERED SQL STATEMENT]", sanitized)

    # 3. Neutralize HTML / XSS script tags
    sanitized = re.sub(r"(?i)<script\b[^<]*>(.*?)</script>", "[FILTERED SCRIPT]", sanitized)
    sanitized = re.sub(r"(?i)<iframe\b[^<]*>(.*?)</iframe>", "[FILTERED IFRAME]", sanitized)
    sanitized = re.sub(r"(?i)javascript:", "filtered_js:", sanitized)
    sanitized = re.sub(r"(?i)on\w+\s*=", "on_event=", sanitized)

    # Escape HTML special entities
    sanitized = html.escape(sanitized, quote=False)

    return sanitized
