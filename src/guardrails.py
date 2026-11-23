"""
Safety Guardrails and Input Sanitization Module (Task U5).
Provides prompt sanitization, boundary clamping, class whitelisting,
in-memory rate limiting, and out-of-distribution refusal checking.
"""

import re
import time
from typing import Dict, Any, Tuple, Optional, List
from src.utils import FASHION_CLASSES


class RateLimiter:
    """
    Sliding window rate limiter to protect free API tiers and CPU resources.
    Limits requests per client IP / session ID within a time window.
    """
    def __init__(self, max_requests: int = 15, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.history: Dict[str, List[float]] = {}

    def is_allowed(self, client_id: str = "default_user") -> Tuple[bool, str]:
        now = time.time()
        if client_id not in self.history:
            self.history[client_id] = []

        # Prune expired timestamps
        self.history[client_id] = [
            t for t in self.history[client_id] if now - t < self.window_seconds
        ]

        if len(self.history[client_id]) >= self.max_requests:
            remaining = int(self.window_seconds - (now - self.history[client_id][0]))
            return False, f"Rate limit reached ({self.max_requests} req/{self.window_seconds}s). Retry in {remaining}s."

        self.history[client_id].append(now)
        return True, "Allowed"


def sanitize_input(text: str, max_chars: int = 150) -> Tuple[bool, str, str]:
    """
    Sanitizes user input string:
    - Rejects empty or excessively long inputs
    - Strips harmful HTML/script tags and shell characters
    - Returns (is_valid, cleaned_text, error_message)
    """
    if not text or not text.strip():
        return False, "", "Input prompt cannot be empty."

    trimmed = text.strip()
    if len(trimmed) > max_chars:
        return False, "", f"Input exceeds maximum allowed length of {max_chars} characters."

    # Remove HTML/XML tags
    cleaned = re.sub(r"<[^>]*>", "", trimmed)

    # Disallow shell metacharacters
    dangerous_patterns = [r"[;&|`$]", r"\b(exec|eval|system|rm|cat|drop)\b"]
    for pat in dangerous_patterns:
        if re.search(pat, cleaned, re.IGNORECASE):
            return False, "", "Input contains disallowed special characters or command sequences."

    return True, cleaned, ""


def validate_guardrails(class_label: str, count: int) -> Tuple[bool, str]:
    """
    Verifies that the extracted class and count conform strictly to Fashion-MNIST constraints.
    """
    if class_label not in FASHION_CLASSES:
        return False, f"Class '{class_label}' is not in the Fashion-MNIST whitelist: {', '.join(FASHION_CLASSES)}."

    if not isinstance(count, int) or count < 1 or count > 16:
        return False, f"Count ({count}) is outside the permitted boundary [1, 16]."

    return True, "Valid"
