from __future__ import annotations

import hashlib
import hmac
import secrets
import time


class LocalWebAuth:
    """Process-local authentication that retains only token digests."""

    __slots__ = (
        "_bootstrap_digest",
        "_consumed",
        "_expires_at",
        "_session_digest",
    )

    def __init__(self, bootstrap_token: str, *, ttl_s: float = 60.0) -> None:
        self._bootstrap_digest = _digest(bootstrap_token)
        self._expires_at = time.monotonic() + ttl_s
        self._consumed = False
        self._session_digest: bytes | None = None

    @classmethod
    def issue(cls, *, ttl_s: float = 60.0) -> tuple[LocalWebAuth, str]:
        # token_urlsafe(32) starts with 32 random bytes (256 bits). The raw token
        # is returned once to the launcher and is never retained by this object.
        token = secrets.token_urlsafe(32)
        return cls(token, ttl_s=ttl_s), token

    def exchange(self, token: str) -> str | None:
        candidate = _digest(token)
        matches = hmac.compare_digest(candidate, self._bootstrap_digest)
        valid = (
            matches
            and not self._consumed
            and time.monotonic() <= self._expires_at
        )
        if not valid:
            return None
        self._consumed = True
        session_token = secrets.token_urlsafe(32)
        self._session_digest = _digest(session_token)
        return session_token

    def authenticated(self, session_token: str | None) -> bool:
        if not session_token or self._session_digest is None:
            return False
        return hmac.compare_digest(_digest(session_token), self._session_digest)


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


__all__ = ["LocalWebAuth"]
