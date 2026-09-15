from __future__ import annotations

import time

from tars_agent.web.auth import LocalWebAuth


def test_bootstrap_token_is_single_use_and_session_cookie_authenticates() -> None:
    auth, bootstrap_token = LocalWebAuth.issue()
    cookie = auth.exchange(bootstrap_token)
    assert cookie is not None
    assert auth.authenticated(cookie)
    assert auth.exchange(bootstrap_token) is None
    assert not auth.authenticated("wrong")
    assert not hasattr(auth, "bootstrap_token")


def test_bootstrap_token_expires(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    started = time.monotonic()
    monkeypatch.setattr("tars_agent.web.auth.time.monotonic", lambda: started)
    auth, bootstrap_token = LocalWebAuth.issue(ttl_s=60)
    monkeypatch.setattr("tars_agent.web.auth.time.monotonic", lambda: started + 61)
    assert auth.exchange(bootstrap_token) is None
