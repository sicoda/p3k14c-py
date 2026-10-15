import sys

import pytest
import requests

from paleopy.net import fetch, temporary_recursion_limit


@pytest.mark.network
def test_fetch_returns_verified_response_when_it_succeeds(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

    calls = []

    def fake_get(url, **kwargs):
        calls.append(kwargs)
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    resp = fetch("https://example.com/data")
    assert isinstance(resp, FakeResponse)
    assert len(calls) == 1
    assert "verify" not in calls[0]


@pytest.mark.network
def test_fetch_does_not_retry_unverified_on_ssl_failure(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(kwargs)
        raise requests.exceptions.SSLError("cert failure")

    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(requests.exceptions.SSLError):
        fetch("https://example.com/data")
    assert len(calls) == 1
    assert calls[0].get("verify", True) is not False


def test_temporary_recursion_limit_restores_previous_limit():
    previous = sys.getrecursionlimit()
    with temporary_recursion_limit(previous + 1000):
        assert sys.getrecursionlimit() == previous + 1000
    assert sys.getrecursionlimit() == previous
