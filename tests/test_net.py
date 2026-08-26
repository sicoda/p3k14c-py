import ssl
import sys

import pytest
import requests

from paleopy.net import fetch_with_fallback, temporary_recursion_limit, temporary_unverified_ssl


@pytest.mark.network
def test_fetch_with_fallback_returns_verified_response_when_it_succeeds(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

    calls = []

    def fake_get(url, **kwargs):
        calls.append(kwargs)
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    resp = fetch_with_fallback("https://example.com/data")
    assert isinstance(resp, FakeResponse)
    assert len(calls) == 1
    assert "verify" not in calls[0]


@pytest.mark.network
def test_fetch_with_fallback_retries_unverified_on_failure(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

    calls = []

    def fake_get(url, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise requests.exceptions.SSLError("cert failure")
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.warns(UserWarning):
        resp = fetch_with_fallback("https://example.com/data")
    assert isinstance(resp, FakeResponse)
    assert len(calls) == 2
    assert calls[1]["verify"] is False


def test_temporary_unverified_ssl_restores_previous_context():
    previous = ssl._create_default_https_context
    with temporary_unverified_ssl():
        assert ssl._create_default_https_context is ssl._create_unverified_context
    assert ssl._create_default_https_context is previous


def test_temporary_recursion_limit_restores_previous_limit():
    previous = sys.getrecursionlimit()
    with temporary_recursion_limit(previous + 1000):
        assert sys.getrecursionlimit() == previous + 1000
    assert sys.getrecursionlimit() == previous
