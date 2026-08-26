import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(request, monkeypatch):
    """Fail any un-mocked network call so unit tests never hit the internet.

    Tests that legitimately need network access (recorded-fixture based
    tests that monkeypatch requests.get/etc. themselves) should mark
    themselves with @pytest.mark.golden or @pytest.mark.network to opt out.
    """
    if request.node.get_closest_marker("golden") or request.node.get_closest_marker("network"):
        yield
        return

    def blocked_connect(*args, **kwargs):
        raise RuntimeError("network access attempted during a test without the 'network'/'golden' marker")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    yield


@pytest.fixture
def tmp_outdir(tmp_path):
    return tmp_path
