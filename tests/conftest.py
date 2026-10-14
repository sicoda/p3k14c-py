import importlib.util
import os
import socket
import sys

import pytest

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Scripts")


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


@pytest.fixture(scope="session")
def legacy_script():
    """Return a loader for the original Scripts/ modules used by parity tests.

    Scripts/ is kept in the repo for reference but not shipped in the sdist,
    so any test depending on it is skipped when the directory is absent.
    """
    def load(filename, module_name, argv=None):
        path = os.path.join(SCRIPTS_DIR, filename)
        if not os.path.isfile(path):
            pytest.skip(f"legacy script {filename} not present (Scripts/ not shipped)")
        spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(spec)
        if argv is not None:
            sys.argv = argv
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod

    return load
