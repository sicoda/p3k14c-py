import socket

import pytest

import paleopy


def test_package_imports():
    assert paleopy.__version__ == "0.1.0"


def test_no_network_fixture_blocks_sockets():
    with pytest.raises(RuntimeError):
        socket.socket().connect(("example.com", 80))
