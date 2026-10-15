"""Shared HTTP helpers.

Replaces two patterns from the original scripts:

- Scripts/05 and 06 retried a failed HTTPS request with ``verify=False``,
  and Scripts/07 disabled certificate verification process-wide at import
  time. paleopy never disables verification: ``fetch`` always verifies
  certificates and lets failures raise.
- Scripts/07 also called ``sys.setrecursionlimit(5000)`` globally at import
  time. ``temporary_recursion_limit`` scopes that to just the call that
  needs it (cartopy's Natural Earth geometry in paleopy.mapping), restoring
  the previous limit afterward.
"""

import contextlib
import sys

import requests


def fetch(url: str, **kwargs) -> requests.Response:
    """GET a URL with certificate verification, raising on any failure."""
    resp = requests.get(url, **kwargs)
    resp.raise_for_status()
    return resp


@contextlib.contextmanager
def temporary_recursion_limit(limit: int):
    """Temporarily raise sys.setrecursionlimit, restoring it afterward."""
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(limit)
    try:
        yield
    finally:
        sys.setrecursionlimit(previous)
