"""Shared HTTP/SSL helpers.

Replaces two patterns from the original scripts that mutated global/process
state as a side effect of being imported or called:

- Scripts/05 and 06 each retried a failed verified HTTPS request with
  ``verify=False`` inline, per call site. ``fetch_with_fallback`` centralizes
  that (still falling back on failure, but as one explicit, reusable call
  with a loud warning) rather than duplicating it in every fetcher.
- Scripts/07 set ``ssl._create_default_https_context =
  ssl._create_unverified_context`` and ``sys.setrecursionlimit(5000)``
  globally at *module import time*, silently disabling certificate
  verification for the entire process for as long as it was imported.
  ``temporary_unverified_ssl`` and ``temporary_recursion_limit`` scope both
  to just the call that needs them (e.g. cartopy's Natural Earth downloader
  in paleopy.mapping), restoring the previous state afterward.
"""

import contextlib
import ssl
import sys
import warnings

import requests


def fetch_with_fallback(url: str, **kwargs) -> requests.Response:
    """GET a URL with certificate verification; retry unverified on failure.

    Mirrors the original scripts' "read-only public data" fallback, but as
    one shared, explicit call instead of copy-pasted per fetcher. Emits a
    warning whenever the unverified fallback is actually used.
    """
    try:
        resp = requests.get(url, **kwargs)
        resp.raise_for_status()
        return resp
    except Exception as exc:
        warnings.warn(
            f"Verified HTTPS request to {url} failed ({exc}); "
            "retrying without certificate verification."
        )
        kwargs["verify"] = False
        resp = requests.get(url, **kwargs)
        resp.raise_for_status()
        return resp


@contextlib.contextmanager
def temporary_unverified_ssl():
    """Temporarily disable HTTPS certificate verification process-wide.

    Only for use around a specific call (e.g. a library's internal
    downloader that doesn't accept a ``verify=`` kwarg, such as cartopy's
    Natural Earth fetcher) — never left enabled for the life of the
    process. Restores the previous context afterward.
    """
    previous = ssl._create_default_https_context
    warnings.warn("Temporarily disabling HTTPS certificate verification for this call.")
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        yield
    finally:
        ssl._create_default_https_context = previous


@contextlib.contextmanager
def temporary_recursion_limit(limit: int):
    """Temporarily raise sys.setrecursionlimit, restoring it afterward."""
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(limit)
    try:
        yield
    finally:
        sys.setrecursionlimit(previous)
