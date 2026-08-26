import importlib.util
import socket
import sys

import numpy as np
import pytest

from paleopy.calibration import calibrate_gaussian_intcal, load_intcal20


def test_load_intcal20_import_has_no_network_side_effect(monkeypatch):
    """paleopy.calibration must not download anything merely on import —
    unlike the original scripts, which called load_intcal20() at module
    import time.
    """
    def blocked_connect(*a, **k):
        raise RuntimeError("unexpected network access on import")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    spec = importlib.util.spec_from_file_location("paleopy_calibration_reimport", "src/paleopy/calibration.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # should not raise


def test_load_intcal20_uses_existing_cache():
    # intcal20.npz is expected to already be cached in the repo root from
    # earlier stages of this migration (or downloaded once here).
    cal_bp, c14_age, c14_error = load_intcal20()
    assert len(cal_bp) > 1000
    assert np.all(np.diff(cal_bp) >= 0)  # sorted ascending


def test_calibrate_gaussian_intcal_parity():
    spec = importlib.util.spec_from_file_location("orig05_calib_test", "Scripts/05_Human_Climate_Interaction.py")
    orig = importlib.util.module_from_spec(spec)
    sys.modules["orig05_calib_test"] = orig
    spec.loader.exec_module(orig)  # triggers orig's own load_intcal20() at import (uses cache)

    years = np.arange(7300, 9500, 10, dtype=float)
    for age, err in [(8500, 40), (8000, 100)]:
        a = orig.calibrate_date(age, err, years)
        b = calibrate_gaussian_intcal(age, err, years)
        np.testing.assert_allclose(a, b)


def test_calibrate_gaussian_intcal_normalizes_to_sum_one():
    years = np.arange(7300, 9500, 10, dtype=float)
    prob = calibrate_gaussian_intcal(8500, 40, years)
    assert prob.sum() == pytest.approx(1.0, rel=1e-6)
