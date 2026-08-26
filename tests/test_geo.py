import math

from paleopy.geo import haversine_km


def test_haversine_zero_distance_for_identical_points():
    assert haversine_km(37.666, 32.8277, 37.666, 32.8277) == 0.0


def test_haversine_known_distance_paris_to_london():
    # Paris (48.8566, 2.3522) to London (51.5074, -0.1278): ~343-344 km great-circle
    d = haversine_km(48.8566, 2.3522, 51.5074, -0.1278)
    assert math.isclose(d, 343.5, rel_tol=0.02)


def test_haversine_symmetric():
    a = haversine_km(37.666, 32.8277, 48.8566, 2.3522)
    b = haversine_km(48.8566, 2.3522, 37.666, 32.8277)
    assert math.isclose(a, b, rel_tol=1e-9)
