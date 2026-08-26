"""Shared site/analysis configuration.

Generalizes Scripts/04_SPD.py's Config dataclass so the same object can be
passed to paleopy.spd / paleopy.human_climate / paleopy.ccsi, eliminating
the "manually keep these constants in sync across 3 files" footgun called
out in the original scripts' own comments (SITE_NAME/SITE_LAT/SITE_LON/
ARCHO_RADIUS_KM/TIME_MIN/TIME_MAX had to be hand-copied identically into
Scripts/04, 05, and 06).

Defaults match the Catalhoyuk case study used throughout the original
repo, so constructing Config() with no arguments reproduces the current
single-site behavior unchanged.

Fields specific to only one of the three downstream scripts (e.g. script
05/06's ENV_RADIUS_KM, PANGAEA/Neotoma proxy settings, PCA parameters)
are added when that script is actually ported (Stages 8/9), rather than
speculatively included here ahead of need.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    site_name: str = "Catalhoyuk"
    site_lat: float = 37.6660  # decimal degrees, positive = North
    site_lon: float = 32.8277  # decimal degrees, positive = East
    archaeo_radius_km: float = 5  # search radius around the site coordinates (km)
    env_radius_km: float = 1000  # search radius for paleoenvironmental proxies (05/06 only)

    time_min: int = 7300  # Cal BP - younger end of analysis window
    time_max: int = 9500  # Cal BP - older end of analysis window

    max_error: int = 100  # maximum accepted lab 14C error (yr)
    bin_h: int = 100  # temporal binning threshold (yr)
    resolution: int = 10  # SPD grid step size (cal yr)
    n_simulations: int = 5000  # Monte Carlo iterations for NHST envelope

    # -- 05/06-specific fields (added when those scripts were ported) --
    confirm: bool = True
    use_gisp2_only: bool = True
    env_proxy_keyword: str = "stable isotope"
    env_neotoma_type: list = field(default_factory=lambda: ["stable isotopes"])
    env_bin_yr: float = 10  # should generally equal `resolution`
    mc_n_sim: int = 999
    mc_model: str = "logistic"  # "exponential" or "uniform"
    anomaly_threshold: float = -1.0
    recovery_steps: int = 10
    baseline_window: float = 400

    # -- 06-specific fields (CCSI/PCA pipeline) --
    kernel_sigma_yr: float = 50
    max_interp_gap_bins: int = 10
    use_gisp2: bool = True
    pangaea_keywords: list = field(default_factory=lambda: ["stable isotope", "speleothem", "pollen"])
    neotoma_types: list = field(default_factory=lambda: ["pollen", "stable isotopes"])
    min_proxies_for_pca: int = 2
    em_pca_max_iter: int = 50
    em_pca_tol: float = 1e-4
    warn_neff_threshold: float = 30
    resilience_cv_threshold: float = 0.5

    outdir: Path = field(default_factory=Path.cwd)

    def __post_init__(self):
        self.outdir = Path(self.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        if self.time_min >= self.time_max:
            raise ValueError("time_min must be less than time_max")

    @property
    def slug(self) -> str:
        return self.site_name.replace(" ", "_")
