"""Console/UX helpers shared by the paleopy CLI wrappers.

These are pure CLI concerns (printing, prompting, argparse groups) — no
domain module (paleopy.spd, paleopy.calibration, etc.) needs them, only
the paleopy.cli.* entry points. Ported from Scripts/04_SPD.py's
_divider/_warn/_abort/_confirm_or_abort, generalized for reuse by
paleopy-spd/paleopy-climate/paleopy-ccsi so their site-scoped arguments
can't drift out of sync with each other (the exact footgun the original
scripts' own comments warned about).
"""

import argparse
import sys

from paleopy.config import Config


def divider(title: str = "", width: int = 72) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "-" * pad + f" {title} " + "-" * (width - pad - len(title) - 2))
    else:
        print("\n" + "-" * width)


def warn(msg: str) -> None:
    print(f"\n  WARNING: {msg}")


def abort(msg: str) -> None:
    sys.exit(f"\n  ERROR: {msg}\n")


def confirm_or_abort(prompt: str) -> None:
    print(f"\n  {prompt} [yes/no] ", end="", flush=True)
    if input().strip().lower() not in ("yes", "y", ""):
        abort("Aborted by user.")


def add_site_arguments(parser: argparse.ArgumentParser, defaults: Config = None) -> None:
    """Adds the shared --site-name/--site-lat/--site-lon/--radius-km/
    --time-min/--time-max/--max-error/--bin-h argument group used by
    paleopy-spd, paleopy-climate, and paleopy-ccsi, so the three CLIs
    can't define these independently and drift apart.
    """
    d = defaults or Config()
    g = parser.add_argument_group("site")
    g.add_argument("--site-name", default=d.site_name, help=f"Site label used in outputs/plots [default: {d.site_name}]")
    g.add_argument("--site-lat", type=float, default=d.site_lat, help=f"Site latitude, + = North [default: {d.site_lat}]")
    g.add_argument("--site-lon", type=float, default=d.site_lon, help=f"Site longitude, + = East [default: {d.site_lon}]")
    g.add_argument("--radius-km", type=float, default=d.archaeo_radius_km, help=f"Search radius around coordinates (km) [default: {d.archaeo_radius_km}]")

    g2 = parser.add_argument_group("time window and data quality")
    g2.add_argument("--time-min", type=int, default=d.time_min, help=f"Younger Cal BP boundary [default: {d.time_min}]")
    g2.add_argument("--time-max", type=int, default=d.time_max, help=f"Older Cal BP boundary [default: {d.time_max}]")
    g2.add_argument("--max-error", type=int, default=d.max_error, help=f"Max accepted lab 14C error (yr) [default: {d.max_error}]")
    g2.add_argument("--bin-h", type=int, default=d.bin_h, help=f"Temporal bin width (yr) [default: {d.bin_h}]")
