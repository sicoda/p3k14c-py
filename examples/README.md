# paleopy examples

Seven scripts, one per pipeline stage, replicating the original `Scripts/01`–`07` case studies by calling the `paleopy` API directly instead of the `paleopy-*` console scripts. Useful as a reference for integrating `paleopy` into your own code rather than shelling out to the CLI.

Run any of them from the repo root with `paleopy` installed (`pip install -e .[all]`):

```bash
python examples/01_clean.py
python examples/02_calibrate.py
python examples/03_summary.py
python examples/04_spd.py
python examples/05_climate.py
python examples/06_ccsi.py
python examples/07_kde.py
```

All outputs are written to `examples_output/` (gitignored). Some scripts reuse an earlier script's output for speed (e.g. `05_climate.py` and `06_ccsi.py` prefer the repo's existing `Catalhoyuk Data/Catalhoyuk_spd_for_06.csv` over rebuilding the SPD from scratch) — delete that file first if you want to see the from-scratch path.

Notes:
- `02_calibrate.py` calibrates only the first 200 rows for speed; drop the `.head(200)` to calibrate the full dataset (takes hours).
- `04_spd.py`/`05_climate.py` use small Monte Carlo iteration counts (100) instead of the originals' publication defaults (5000/999) so they run in seconds — raise them for real analysis.
- `06_ccsi.py` and `07_kde.py` make live network calls (Neotoma site lookup, GISP2 download, cartopy Natural Earth download) the first time they run; subsequent runs reuse cached files.
