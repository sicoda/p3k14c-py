"""Example: clean the raw p3k14c dataset using the paleopy API directly.

Replicates Scripts/01_Data_Cleaning_and_Prep.py without going through the
paleopy-clean console script. Run from the repo root:

    python examples/01_clean.py
"""

from pathlib import Path

from paleopy.cleaning import clean, get_records

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "examples_output"
OUTDIR.mkdir(exist_ok=True)


def main() -> None:
    raw_path = REPO_ROOT / "Datasets" / "p3k14c_dataset.csv"
    labs_path = REPO_ROOT / "Datasets" / "Labs.csv"

    original = get_records(str(raw_path))
    cleaned, graveyard, unknown_codes = clean(original, labs_path=str(labs_path), family_tree_path=None)

    cleaned.to_csv(OUTDIR / "cleaned_p3k14c.csv")
    graveyard.to_csv(OUTDIR / "graveyard.csv")
    unknown_codes.to_csv(OUTDIR / "unknown_codes.csv", index=False)

    n_in, n_out = len(original), len(cleaned)
    print(f"\nInput: {n_in:,}  Output: {n_out:,}  Removed: {n_in - n_out:,} ({(n_in - n_out) / n_in:.1%})")
    print(f"Wrote cleaned_p3k14c.csv, graveyard.csv, unknown_codes.csv -> {OUTDIR}")


if __name__ == "__main__":
    main()
