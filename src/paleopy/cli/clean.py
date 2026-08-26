"""paleopy-clean : cleans the raw p3k14c dataset.

CLI wrapper around paleopy.cleaning.clean(); mirrors
Scripts/01_Data_Cleaning_and_Prep.py's behavior with named arguments
instead of positional sys.argv parsing.
"""

import argparse
import sys

from paleopy.cleaning import clean, get_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paleopy-clean",
        description="Clean/scrub the raw p3k14c dataset.",
    )
    parser.add_argument("--input", required=True, help="Path to the raw p3k14c dataset CSV")
    parser.add_argument("--output", required=True, help="Path to write the cleaned CSV")
    parser.add_argument("--graveyard", required=True, help="Path to write the graveyard (removed records) CSV")
    parser.add_argument("--unknown-codes", required=True, help="Path to write the unknown lab codes CSV")
    parser.add_argument("--labs", default=None, help="Path to Labs.csv (lab-code reference table)")
    parser.add_argument("--family-tree", default=None, help="Path to DatasetFamilyTree.csv (optional)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    original = get_records(args.input)
    n_in = len(original)

    records, graveyard, unknown_codes = clean(
        original, labs_path=args.labs, family_tree_path=args.family_tree
    )

    print(f"\n[scrub] Saving → {args.graveyard}")
    graveyard.to_csv(args.graveyard, sep=",", encoding="utf-8")

    print(f"[scrub] Saving → {args.output}")
    records.to_csv(args.output, sep=",", encoding="utf-8")

    unknown_codes.to_csv(args.unknown_codes, index=False, encoding="utf-8")

    n_out = len(records)
    n_removed = n_in - n_out
    pct = n_removed / n_in * 100 if n_in else 0.0

    print(f"\n{'=' * 60}")
    print(f"  Input    : {n_in:>7,} records")
    print(f"  Output   : {n_out:>7,} records")
    print(f"  Removed  : {n_removed:>7,} records  ({pct:.1f}%)")
    print(f"  Cleaned  : {args.output}")
    print(f"  Graveyard: {args.graveyard}")
    print(f"  Unknown  : {args.unknown_codes}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    sys.exit(main())
