"""Golden/snapshot tests: verify the paleopy package reproduces the same
output as the original Scripts/ pipeline on real data.

Marked @pytest.mark.golden since they need the real (large) data files
present in Datasets/ / "Catalhoyuk Data/" and can be slow.
"""

import os

import pandas as pd
import pytest

from paleopy.cleaning import clean, get_records

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS = os.path.join(REPO_ROOT, "Datasets")


@pytest.mark.golden
def test_clean_reproduces_committed_cleaned_dataset():
    raw_path = os.path.join(DATASETS, "p3k14c_dataset.csv")
    labs_path = os.path.join(DATASETS, "Labs.csv")
    golden_cleaned_path = os.path.join(DATASETS, "cleaned_p3k14c.csv")
    golden_graveyard_path = os.path.join(DATASETS, "graveyard.csv")

    if not all(os.path.isfile(p) for p in (raw_path, labs_path, golden_cleaned_path, golden_graveyard_path)):
        pytest.skip("Real Datasets/ CSVs not present")

    original = get_records(raw_path)
    cleaned, graveyard, unknown_codes = clean(original, labs_path=labs_path, family_tree_path=None)

    golden_cleaned = pd.read_csv(golden_cleaned_path, low_memory=False, index_col=0)
    golden_graveyard = pd.read_csv(golden_graveyard_path, low_memory=False)

    assert len(cleaned) == len(golden_cleaned), (
        f"row count mismatch: got {len(cleaned)}, golden has {len(golden_cleaned)}"
    )
    assert list(cleaned.index) == list(golden_cleaned.index), "LabID index order/content differs"
    assert list(cleaned.columns) == list(golden_cleaned.columns), "column set/order differs"

    assert len(graveyard) == len(golden_graveyard), (
        f"graveyard row count mismatch: got {len(graveyard)}, golden has {len(golden_graveyard)}"
    )
