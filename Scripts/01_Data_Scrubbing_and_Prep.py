import os
import re
import sys
import numpy as np
import pandas as pd

os.chdir("/Users/daniellesicotte/p3k14c_py")
 
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):   # type: ignore[misc]
        return iterable
 
# ---------------------------------------------------------------------------
# Optional ftfy — stdlib fallback when not installed
# ---------------------------------------------------------------------------
try:
    import ftfy as _ftfy
    def fix_encoding(text: str) -> str:
        return _ftfy.fix_encoding(str(text))
except ImportError:
    def fix_encoding(text: str) -> str:          # type: ignore[misc]
        """
        Minimal mojibake repair: re-encode as latin-1 then decode as utf-8.
        Covers the most common corruption pattern produced by Excel.
        """
        try:
            return str(text).encode('latin-1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return str(text)
 
pd.options.mode.chained_assignment = None
 
# ---------------------------------------------------------------------------
# Constants  (mirror common.py column-name constants from the original repo)
# ---------------------------------------------------------------------------
LAB_ID        = 'LabID'
LAT           = 'Lat'
LON           = 'Long'
AGE           = 'Age'
STD_DEV       = 'Error'
LOC_ACCURACY  = 'LocAccuracy'
SOURCE        = 'Source'
PROVINCE      = 'Province'
 
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
LAB_CODE_FILE = os.path.join(_SCRIPT_DIR, 'Labs.csv')
FAMILY_TREE_FILE = os.path.join(_SCRIPT_DIR, 'DatasetFamilyTree.csv')
 
# Records whose coordinates are hard-coded as erroneous in the original scrub.py
BAD_COORD_IDS = {'M-1900', 'M-2281', 'GXO-676', 'M-1483', 'M-1602', 'GaK-3896'}
 
# Text columns that may contain non-Latin characters needing encoding repair
ENCODING_COLS = ['SiteName', 'Country', 'Province', 'Continent', 'Source', 'Reference']
 
# Columns to sanitise (strip exotic quotes, commas, exotic whitespace)
SANITISE_COLS = [
    'Reference', 'Source', 'Province', 'Country',
    'SiteName', 'Period', 'Method', 'Taxa', 'Material',
]
 
# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
 
def is_nan(value) -> bool:
    """Robust NaN check that works on strings, floats, and None."""
    return str(value) == 'nan'
 
 
def flush_msg(msg: str) -> None:
    print(f'\r{msg}', end='', flush=True)
 
 
def concat_graveyard(graveyard: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """pandas 2.x-safe replacement for the deprecated DataFrame.append()."""
    if new_rows.empty:
        return graveyard
    return pd.concat([graveyard, new_rows], ignore_index=True)
 
 
def send_to_graveyard(
    df: pd.DataFrame,
    graveyard: pd.DataFrame,
    mask: pd.Series,
    reason: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Move all rows where *mask* is True into *graveyard* (tagging each with
    *reason*), drop them from *df*, and return the updated pair.
    """
    victims = df[mask].copy()
    if victims.empty:
        return df[~mask].copy(), graveyard
    victims['removal_reason'] = reason
    print(f'  Removed {len(victims):>6,}  →  {reason}')
    return df[~mask].copy(), concat_graveyard(graveyard, victims)
 
# ---------------------------------------------------------------------------
# Lab-code helpers  (mirror the original scrub.py's codeFromLabNum etc.)
# ---------------------------------------------------------------------------
 
# Characters to strip when extracting the lab-code prefix
_STRIP_FOR_CODE = str.maketrans(
    '', '',
    " \u00a0*\u2010_,()#/\u2019&?\uff1f"
)
 
 
def code_from_lab_num(labnum: str) -> str:
    """
    Extract the lower-case alphabetic prefix from a full lab number.
    Matches the original codeFromLabNum() exactly.
      'BETA-123456'  ->  'beta'
      'AA-1234'      ->  'aa'
    """
    prefix = str(labnum).split('-')[0]
    letters_only = ''.join(c for c in prefix if not c.isdigit())
    return letters_only.lower().translate(_STRIP_FOR_CODE)
 
 
def standardise_lab_id(labid: str) -> str:
    """
    Normalise a lab ID: keep only A-Z and 0-9 (uppercased), then insert a
    dash between the leading letter block and the trailing digit block.
      'beta123456'  ->  'BETA-123456'
    """
    cleaned = ''.join(
        c.upper() for c in str(labid)
        if c.upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )
    sep = None
    for i, c in enumerate(cleaned):
        if c.isdigit():
            sep = i
            break
    if sep is None:
        return cleaned           # no digits — return as-is
    return cleaned[:sep] + '-' + cleaned[sep:]
 
 
# Ordinary symbols that are stripped before the isalnum() Unicode corruption check
_ORDINARY_SYMS = set(
    '-\t/:.,"\' =&*°\\[]#+%<>»«–;\n()_'
    '0123456789'
)
 
 
def is_corrupted_unicode(text: str) -> bool:
    """
    Return True if *text* still contains non-alphanumeric characters after
    stripping all ordinary punctuation/symbols — a sign of garbled Unicode.
    """
    stripped = ''.join(c for c in str(text) if c not in _ORDINARY_SYMS)
    return not stripped.replace(' ', '').isalnum()
 
# ---------------------------------------------------------------------------
# Step 1  –  Load the input CSV
# ---------------------------------------------------------------------------
 
def get_records(file_path: str) -> pd.DataFrame:
    print(f'[scrub] Reading : {file_path}')
    df = pd.read_csv(
        file_path,
        encoding='utf-8',
        encoding_errors='replace',
        low_memory=False,
        dtype=str,
    )
    # Strip leading/trailing whitespace from every string column up front
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    print(f'[scrub] Loaded  : {len(df):,} records, {len(df.columns)} columns')
    return df
 
# ---------------------------------------------------------------------------
# Step 2  –  Lab-code validation and LabID standardisation
# ---------------------------------------------------------------------------
 
def delete_bad_labs(
    records: pd.DataFrame,
    graveyard: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
 
    print('\n[scrub] Step 2 – Lab-code validation')
 
    if not os.path.isfile(LAB_CODE_FILE):
        print(f'  WARNING: Labs.csv not found at {LAB_CODE_FILE}. '
              'Skipping lab-code validation.')
        records = records.set_index(LAB_ID)
        return records, graveyard
 
    labs_df = pd.read_csv(LAB_CODE_FILE)
 
    # All valid lower-case codes (including typo aliases)
    known_codes = set(labs_df['CODE'].str.lower().unique())
 
    # Typo table: CODE -> PARENT_CODE (only rows where PARENT_CODE is set)
    typo_df   = labs_df[labs_df['PARENT_CODE'].notna()].copy()
    typo_df   = typo_df.set_index('CODE')
    typo_codes = set(typo_df.index.str.lower())
 
    # 2a  Export unknown codes
    all_codes_in_file = records[LAB_ID].apply(code_from_lab_num).unique()
    unknown = [c for c in all_codes_in_file if c not in known_codes]
    pd.DataFrame(unknown, columns=['Code']).to_csv(
        'unknown_codes.csv', index=False, encoding='utf-8'
    )
    print(f'  Unknown lab codes : {len(unknown)}  → unknown_codes.csv')
 
    # 2b  Remove records with unknown lab code
    is_known = records[LAB_ID].apply(lambda x: code_from_lab_num(x) in known_codes)
    records, graveyard = send_to_graveyard(
        records, graveyard, ~is_known, 'Unknown lab ID'
    )
 
    # 2c  Fix known typos in lab IDs
    def replace_typo(labnum: str) -> str:
        code = code_from_lab_num(labnum)
        if code in typo_codes:
            parent = typo_df.at[code, 'PARENT_CODE']
            return str(labnum).lower().replace(code, str(parent))
        return labnum
 
    records[LAB_ID] = records[LAB_ID].apply(replace_typo)
 
    # 2d  Remove records with no numeral in the lab ID
    has_numeral = records[LAB_ID].apply(
        lambda s: any(c.isdigit() for c in str(s))
    )
    records, graveyard = send_to_graveyard(
        records, graveyard, ~has_numeral,
        'Known lab code but no numeric suffix (no digits in LabID)'
    )
 
    # 2e  Remove records containing '?' in the lab ID
    has_q = records[LAB_ID].str.contains('?', regex=False, na=False)
    records, graveyard = send_to_graveyard(
        records, graveyard, has_q, 'Question mark in LabID'
    )
 
    # 2f  Remove records with corrupted Unicode in the lab ID
    is_corrupt = records[LAB_ID].apply(is_corrupted_unicode)
    records, graveyard = send_to_graveyard(
        records, graveyard, is_corrupt, 'Corrupted Unicode characters in LabID'
    )
 
    # 2g  Strip leading whitespace, then standardise format
    records[LAB_ID] = records[LAB_ID].str.lstrip()
    records[LAB_ID] = records[LAB_ID].apply(standardise_lab_id)
 
    # Set LabID as the DataFrame index (matches original behaviour)
    records = records.set_index(LAB_ID)
    return records, graveyard
 
# ---------------------------------------------------------------------------
# Step 3  –  Coordinate format conversion
# ---------------------------------------------------------------------------
 
def _deg_min_sec_to_dec(coord: str, axis: str) -> float:
    """
    Convert a degree/minute/second string (the '*' char represents the degree
    symbol) to decimal degrees.  Mirrors the original degMinSecToDec().
    """
    coord = coord.replace(',', '')
    parts = coord.replace(' ', '').split('*')
    degrees = int(parts[0])
    rest    = parts[1].split("'")
    mins_s  = rest[0].rstrip('NESW')
    minutes = int(mins_s) if mins_s else 0
    seconds = 0
    if len(rest) > 1 and '"' in rest[1]:
        secs_s  = rest[1].split('"')[0]
        seconds = int(secs_s) if secs_s else 0
    factor = -1 if axis == 'long' else 1
    return factor * (degrees + minutes / 60.0 + seconds / 3600.0)
 
 
def _solheim_to_dec(coord: str) -> float:
    """Convert Solheim Northing/Easting format to decimal degrees."""
    parts    = coord.split(' ')
    combined = parts[0] + '*' + parts[1] + "'"
    return _deg_min_sec_to_dec(combined, 'solheim')
 
 
def _convert_coord(coord, axis: str):
    if isinstance(coord, float):
        return coord
    s = str(coord)
    if '*' in s:                        # degree/minute/second notation
        return _deg_min_sec_to_dec(s, axis)
    if s and s[-1] in ('N', 'E'):       # Solheim Northing/Easting
        return _solheim_to_dec(s)
    return s                            # already decimal string
 
 
def convert_coordinates(records: pd.DataFrame) -> pd.DataFrame:
    print('\n[scrub] Step 3 – Homogenising coordinate formats')
    records[LAT] = records[LAT].apply(lambda x: _convert_coord(x, 'lat'))
    records[LON] = records[LON].apply(lambda x: _convert_coord(x, 'long'))
    return records
 
# ---------------------------------------------------------------------------
# Step 4  –  SiteName / SiteID whitespace stripping
# ---------------------------------------------------------------------------
 
def strip_whitespace(records: pd.DataFrame) -> pd.DataFrame:
    for col in ('SiteName', 'SiteID'):
        if col in records.columns:
            records[col] = records[col].apply(
                lambda x: str(x).strip() if not is_nan(x) else x
            )
    return records
 
# ---------------------------------------------------------------------------
# Step 5  –  Duplicate removal
# ---------------------------------------------------------------------------
 
def _oldest_source(sources: list, family_tree: pd.DataFrame) -> str:
    """Return the source with no parent (the oldest/most-original dataset)."""
    for src in sources:
        if src in family_tree.index and is_nan(family_tree.at[src, 'ParentDatasets']):
            return src
    return sources[0]    # fallback
 
 
def handle_duplicates(
    records: pd.DataFrame,
    graveyard: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print('\n[scrub] Step 5 – Removing duplicate records')
 
    # Load optional family-tree for source-priority tiebreaking
    family_tree = None
    if os.path.isfile(FAMILY_TREE_FILE):
        family_tree = pd.read_csv(FAMILY_TREE_FILE, index_col='Dataset')
 
    dup_mask = records.index.duplicated(keep=False)
    n_dup    = dup_mask.sum()
 
    if n_dup == 0:
        print('  No duplicate LabIDs found.')
        return records, graveyard
 
    print(f'  {n_dup:,} rows share a LabID with at least one other row')
 
    keep_indices  = []
    remove_rows   = []
 
    for lab_id, group in records[dup_mask].groupby(level=0):
        if len(group) == 1:
            keep_indices.append(group.index[0])
            continue
 
        resolved = False
        if family_tree is not None and SOURCE in group.columns:
            sources = []
            for cell in group[SOURCE]:
                if not is_nan(cell):
                    sources.extend(s.strip() for s in str(cell).split(';'))
            valid = [s for s in dict.fromkeys(sources) if s in family_tree.index]
            if valid:
                best = _oldest_source(valid, family_tree)
                for idx, row in group.iterrows():
                    if not is_nan(row.get(SOURCE, '')) and best in str(row[SOURCE]):
                        keep_indices.append(idx)
                        resolved = True
                        break
                if not resolved:
                    keep_indices.append(group.index[0])
                    resolved = True
                for idx, row in group.iterrows():
                    if idx not in keep_indices:
                        remove_rows.append(row.rename(idx))
 
        if not resolved:
            keep_indices.append(group.index[0])
            for idx, row in group.iloc[1:].iterrows():
                remove_rows.append(row.rename(idx))
 
    if remove_rows:
        remove_df = pd.DataFrame(remove_rows)
        remove_df['removal_reason'] = 'Duplicate LabID'
        graveyard = concat_graveyard(graveyard, remove_df)
        print(f'  Removed {len(remove_df):>6,}  →  Duplicate LabID')
 
    records = records[~records.index.duplicated(keep='first')]
    return records, graveyard
 
# ---------------------------------------------------------------------------
# Step 6  –  Miscellaneous scrubbing
# ---------------------------------------------------------------------------
 
def _is_integer(x) -> bool:
    """Return True if x represents a whole number (integer-valued float ok)."""
    try:
        return float(x) == int(float(x))
    except (ValueError, TypeError):
        return False
 
 
def _to_float(x):
    return pd.to_numeric(x, errors='coerce')
 
 
def apply_misc_scrubbing(
    records: pd.DataFrame,
    graveyard: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print('\n[scrub] Step 6 – Miscellaneous scrubbing')
 
    # Null Age or Error
    null_ae = records[AGE].isna() | records[STD_DEV].isna()
    records, graveyard = send_to_graveyard(
        records, graveyard, null_ae, 'Null age and/or error'
    )
 
    # Age must be an integer value
    non_int_age = ~records[AGE].apply(_is_integer)
    records, graveyard = send_to_graveyard(
        records, graveyard, non_int_age, 'Non-integer age'
    )
 
    # Error must be an integer value
    non_int_err = ~records[STD_DEV].apply(_is_integer)
    records, graveyard = send_to_graveyard(
        records, graveyard, non_int_err, 'Non-integer error'
    )
 
    # Safe to cast to int now
    records[AGE]     = records[AGE].apply(lambda x: int(float(x)))
    records[STD_DEV] = records[STD_DEV].apply(lambda x: int(float(x)))
 
    # Future dates  (Age must be > 0)
    future = records[AGE] <= 0
    records, graveyard = send_to_graveyard(
        records, graveyard, future, 'Age from the future (Age ≤ 0)'
    )
 
    # Improbably small error  (< 15 BP, matching the original threshold)
    small_err = records[STD_DEV] < 15
    records, graveyard = send_to_graveyard(
        records, graveyard, small_err, 'Error less than 15 years'
    )
 
    # Error greater than Age
    err_gt_age = records[STD_DEV] > records[AGE]
    records, graveyard = send_to_graveyard(
        records, graveyard, err_gt_age, 'Error greater than age'
    )
 
    # Too old for meaningful 14C dating (> 55,000 BP)
    too_old = records[AGE] > 55_000
    records, graveyard = send_to_graveyard(
        records, graveyard, too_old, 'Record older than 55,000 BP'
    )
 
    # Normalise "United States" → "USA"
    if 'Country' in records.columns:
        records['Country'] = records['Country'].apply(
            lambda x: 'USA' if str(x).strip() == 'United States' else x
        )
 
    # Coerce Lat/Long to numeric float
    records[LAT] = records[LAT].apply(_to_float)
    records[LON] = records[LON].apply(_to_float)
 
    # Null out coordinates for known-bad LabIDs
    for bad_id in BAD_COORD_IDS:
        if bad_id in records.index:
            records.at[bad_id, LAT] = np.nan
            records.at[bad_id, LON] = np.nan
 
    return records, graveyard
 
# ---------------------------------------------------------------------------
# Step 7  –  Encoding repair
# ---------------------------------------------------------------------------
 
def fix_encoding_cols(records: pd.DataFrame) -> pd.DataFrame:
    print('\n[scrub] Step 7 – Repairing character encoding')
    fixer = lambda x: '' if is_nan(x) else fix_encoding(str(x))
    for i, col in enumerate(ENCODING_COLS):
        if col in records.columns:
            flush_msg(f'  Encoding ({i+1}/{len(ENCODING_COLS)}): {col:<20}')
            records[col] = records[col].apply(fixer)
    print()
    return records
 
# ---------------------------------------------------------------------------
# Step 8  –  Column sanitisation
# ---------------------------------------------------------------------------
 
_EXOTIC_WS_RE  = re.compile(
    r'(\s|\u180B|\u200B|\u200C|\u200D|\u2060|\uFEFF)+'
)
_STRIP_PUNCT   = set('\\"\\u201c\\u201d,')
 
 
def _col_fix(x):
    if is_nan(x):
        return np.nan
    s = str(x)
    # Remove stray quotes and commas
    for c in ('"', "'", '\u201c', '\u201d', ',', '\\'):
        s = s.replace(c, '')
    # Sentinel-swap ordinary spaces so the regex doesn't eat them
    s = s.replace(' ', '|')
    s = _EXOTIC_WS_RE.sub('', s)
    s = s.replace('|', ' ')
    return s
 
 
def fix_columns(records: pd.DataFrame) -> pd.DataFrame:
    print('\n[scrub] Step 8 – Sanitising text columns')
    for col in SANITISE_COLS:
        if col in records.columns:
            records[col] = records[col].apply(_col_fix)
    return records
 
# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
 
def save(df: pd.DataFrame, path: str) -> None:
    print(f'\n[scrub] Saving → {path}')
    df.to_csv(path, sep=',', encoding='utf-8')
 
# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
 
def main() -> None:
    if len(sys.argv) not in (3, 4):
        print('Usage:')
        print('  python scrub.py <input_file.csv> <output_file.csv> [graveyard.csv]')
        sys.exit(1)
 
    in_path        = sys.argv[1]
    out_path       = sys.argv[2]
    graveyard_path = sys.argv[3] if len(sys.argv) == 4 else 'graveyard.csv'
 
    if not os.path.isfile(in_path):
        print(f'Error: input file not found: {in_path}')
        sys.exit(1)
 
    # ---- Load ---------------------------------------------------------------
    original = get_records(in_path)
    n_in     = len(original)
    graveyard = pd.DataFrame()
 
    # ---- Pipeline (order matches the official package) ----------------------
    records, graveyard = delete_bad_labs(original, graveyard)
    records            = convert_coordinates(records)
    records            = strip_whitespace(records)
    records, graveyard = handle_duplicates(records, graveyard)
    records, graveyard = apply_misc_scrubbing(records, graveyard)
 
    # Save graveyard first (mirrors original order)
    save(graveyard, graveyard_path)
 
    records = fix_encoding_cols(records)
    records = fix_columns(records)
 
    save(records, out_path)
 
    # ---- Summary ------------------------------------------------------------
    n_out     = len(records)
    n_removed = n_in - n_out
    pct       = n_removed / n_in * 100 if n_in else 0.0
 
    print(f'\n{"="*60}')
    print(f'  Input    : {n_in:>7,} records')
    print(f'  Output   : {n_out:>7,} records')
    print(f'  Removed  : {n_removed:>7,} records  ({pct:.1f}%)')
    print(f'  Cleaned  : {out_path}')
    print(f'  Graveyard: {graveyard_path}')
    print(f'  Unknown  : unknown_codes.csv')
    print(f'{"="*60}')
 
 
if __name__ == '__main__':
    main()
