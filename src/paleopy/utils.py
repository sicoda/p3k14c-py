"""Small generic helpers shared across paleopy modules.

Ported from Scripts/01_Data_Cleaning_and_Prep.py, generalized for reuse
beyond the cleaning pipeline (e.g. progress bars in calibration/SPD).
"""

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

try:
    import ftfy as _ftfy

    def fix_encoding(text) -> str:
        return _ftfy.fix_encoding(str(text))
except ImportError:
    def fix_encoding(text) -> str:  # type: ignore[misc]
        """Minimal mojibake repair: re-encode as latin-1 then decode as utf-8.

        Covers the most common corruption pattern produced by Excel, used
        when the optional ``ftfy`` dependency isn't installed.
        """
        try:
            return str(text).encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return str(text)


def is_nan(value) -> bool:
    return str(value) == "nan"


def flush_msg(msg: str) -> None:
    print(f"\r{msg}", end="", flush=True)
