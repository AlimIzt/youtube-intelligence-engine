"""Clean noisy user-generated comment text.

Steps: strip HTML/URLs, normalise unicode, collapse repeated chars, drop empty /
non-English-ish rows, and (optionally) SymSpell spell correction.
"""
from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache

import pandas as pd

from config import settings

_URL = re.compile(r"https?://\S+|www\.\S+")
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_REPEAT = re.compile(r"(.)\1{2,}")  # "soooo" -> "soo"
_NON_PRINT = re.compile(r"[^\x09\x0A\x0D\x20-\x7E -￿]")


@lru_cache(maxsize=1)
def _speller():
    """Lazily build a SymSpell dictionary. Returns None if unavailable."""
    try:
        import importlib.resources as ir

        from symspellpy import SymSpell, Verbosity  # noqa: F401

        sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dict_path = ir.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
        sym.load_dictionary(str(dict_path), term_index=0, count_index=1)
        return sym
    except Exception as e:  # pragma: no cover
        print(f"[clean] spell correction disabled: {e}")
        return None


def correct_spelling(text: str) -> str:
    sym = _speller()
    if not sym:
        return text
    from symspellpy import Verbosity

    out = []
    for tok in text.split():
        if not tok.isalpha():
            out.append(tok)
            continue
        sug = sym.lookup(tok, Verbosity.CLOSEST, max_edit_distance=2)
        out.append(sug[0].term if sug else tok)
    return " ".join(out)


def clean_text(text: str, spell: bool = False) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = _NON_PRINT.sub("", text)
    text = _REPEAT.sub(r"\1\1", text)
    text = _WS.sub(" ", text).strip()
    if spell:
        text = correct_spelling(text)
    return text


def clean_dataframe(
    df: pd.DataFrame, text_col: str = "text", spell: bool = False, min_len: int = 3
) -> pd.DataFrame:
    df = df.copy()
    df["text_raw"] = df[text_col]
    df[text_col] = df[text_col].map(lambda t: clean_text(t, spell=spell))
    df = df[df[text_col].str.len() >= min_len]
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    df["comment_id"] = df.index.map(lambda i: f"c{i:06d}")
    return df


def main() -> None:
    df = pd.read_csv(settings.raw_csv)
    print(f"Loaded {len(df):,} raw comments.")
    # Spell correction is slow on 20k rows; default off. Pass spell=True to enable.
    clean = clean_dataframe(df, spell=False)
    clean.to_csv(settings.clean_csv, index=False, encoding="utf-8-sig")
    print(f"Saved {len(clean):,} cleaned comments → {settings.clean_csv}")


if __name__ == "__main__":
    main()
