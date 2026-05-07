import io
import json
import struct
import unicodedata
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import zstandard as zstd


# =========================
# Normalization
# =========================
_TRANSLATE_PUNCT = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2212": "-", "\uFE63": "-", "\uFF0D": "-",

    "\u2018": "'", "\u2019": "'", "\u02BC": "'", "\uFF07": "'",
    "\u201C": '"', "\u201D": '"', "\uFF02": '"',

    "\u200B": "", "\u200C": "", "\u200D": "", "\uFEFF": "",
})


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.translate(_TRANSLATE_PUNCT)
    return s.lower()


# =========================
# JSONL readers
# =========================
def iter_jsonl_text_with_byte_offsets(
    fp: str,
    build_offsets: bool = False,
    offsets_out_path: Optional[str] = None,
    offsets_suffix: str = ".offsets.u64",
) -> Iterator[Tuple[int, int, str]]:
    """Yields (line_no_1based, byte_offset, text) for each JSONL line."""
    if build_offsets and offsets_out_path is None:
        offsets_out_path = fp + offsets_suffix

    offsets_f = open(offsets_out_path, "wb") if build_offsets else None

    try:
        with open(fp, "rb") as f:
            line_no = 0
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                line_no += 1

                if offsets_f is not None:
                    offsets_f.write(struct.pack("<Q", pos))

                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    obj = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                text = obj.get("text", "")
                if text:
                    yield line_no, pos, text
    finally:
        if offsets_f is not None:
            offsets_f.close()


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def iter_jsonl_zst(path: str) -> Iterator[dict]:
    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_wrapper = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_wrapper:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def iter_any_jsonl(path: str) -> Iterator[dict]:
    if path.endswith(".zst"):
        yield from iter_jsonl_zst(path)
    else:
        yield from iter_jsonl(path)


# =========================
# Input file builder
# =========================
def build_input_files(run_type: str, cfg: dict) -> List[str]:
    paths = cfg["paths"]
    if run_type == "web":
        return [paths["web_corpus"]]
    elif run_type == "academic":
        corpus_dir = Path(paths["academic_corpus_dir"])
        files = sorted(corpus_dir.glob("pes2o-*.json"))
        return [str(p) for p in files]
    else:
        raise ValueError(f"Unknown run_type: {run_type!r}")
