"""
pipeline/split.py

Splits entities into WH / WT / AH / AT groups using:
  1) mention_freq mass thirds (top 1/3 → new_head, bottom 1/3 → new_tail)
  2) intersection with the original DBpedia/OpenAlex bucket label
  3) only entities that appear in at least one non-empty chunk

Output (in {out_dir}/splits/):
  freq_WH.jsonl, freq_WT.jsonl, freq_AH.jsonl, freq_AT.jsonl
  chunk_WH.jsonl, chunk_WT.jsonl, chunk_AH.jsonl, chunk_AT.jsonl
"""

import io
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .utils import iter_any_jsonl


# ============================================================
# Data model
# ============================================================

@dataclass
class _EntityRec:
    entity:       str
    bucket:       str
    doc_freq:     int
    mention_freq: int
    raw:          dict


# ============================================================
# Mass-thirds split
# ============================================================

def _load_freq_recs(freq_path: str) -> List[_EntityRec]:
    out: List[_EntityRec] = []
    for r in iter_any_jsonl(freq_path):
        ent = (r.get("entity") or "").strip()
        if not ent:
            continue
        out.append(_EntityRec(
            entity       = ent,
            bucket       = (r.get("bucket") or "").strip(),
            doc_freq     = int(r.get("doc_freq") or 0),
            mention_freq = int(r.get("mention_freq") or 0),
            raw          = r,
        ))
    return out


def _mass_thirds(recs: List[_EntityRec]) -> Tuple[Set[str], Set[str]]:
    total = sum(r.mention_freq for r in recs)
    third = total / 3.0 if total else 0.0

    head_set: Set[str] = set()
    cum = 0
    for r in recs:                          # already sorted desc
        cum += r.mention_freq
        head_set.add(r.entity)
        if cum >= third:
            break

    tail_set: Set[str] = set()
    cum = 0
    for r in reversed(recs):
        cum += r.mention_freq
        tail_set.add(r.entity)
        if cum >= third:
            break

    return head_set, tail_set


def _compute_groups(freq_path: str, prefix: str) -> Dict[str, Set[str]]:
    recs = _load_freq_recs(freq_path)
    recs.sort(key=lambda r: r.mention_freq, reverse=True)

    new_head, new_tail = _mass_thirds(recs)
    total = sum(r.mention_freq for r in recs)

    H: Set[str] = {r.entity for r in recs
                   if r.doc_freq > 0 and r.mention_freq > 0
                   and r.entity in new_head and r.bucket == "head"}
    T: Set[str] = {r.entity for r in recs
                   if r.doc_freq > 0 and r.mention_freq > 0
                   and r.entity in new_tail and r.bucket == "tail"}

    print(f"\n[{prefix}] freq={freq_path}")
    print(f"  total_mention_mass={total:,}")
    print(f"  new_head (top 1/3 mass)   = {len(new_head):,}")
    print(f"  new_tail (bottom 1/3 mass)= {len(new_tail):,}")
    print(f"  overlap {prefix}H (new_head ∩ bucket=head) = {len(H):,}")
    print(f"  overlap {prefix}T (new_tail ∩ bucket=tail) = {len(T):,}")

    return {f"{prefix}H": H, f"{prefix}T": T}


# ============================================================
# Non-empty chunk filter
# ============================================================

def _nonempty_chunk_entities(chunks_path: str) -> Set[str]:
    s: Set[str] = set()
    for obj in iter_any_jsonl(chunks_path):
        ent    = (obj.get("entity") or "").strip()
        chunks = obj.get("chunks")
        if ent and isinstance(chunks, list) and chunks:
            s.add(ent)
    return s


# ============================================================
# Writers
# ============================================================

def _write_splits(
    src_path:  str,
    out_dir:   str,
    groups:    Dict[str, Set[str]],
    file_stem: str,          # "freq" or "chunk"
) -> None:
    fouts: Dict[str, io.TextIOBase] = {}
    try:
        for g in groups:
            fouts[g] = open(os.path.join(out_dir, f"{file_stem}_{g}.jsonl"), "w", encoding="utf-8")

        kept  = {g: 0 for g in groups}
        total = 0
        for obj in iter_any_jsonl(src_path):
            total += 1
            ent = (obj.get("entity") or "").strip()
            for g, s in groups.items():
                if ent in s:
                    fouts[g].write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept[g] += 1

        print(f"[{file_stem} split] {src_path}  total={total:,}  kept={kept}")
    finally:
        for fh in fouts.values():
            try:
                fh.close()
            except Exception:
                pass


# ============================================================
# Step entry point
# ============================================================

def run(cfg: dict) -> None:
    paths     = cfg["paths"]
    chunk_dir = os.path.join(paths["out_dir"], "chunks")
    split_dir = os.path.join(paths["out_dir"], "splits")
    os.makedirs(split_dir, exist_ok=True)

    freq_web   = os.path.join(chunk_dir, "freq_web.jsonl")
    freq_acad  = os.path.join(chunk_dir, "freq_academic.jsonl")
    chunk_web  = os.path.join(chunk_dir, "chunk_web.jsonl.zst")
    chunk_acad = os.path.join(chunk_dir, "chunk_academic.jsonl.zst")

    for p in [freq_web, freq_acad, chunk_web, chunk_acad]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required input not found: {p}\nRun the 'chunk' step first.")

    print("\n=== [Step 3] Entity Split (WH/WT/AH/AT) ===")

    web_groups  = _compute_groups(freq_web,  "W")
    acad_groups = _compute_groups(freq_acad, "A")

    # Filter to entities that have ≥1 chunk
    print("\n[split] scanning non-empty chunk entity sets ...")
    web_nonempty  = _nonempty_chunk_entities(chunk_web)
    acad_nonempty = _nonempty_chunk_entities(chunk_acad)
    print(f"  web_nonempty={len(web_nonempty):,}  acad_nonempty={len(acad_nonempty):,}")

    for g in ["WH", "WT"]:
        web_groups[g]  = {e for e in web_groups[g]  if e in web_nonempty}
    for g in ["AH", "AT"]:
        acad_groups[g] = {e for e in acad_groups[g] if e in acad_nonempty}

    print("\n[split] writing freq splits ...")
    _write_splits(freq_web,  split_dir, web_groups,  "freq")
    _write_splits(freq_acad, split_dir, acad_groups, "freq")

    print("\n[split] writing chunk splits ...")
    _write_splits(chunk_web,  split_dir, web_groups,  "chunk")
    _write_splits(chunk_acad, split_dir, acad_groups, "chunk")

    print(f"\n[Step 3] Done. Outputs in {split_dir}\n")
