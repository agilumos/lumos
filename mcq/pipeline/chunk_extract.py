"""
pipeline/chunk_extract.py

Scans text corpora with Aho-Corasick automaton and stores the densest
context windows around each entity mention.

Output (in {out_dir}/chunks/):
  freq_web.jsonl          entity document/mention frequencies
  freq_academic.jsonl
  chunk_web.jsonl.zst     per-entity top-k context chunks (zstd)
  chunk_academic.jsonl.zst
"""

import heapq
import json
import os
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import ahocorasick
import zstandard as zstd

from .utils import (
    normalize,
    iter_jsonl_text_with_byte_offsets,
    build_input_files,
)


# =========================
# Entity loading
# =========================

def _load_entity_lists(entity_json_path: str, run_type: str):
    with open(entity_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    key = "degree" if run_type == "web" else "cited_by_count"

    def unpack(lst):
        names, vals = [], []
        for it in (lst or []):
            if isinstance(it, dict):
                names.append((it.get("name") or "").strip())
                vals.append(int(it.get(key, 0) or 0))
            else:
                names.append(str(it).strip())
                vals.append(0)
        return names, vals

    head_names, head_vals = unpack(obj.get("head", []))
    tail_names, tail_vals = unpack(obj.get("tail", []))
    return head_names, tail_names, head_vals, tail_vals


# =========================
# Aho-Corasick automaton
# =========================

def _build_automaton(
    canonical_entities: List[str],
    min_entity_len: int = 4,
) -> Tuple["ahocorasick.Automaton", List[str], List[int]]:
    A = ahocorasick.Automaton()
    id2entity: List[str] = []
    kept_src_idx: List[int] = []
    seen_key = set()

    for src_i, canon in enumerate(canonical_entities):
        canon = (canon or "").strip()
        if not canon:
            continue
        key = normalize(canon)
        if len(key) < min_entity_len or key in seen_key:
            continue
        seen_key.add(key)

        canon_id = len(id2entity)
        id2entity.append(canon)
        kept_src_idx.append(src_i)
        A.add_word(key, (canon_id, len(key)))

    A.make_automaton()
    return A, id2entity, kept_src_idx


# =========================
# Window scoring
# =========================

def _best_dense_window(
    positions: List[int],
    text_len: int,
    radius: int,
) -> Tuple[int, int, float]:
    if not positions:
        return 0, 0, 0.0

    best_L, best_R, best_count, best_density = 0, 0, 0, -1.0
    n = len(positions)
    left = right = 0

    for i in range(n):
        p = positions[i]
        lo = p - radius
        hi = p + radius

        while left < n and positions[left] < lo:
            left += 1
        if right < left:
            right = left
        while right < n and positions[right] <= hi:
            right += 1

        count = right - left
        W = 2 * radius + 1
        L = max(0, lo)
        R = min(text_len, hi + 1)

        cur = R - L
        if cur < W:
            need = W - cur
            if L == 0:
                R = min(text_len, R + need)
            elif R == text_len:
                L = max(0, L - need)

        win_len = max(1, R - L)
        density  = count / win_len

        if (density > best_density
                or (density == best_density and count > best_count)
                or (density == best_density and count == best_count and L < best_L)):
            best_density = density
            best_count   = count
            best_L, best_R = L, R

    return best_L, best_R, best_density


# =========================
# Chunk quality filter
# =========================

_NOISE_PATTERNS = [
    re.compile(r"\bimage\d+\b"),
    re.compile(r"\bimg\d+\b"),
    re.compile(r"\bfigure\d+\b"),
    re.compile(r"\btable of contents\b"),
    re.compile(r"\.(png|jpg|jpeg|gif)\b"),
]
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_WS_RE    = re.compile(r"\s+")


def _is_low_quality(
    t: str,
    min_chars: int = 200,
    max_repeated_line_frac: float = 0.35,
    min_unique_token_ratio: float = 0.20,
    max_noise_hits: int = 3,
    min_lines_for_table_check: int = 20,
    max_avg_tokens_per_line: float = 4.0,
    max_top1_token_frac: float = 0.25,
) -> bool:
    if len(t) < min_chars:
        return True

    hits = 0
    for cre in _NOISE_PATTERNS:
        for _ in cre.finditer(t.lower()):
            hits += 1
            if hits >= max_noise_hits:
                return True

    raw_lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(raw_lines) >= 8:
        norm_lines = [_WS_RE.sub(" ", ln) for ln in raw_lines]
        top_cnt = Counter(norm_lines).most_common(1)[0][1]
        if top_cnt / len(norm_lines) >= max_repeated_line_frac:
            return True

    toks = _TOKEN_RE.findall(t.lower())
    if len(toks) >= 30:
        if len(set(toks)) / len(toks) < min_unique_token_ratio:
            return True
        top1 = Counter(toks).most_common(1)[0][1]
        if top1 / len(toks) >= max_top1_token_frac:
            return True

    if len(raw_lines) >= min_lines_for_table_check:
        avg = sum(len(_TOKEN_RE.findall(ln.lower())) for ln in raw_lines) / len(raw_lines)
        if 0 < avg <= max_avg_tokens_per_line:
            return True

    return False


# =========================
# Core pipeline per run_type
# =========================

def _run_single(
    run_type: str,
    entity_json: str,
    input_files: List[str],
    out_freq: str,
    out_chunks: str,
    chunk_cfg: dict,
) -> None:
    radius    = int(chunk_cfg["window"])
    topk      = int(chunk_cfg["topk_chunks"])
    min_alen  = int(chunk_cfg["min_alias_len"])
    log_every = int(chunk_cfg["log_every_lines"])
    build_off = bool(chunk_cfg["build_offsets"])
    off_sfx   = chunk_cfg["offsets_suffix"]

    # Load entities
    head, tail, head_vals, tail_vals = _load_entity_lists(entity_json, run_type)
    all_entities  = head + tail
    entity_vals   = head_vals + tail_vals
    entity_bucket = (["head"] * len(head)) + (["tail"] * len(tail))
    print(f"[entities] head={len(head):,}  tail={len(tail):,}  total={len(all_entities):,}")

    # Build automaton
    A, id2entity, kept_src_idx = _build_automaton(all_entities, min_alen)
    entity_bucket = [entity_bucket[i] for i in kept_src_idx]
    entity_vals   = [entity_vals[i]   for i in kept_src_idx]
    print(f"[automaton] canonical_entities={len(id2entity):,}")

    # Stats
    doc_freq     = [0] * len(id2entity)
    mention_freq = [0] * len(id2entity)
    heaps        = [[] for _ in range(len(id2entity))]
    seen_per_eid = [set() for _ in range(len(id2entity))]
    uniq_counter = 0
    cctx         = zstd.ZstdCompressor(level=3)

    t0 = time.perf_counter()
    total_docs = total_lines = 0

    for fp in input_files:
        print(f"[file] {fp}")
        file_t0   = time.perf_counter()
        used_docs = lines_seen = 0
        offsets_path = fp + off_sfx

        for line_no, byte_off, text in iter_jsonl_text_with_byte_offsets(
            fp,
            build_offsets=build_off,
            offsets_out_path=(offsets_path if build_off else None),
            offsets_suffix=off_sfx,
        ):
            lines_seen  += 1
            total_lines += 1
            used_docs   += 1
            total_docs  += 1

            text_n  = normalize(text)
            doc_len = len(text_n)

            positions_by_eid: Dict[int, List[int]] = defaultdict(list)
            for end_idx, (eid, alen) in A.iter(text_n):
                start_idx = end_idx - alen + 1
                if start_idx < 0:
                    continue
                next_i = end_idx + 1
                if next_i < doc_len and text_n[next_i].isalnum():
                    continue
                prev_i = start_idx - 1
                if prev_i >= 0 and text_n[prev_i].isalnum():
                    continue
                positions_by_eid[eid].append(start_idx)
                mention_freq[eid] += 1

            for eid in positions_by_eid:
                doc_freq[eid] += 1

            for eid, pos_list in positions_by_eid.items():
                L, R, density = _best_dense_window(pos_list, doc_len, radius)
                if R <= L or density >= 0.02:
                    continue

                chunk_text = text_n[L:R].strip()
                first_line = chunk_text.splitlines()[0].strip()
                if "(disambiguation)" in first_line.lower():
                    continue
                if _is_low_quality(chunk_text):
                    continue
                if chunk_text in seen_per_eid[eid]:
                    continue
                seen_per_eid[eid].add(chunk_text)

                h = heaps[eid]
                uniq_counter += 1
                item = (float(density), uniq_counter, chunk_text, fp, line_no, byte_off)

                if len(h) < topk:
                    heapq.heappush(h, item)
                elif density > h[0][0]:
                    heapq.heapreplace(h, item)

            if log_every > 0 and lines_seen % log_every == 0:
                print(f"  lines={lines_seen:,}  elapsed={time.perf_counter()-file_t0:.1f}s")

        print(f"  done lines={lines_seen:,}  elapsed={time.perf_counter()-file_t0:.1f}s")

    # Write freq stats
    with open(out_freq, "w", encoding="utf-8") as f_stats:
        for eid, ent in enumerate(id2entity):
            f_stats.write(json.dumps({
                "entity_id":    eid,
                "entity":       ent,
                "bucket":       entity_bucket[eid],
                "doc_freq":     doc_freq[eid],
                "mention_freq": mention_freq[eid],
                "original_freq": entity_vals[eid],
            }, ensure_ascii=False) + "\n")

    # Write chunks
    with open(out_chunks, "wb") as f_raw, cctx.stream_writer(f_raw) as f_out:
        for eid, ent in enumerate(id2entity):
            h     = heaps[eid]
            items = sorted(h, key=lambda x: x[0], reverse=True)
            chunks = [
                {
                    "chunk":   chunk_text,
                    "density": density,
                    "source":  {"file": src_fp, "line": ln, "byte_offset": boff},
                }
                for density, _, chunk_text, src_fp, ln, boff in items
            ]
            rec = {"entity_id": eid, "entity": ent, "bucket": entity_bucket[eid], "chunks": chunks}
            f_out.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))

    elapsed = time.perf_counter() - t0
    print(f"[chunk_extract] run_type={run_type}  total_lines={total_lines:,}  elapsed={elapsed/60:.2f}min")
    print(f"  freq  -> {out_freq}")
    print(f"  chunks-> {out_chunks}")


# =========================
# Step entry point
# =========================

def run(cfg: dict) -> None:
    paths     = cfg["paths"]
    chunk_cfg = cfg["chunk"]
    entity_dir = os.path.join(paths["out_dir"], "entities")
    chunk_dir  = os.path.join(paths["out_dir"], "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    print("\n=== [Step 2] Chunk Extraction ===")

    for run_type, entity_fname in [("web", "dbpedia_entity.json"), ("academic", "openalex_entity.json")]:
        entity_json = os.path.join(entity_dir, entity_fname)
        if not os.path.exists(entity_json):
            raise FileNotFoundError(
                f"Entity file not found: {entity_json}\n"
                "Run the 'entity' step first."
            )

        input_files = build_input_files(run_type, cfg)
        out_freq    = os.path.join(chunk_dir, f"freq_{run_type}.jsonl")
        out_chunks  = os.path.join(chunk_dir, f"chunk_{run_type}.jsonl.zst")

        print(f"\n--- run_type={run_type}  input_files={len(input_files):,} ---")
        _run_single(run_type, entity_json, input_files, out_freq, out_chunks, chunk_cfg)

    print("[Step 2] Done.\n")
