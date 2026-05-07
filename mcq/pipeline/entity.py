"""
pipeline/entity.py

Extracts head/tail entity lists from:
  - DBpedia  → {out_dir}/entities/dbpedia_entity.json
  - OpenAlex → {out_dir}/entities/openalex_entity.json

Adapted from:
  https://github.com/facebookresearch/head-to-tail  (CC BY-NC 4.0)
"""

import bz2
import glob
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import unquote


# ============================================================
# DBpedia
# ============================================================

def _parse_ttl_triples(fp_bz2: str):
    """Yield (subj, pred, obj) string triples from a bz2-compressed TTL file."""
    with bz2.open(fp_bz2, mode="rt") as f:
        for line in f:
            line = line.strip().strip(".")
            triple = []
            cnt = 0
            start = 0
            for i, ch in enumerate(line):
                if ch == "<":
                    cnt += 1
                elif ch == ">":
                    cnt -= 1
                    if cnt == 0:
                        triple.append(line[start:i + 1].strip())
                        start = i + 1
            if len(triple) >= 3:
                yield triple[0], triple[1], triple[2]


def build_dbpedia_entities(kg_dir: str, out_dir: str) -> str:
    """
    Parse DBpedia TTL, split entities into head/tail by degree,
    write dbpedia_entity.json, and return the output path.
    """
    ttl_path = os.path.join(kg_dir, "mappingbased-objects_lang=en.ttl.bz2")

    # --- Pass 1: count entity degrees ---
    print("[dbpedia] Pass 1: counting entity degrees ...")
    entity: Dict[str, int] = {}
    for subj, pred, obj in _parse_ttl_triples(ttl_path):
        entity[subj] = entity.get(subj, 0) + 1
        entity[obj]  = entity.get(obj, 0) + 1

    # --- Determine head/tail cutoffs (equal-mass thirds) ---
    degrees = sorted(entity.values())
    total   = sum(degrees)
    third   = total / 3.0
    cum, co1, co2 = 0, None, None
    for i, d in enumerate(degrees):
        cum += d
        if co1 is None and cum / total >= 1 / 3:
            co1 = i
        if co2 is None and cum / total >= 2 / 3:
            co2 = i
            break

    thresh_tail = degrees[co1]
    thresh_head = degrees[co2]

    def _htt(ent_str: str) -> str:
        d = entity.get(ent_str, 0)
        if d <= thresh_tail:
            return "tail"
        if d <= thresh_head:
            return "torso"
        return "head"

    # --- Pass 2: collect relation sample sizes ---
    print("[dbpedia] Pass 2: collecting relation sample sizes ...")
    _SKIP_RELATIONS = {
        "<http://xmlns.com/foaf/0.1/page>",
        "<http://xmlns.com/foaf/0.1/homepage>",
        "<http://dbpedia.org/ontology/webcast>",
    }

    maxsample: Dict[str, Dict[str, int]] = {}
    for subj, pred, obj in _parse_ttl_triples(ttl_path):
        if obj.startswith("<http") and pred not in _SKIP_RELATIONS:
            continue
        if "<http://dbpedia.org/resource/List_of_" in subj:
            continue
        h = _htt(subj)
        maxsample.setdefault(pred, {}).setdefault(h, 0)
        maxsample[pred][h] += 1

    for pred in maxsample:
        maxsample[pred] = min(
            maxsample[pred].get("head", 0),
            maxsample[pred].get("torso", 0),
            maxsample[pred].get("tail", 0),
        )

    # --- Pass 3: sample entities per head/tail ---
    print("[dbpedia] Pass 3: sampling entities ...")
    entities: Dict[str, Dict[str, int]] = {"head": {}, "tail": {}}
    stat: Dict[str, Dict[str, int]] = {"head": {}, "tail": {}}

    for subj, pred, obj in _parse_ttl_triples(ttl_path):
        if obj.startswith("<http") and pred not in _SKIP_RELATIONS:
            continue
        if "<http://dbpedia.org/resource/List_of_" in subj:
            continue
        h = _htt(subj)
        if h == "torso":
            continue
        if pred not in maxsample or maxsample[pred] == 0:
            continue

        stat[h].setdefault(pred, 0)
        if stat[h][pred] >= maxsample[pred]:
            continue

        name = unquote(subj.split("/")[-1][:-1]).replace("_", " ")
        if "__" in name or not name:
            continue

        entities[h][name] = entity[subj]
        stat[h][pred] += 1

    output = {
        grp: sorted(
            [{"name": n, "degree": d} for n, d in ents.items()],
            key=lambda r: r["degree"],
            reverse=True,
        )
        for grp, ents in entities.items()
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dbpedia_entity.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=1, ensure_ascii=False)

    print(f"[dbpedia] head={len(output['head']):,}  tail={len(output['tail']):,}")
    print(f"[dbpedia] wrote {out_path}")
    return out_path


# ============================================================
# OpenAlex
# ============================================================

def _load_openalex_concepts(
    files: List[str],
    target_levels: List[int],
) -> List[Tuple[str, int]]:
    """Return [(display_name, cited_by_count)] for concepts at target levels."""
    name2count: Dict[str, int] = {}
    for fp in files:
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("level") not in target_levels:
                    continue
                name = (obj.get("display_name") or "").strip()
                if not name:
                    continue
                cbc = int(obj.get("cited_by_count") or 0)
                if name not in name2count or cbc > name2count[name]:
                    name2count[name] = cbc

    return list(name2count.items())


def build_openalex_entities(
    openalex_glob: str,
    target_levels: List[int],
    out_dir: str,
) -> str:
    """
    Load OpenAlex concepts, split by cumulative-mass thirds,
    write openalex_entity.json, and return the output path.
    """
    files = sorted(glob.glob(openalex_glob, recursive=True))
    if not files:
        raise FileNotFoundError(f"No OpenAlex files matched: {openalex_glob}")
    print(f"[openalex] matched files={len(files):,}  levels={target_levels}")

    items = _load_openalex_concepts(files, target_levels)
    print(f"[openalex] unique concepts={len(items):,}")

    items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
    total_mass   = sum(v for _, v in items_sorted)
    one_third    = total_mass / 3.0

    head_items, tail_items = [], []
    cum = 0.0
    for name, val in items_sorted:
        if cum >= one_third:
            break
        head_items.append({"name": name, "cited_by_count": val})
        cum += val

    cum = 0.0
    for name, val in reversed(items_sorted):
        if cum >= one_third:
            break
        tail_items.append({"name": name, "cited_by_count": val})
        cum += val

    output = {"head": head_items, "tail": tail_items}

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "openalex_entity.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=1, ensure_ascii=False)

    print(f"[openalex] head={len(head_items):,}  tail={len(tail_items):,}")
    print(f"[openalex] wrote {out_path}")
    return out_path


# ============================================================
# Step entry point
# ============================================================

def run(cfg: dict) -> None:
    paths  = cfg["paths"]
    e_cfg  = cfg["entity"]
    entity_dir = os.path.join(paths["out_dir"], "entities")

    print("\n=== [Step 1] Entity Extraction ===")

    build_dbpedia_entities(
        kg_dir  = paths["kg_dir"],
        out_dir = entity_dir,
    )

    build_openalex_entities(
        openalex_glob  = paths["openalex_glob"],
        target_levels  = e_cfg["openalex_levels"],
        out_dir        = entity_dir,
    )

    print("[Step 1] Done.\n")
