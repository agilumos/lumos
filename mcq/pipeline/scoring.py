"""
pipeline/scoring.py

Scores every chunk with a CrossEncoder model and re-sorts by score.

Input  (from {out_dir}/splits/): chunk_WH.jsonl, chunk_WT.jsonl, chunk_AH.jsonl, chunk_AT.jsonl
Output (to   {out_dir}/scored/): chunk_WH_scored.jsonl, ...
"""

import json
import os
from typing import Any, Dict, Iterator, List, Tuple

from tqdm import tqdm


def _batched(iterable, batch_size: int) -> Iterator[List]:
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_bad_entities(path: str | None) -> set:
    bad: set = set()
    if not path or not os.path.exists(path):
        return bad
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                bad.add(s)
    print(f"[scoring] bad_entities loaded={len(bad)}  from={path}")
    return bad


def _score_entity_chunks(
    model,
    entity: str,
    chunks: List[Dict[str, Any]],
    batch_size: int,
    max_chunks: int | None,
) -> List[Dict[str, Any]]:
    if not chunks:
        return chunks
    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    query  = f"Describe {entity}."
    pairs  = [(query, ch.get("chunk", "")) for ch in chunks]
    scores: List[float] = []
    for batch in _batched(pairs, batch_size):
        scores.extend(float(s) for s in model.predict(batch))

    out = [dict(ch, ce_score=sc) for ch, sc in zip(chunks, scores)]
    out.sort(key=lambda x: x.get("ce_score", float("-inf")), reverse=True)
    return out


# ============================================================
# Step entry point
# ============================================================

def run(cfg: dict) -> None:
    paths      = cfg["paths"]
    score_cfg  = cfg["scoring"]
    split_dir  = os.path.join(paths["out_dir"], "splits")
    scored_dir = os.path.join(paths["out_dir"], "scored")
    os.makedirs(scored_dir, exist_ok=True)

    # Optionally restrict visible GPUs before importing torch/transformers
    cuda_devs = score_cfg.get("cuda_visible_devices")
    if cuda_devs is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devs)
        print(f"[scoring] CUDA_VISIBLE_DEVICES={cuda_devs}")

    from sentence_transformers import CrossEncoder  # import after env is set

    print("\n=== [Step 4] Chunk Scoring ===")

    model = CrossEncoder(
        score_cfg["model"],
        device=score_cfg.get("device") or None,
    )
    print(f"[scoring] model={score_cfg['model']}")

    bad_entities = _load_bad_entities(score_cfg.get("bad_entities_file"))

    for source in ["WH", "WT", "AH", "AT"]:
        in_path  = os.path.join(split_dir,  f"chunk_{source}.jsonl")
        out_path = os.path.join(scored_dir, f"chunk_{source}_scored.jsonl")

        if not os.path.exists(in_path):
            print(f"[scoring] SKIP {source}: {in_path} not found")
            continue

        print(f"\n--- source={source} ---")
        print(f"  input : {in_path}")
        print(f"  output: {out_path}")

        n = skipped = 0
        flush_every = int(score_cfg.get("flush_every", 200))

        with open(in_path, "r", encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:

            for line in tqdm(fin, desc=f"Scoring {source}"):
                line = line.strip()
                if not line:
                    continue
                obj    = json.loads(line)
                entity = obj.get("entity", "")

                if entity in bad_entities:
                    skipped += 1
                    continue

                obj["chunks"] = _score_entity_chunks(
                    model, entity, obj.get("chunks", []),
                    batch_size=int(score_cfg["batch_size"]),
                    max_chunks=score_cfg.get("max_chunks"),
                )
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n += 1

                if flush_every and n % flush_every == 0:
                    fout.flush()

        print(f"  wrote={n:,}  skipped_bad={skipped:,}  -> {out_path}")

    print("[Step 4] Done.\n")
