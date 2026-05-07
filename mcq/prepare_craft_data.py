"""
prepare_craft_data.py

Converts scored chunk files into retrieved.jsonl format required by CRAFT.

Run after preprocess.py (all 4 steps) and before the CRAFT MCQ pipeline.

Usage
-----
# Use defaults (reads from {scored_dir}, writes to {craft_assets_dir}):
  python prepare_craft_data.py --config config.yaml

# Override paths on the fly:
  python prepare_craft_data.py --config config.yaml \\
      --set paths.scored_dir=./output/scored \\
      --set paths.craft_assets_dir=./craft_assets
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


# ============================================================
# Config helpers
# ============================================================

def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_overrides(cfg: dict, overrides: list) -> None:
    for kv in overrides:
        if "=" not in kv:
            sys.exit(f"[error] --set requires key=value format, got: {kv!r}")
        key_str, val_str = kv.split("=", 1)
        keys  = key_str.strip().split(".")
        value = yaml.safe_load(val_str)
        node  = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value


# ============================================================
# Core conversion
# ============================================================

def convert_jsonl_to_retrieved(in_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retrieved.jsonl"

    n_entities = n_written = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            item   = json.loads(line)
            entity = item.get("entity")
            if not entity:
                continue
            n_entities += 1

            chunks = item.get("chunks", [])
            if not isinstance(chunks, list) or not chunks:
                continue

            for ch in chunks:
                ctx = (ch.get("chunk") or "").strip()
                if not ctx:
                    continue

                # keep only chunks with a positive CE score
                if (ch.get("ce_score") or 0) <= 0:
                    continue

                text = f"Target Entity: {entity}\nContext: {ctx}"
                meta = {
                    "density":  ch.get("density"),
                    "ce_score": ch.get("ce_score"),
                }
                fout.write(json.dumps({"text": text, **meta}, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"[OK] {in_path} -> {out_path}  (entities={n_entities}, written={n_written})")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config file (default: config.yaml)")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                        help="Override config values, e.g. --set paths.craft_assets_dir=./assets")
    args = parser.parse_args()

    if not Path(args.config).exists():
        sys.exit(f"[error] Config file not found: {args.config}")

    cfg = _load_config(args.config)
    _apply_overrides(cfg, args.set)

    scored_dir      = Path(cfg["paths"]["out_dir"]) / "scored"
    craft_assets    = Path(cfg["paths"]["craft_assets_dir"])

    for source in ["AH", "AT", "WH", "WT"]:
        in_path = scored_dir / f"chunk_{source}_scored.jsonl"
        if not in_path.exists():
            print(f"[SKIP] {in_path} not found — run the 'scoring' step first.")
            continue

        out_dir = craft_assets / source / "corpus_samples"
        convert_jsonl_to_retrieved(in_path, out_dir)


if __name__ == "__main__":
    main()