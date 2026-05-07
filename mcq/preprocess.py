#!/usr/bin/env python3
"""
preprocess.py — Single entry point for the MCQ pipeline.

Steps (run in order, or choose specific steps with --steps):
  1. entity   : Extract head/tail entity lists from DBpedia & OpenAlex
  2. chunk    : Scan corpora and collect context chunks per entity
  3. split    : Split entities into WH/WT/AH/AT groups
  4. scoring  : Score chunks with a CrossEncoder model

Usage examples
--------------
# Run the full pipeline with the default config:
  python preprocess.py

# Run only the entity and chunk steps:
  python preprocess.py --steps entity chunk

# Use a custom config file:
  python preprocess.py --config my_config.yaml

# Override specific config values on the fly:
  python preprocess.py --set paths.out_dir=/tmp/mcq scoring.device=cpu

# Dry-run: show resolved config and exit:
  python preprocess.py --dry-run
"""

import argparse
import sys
from pathlib import Path

import yaml


# ============================================================
# Config helpers
# ============================================================

def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """
    Apply key=value overrides to a nested config dict.
    Key uses dot notation, e.g. "paths.out_dir=/tmp/out".
    Values are yaml-parsed so integers/booleans/null work correctly.
    """
    for kv in overrides:
        if "=" not in kv:
            sys.exit(f"[error] --set requires key=value format, got: {kv!r}")
        key_str, val_str = kv.split("=", 1)
        keys  = key_str.strip().split(".")
        value = yaml.safe_load(val_str)

        node = cfg
        for k in keys[:-1]:
            if k not in node:
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value


STEPS = ["entity", "chunk", "split", "scoring"]

STEP_DESCRIPTIONS = {
    "entity":  "Extract head/tail entity lists from DBpedia & OpenAlex",
    "chunk":   "Scan corpora and collect context chunks per entity",
    "split":   "Split entities into WH/WT/AH/AT groups",
    "scoring": "Score chunks with a CrossEncoder model",
}


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--steps", nargs="+", choices=STEPS, default=STEPS, metavar="STEP",
        help=(
            "Which steps to run (default: all). "
            f"Choices: {STEPS}. "
            "Steps always run in fixed order: entity → chunk → split → scoring."
        ),
    )
    parser.add_argument(
        "--set", nargs="+", default=[], metavar="KEY=VALUE",
        help=(
            "Override config values using dot-notation. "
            "E.g. --set paths.out_dir=/tmp/mcq scoring.device=cpu"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print resolved config and selected steps, then exit.",
    )
    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    # Load & patch config
    config_path = args.config
    if not Path(config_path).exists():
        sys.exit(f"[error] Config file not found: {config_path}")

    cfg = _load_config(config_path)
    _apply_overrides(cfg, args.set)

    # Preserve original step order even if user specifies out-of-order
    selected = [s for s in STEPS if s in args.steps]

    print("=" * 60)
    print("MCQ Pipeline")
    print("=" * 60)
    print(f"Config : {config_path}")
    print(f"out_dir: {cfg['paths']['out_dir']}")
    print(f"Steps  : {' → '.join(selected)}")
    print("=" * 60)

    if args.dry_run:
        import json as _json
        print("\n[dry-run] Resolved config:")
        print(_json.dumps(cfg, indent=2, default=str))
        return

    # Lazy-import pipeline modules so torch/GPU imports happen only when needed
    if "entity" in selected:
        from pipeline import entity
        entity.run(cfg)

    if "chunk" in selected:
        from pipeline import chunk_extract
        chunk_extract.run(cfg)

    if "split" in selected:
        from pipeline import split
        split.run(cfg)

    if "scoring" in selected:
        from pipeline import scoring
        scoring.run(cfg)

    print("=" * 60)
    print("All selected steps completed successfully.")
    print(f"Outputs are in: {cfg['paths']['out_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
