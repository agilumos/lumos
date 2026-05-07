# MCQ Pipeline

Retrieves head/tail entities and their context chunks from text corpora,
then scores the chunks with a CrossEncoder model.

---

## Project Structure

```
.
├── preprocess.py
├── pipeline/
│   ├── entity.py        # Step 1: DBpedia & OpenAlex entity extraction
│   ├── chunk_extract.py # Step 2: Aho-Corasick corpus scanning
│   ├── split.py         # Step 3: WH/WT/AH/AT group splitting
│   ├── scoring.py       # Step 4: CrossEncoder chunk scoring
│   └── utils.py         # Shared utilities (normalization, readers, …)
├── prepare_craft_data.py        # Step 5: pre-CRAFT
└── config.yaml
```

---

## Setup

```bash
pip install ahocorasick zstandard sentence-transformers pyyaml tqdm
```

---

## Configuration

Edit **`config.yaml`** before running.  
The most important fields are under `paths:`:

| Key | Description |
|-----|-------------|
| `paths.kg_dir` | Directory containing DBpedia `mappingbased-objects_lang=en.ttl.bz2` |
| `paths.openalex_glob` | Glob for OpenAlex concept snapshot `.gz` files |
| `paths.web_corpus` | Path to the web corpus JSONL file |
| `paths.academic_corpus_dir` | Directory containing `pes2o-*.json` files |
| `paths.out_dir` | Root output directory (created automatically) |

---

## Usage

### Run the full pipeline

```bash
python preprocess.py
```

### Run specific steps only

```bash
# Steps always execute in fixed order: entity → chunk → split → scoring
python preprocess.py --steps entity chunk
python preprocess.py --steps split scoring
```

### Use a custom config file

```bash
python preprocess.py --config my_config.yaml
```

### Override config values on the fly

```bash
python preprocess.py --set paths.out_dir=/tmp/mcq scoring.device=cpu
```

### Dry-run (show resolved config without executing)

```bash
python preprocess.py --dry-run
```

---

## Pipeline Steps & Outputs

```
Step 1 — entity
  {out_dir}/entities/dbpedia_entity.json
  {out_dir}/entities/openalex_entity.json

Step 2 — chunk
  {out_dir}/chunks/freq_web.jsonl
  {out_dir}/chunks/freq_academic.jsonl
  {out_dir}/chunks/chunk_web.jsonl.zst
  {out_dir}/chunks/chunk_academic.jsonl.zst

Step 3 — split
  {out_dir}/splits/freq_{WH,WT,AH,AT}.jsonl
  {out_dir}/splits/chunk_{WH,WT,AH,AT}.jsonl

Step 4 — scoring
  {out_dir}/scored/chunk_{WH,WT,AH,AT}_scored.jsonl

Step 5 — (pre-CRAFT) run prepare_craft_data.py to generate retrieved.jsonl for each group, then launch the CRAFT MCQ generation pipeline.
```

Groups: **W** = Web, **A** = Academic, **H** = Head, **T** = Tail

---

## MCQ Generation (CRAFT)

After completing all 5 steps above, use the [CRAFT](https://github.com/ziegler-ingo/CRAFT) repository to generate MCQs from the retrieved chunks.

```bash
git clone https://github.com/ziegler-ingo/CRAFT
```

Refer to the CRAFT repository for setup and usage instructions.

---

## Credits

Entity extraction logic adapted from
[facebookresearch/head-to-tail](https://github.com/facebookresearch/head-to-tail)
(CC BY-NC 4.0).
