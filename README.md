# autoresearch

Autonomous research loop for optimizing Hungarian bounding-box matching against a fixed golden benchmark.

## Project purpose

This repository is configured so an agent can run keep/discard experiments where:

- only `matcher.py` is tuned,
- `evaluate_matcher.py` is deterministic and fixed,
- one scalar objective drives optimization: `score = 1 - macro_f1` (lower is better).

## Core files

- `matcher.py`: tunable matching logic (cost terms, Hungarian assignment, unmatched handling).
- `evaluate_matcher.py`: fixed evaluator for search/holdout split metrics.
- `program.md`: autonomous experiment workflow instructions.
- `golden_set_rotation_v1.json`: benchmark dataset with split indices and sample-level labels.
- `results.tsv`: local run log template (`commit`, `score`, `status`, `description`, `runtime_ms`).

Legacy files `train.py` and `prepare.py` are retained for reference and are not part of the matcher optimization workflow.

## Golden set JSON schema

The evaluator expects these top-level keys:

- `version`
- `task`
- `bbox_format`
- `class_map`
- `generation_config`
- `splits`
- `samples`

### Splits

- `splits.search`: list of sample indices used for iterative optimization.
- `splits.holdout`: list of sample indices used for periodic generalization checks.

### Sample fields

Each sample should provide:

- `sample_id`
- `image`: `orig_path`, `aug_path`, `width`, `height`
- `original_anns`: list of `{ann_id, category_id, bbox_xywh}`
- `detections`: list of `{det_id, bbox_xywh, score, extra?}`
- `gold_matches`: list of `{ann_id, matched_det_id, category_id}`
- `augmentation`

Evaluation uses sample-level `gold_matches` as ground truth.

## Setup

Requirements:

- Python 3.10+
- `uv` (recommended)

Install dependencies:

```bash
uv sync
```

## Evaluation

Run search split:

```bash
python evaluate_matcher.py --golden-set golden_set_rotation_v1.json --split search
```

Run holdout split:

```bash
python evaluate_matcher.py --golden-set golden_set_rotation_v1.json --split holdout
```

Expected machine-parseable output:

```text
score: <float>
macro_f1: <float>
unmatched_rate: <float>
wrong_class_rate: <float>
runtime_ms: <float>
num_samples: <int>
```

Optional matcher overrides:

```bash
python evaluate_matcher.py --golden-set golden_set_rotation_v1.json --split search --config iou_weight=0.8 --config class_aware=true
```

## Experiment loop philosophy

- Keep evaluator and benchmark fixed.
- Edit only `matcher.py` per experiment.
- Commit before each run, keep/discard by score improvement.
- Run holdout every 10 accepted improvements.
- Log run outcomes in local `results.tsv`.

See `program.md` for the full autonomous loop procedure.

## License

MIT
