# autoresearch program

This repository is now focused on autonomous optimization of Hungarian bounding-box matching.

## Scope and invariants

- Editable experiment target: `matcher.py` only.
- Fixed evaluator: `evaluate_matcher.py` (read-only during experiment loop).
- Fixed benchmark data: a golden JSON file with split indices and sample-level `gold_matches`.
- Primary objective metric: `score = 1 - macro_f1` (lower is better).

Do not edit `evaluate_matcher.py` or the golden JSON while running experiments.

## Setup

1. Create or switch to an experiment branch, e.g. `autoresearch/<tag>`.
2. Confirm the benchmark JSON path exists (for example `golden_set_rotation_v1.json`).
3. Ensure dependencies are installed: `uv sync`.
4. Ensure `results.tsv` exists locally with this header:

```tsv
commit\tscore\tstatus\tdescription\truntime_ms
```

## Single run command

```bash
python evaluate_matcher.py --golden-set <path-to-golden-set.json> --split search > run.log 2>&1
```

Parse key metrics:

```bash
grep "^score:\|^macro_f1:\|^runtime_ms:" run.log
```

Expected evaluator output keys:

- `score`
- `macro_f1`
- `unmatched_rate`
- `wrong_class_rate`
- `runtime_ms`
- `num_samples`

## Autonomous keep/discard loop

Repeat forever:

1. Record the current commit as the rollback point.
2. Edit only `matcher.py` with one concrete matching hypothesis.
3. Commit the change.
4. Run the evaluator on `search` split and parse `score`.
5. If run crashes or no `score` is produced:
   - mark status as `crash`
   - log `score` as `0.000000` if needed for local bookkeeping
   - revert to rollback commit
   - continue
6. Compare against best accepted `score`:
   - lower `score`: keep commit (`status=keep`)
   - equal/higher `score`: discard by resetting to rollback commit (`status=discard`)
7. Append a line to `results.tsv`:
   - `commit`, `score`, `status`, `description`, `runtime_ms`

### Holdout cadence

Every 10 accepted (`keep`) improvements, run:

```bash
python evaluate_matcher.py --golden-set <path-to-golden-set.json> --split holdout > holdout.log 2>&1
```

Parse and record holdout `score` and runtime the same way.

## Research guidance

- Prefer simple, interpretable matcher edits first, then progressively test richer cost designs.
- Keep behavior deterministic.
- Handle edge cases robustly:
  - no annotations
  - no detections
  - invalid or duplicated IDs
- Treat evaluator and golden benchmark as fixed infrastructure.
- Explore both:
  1) hyperparameter tuning (weights, gates, thresholds), and
  2) cost-function design changes (add/remove/reshape cost terms).
- Allowed cost-term experiments include, for example:
  - IoU/GIoU variants
  - center distance / normalized geometry penalties
  - scale/aspect-ratio consistency
  - confidence-aware priors
  - appearance terms (e.g., color histogram similarity, Bhattacharyya distance)
- Any new cost term must:
  - be ablated (on/off or weight=0 vs >0),
  - report runtime impact,
  - improve score on search split before being kept,
  - be validated periodically on holdout to avoid overfitting.
