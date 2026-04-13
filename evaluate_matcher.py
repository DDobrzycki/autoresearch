"""Fixed evaluator for Hungarian bbox matching experiments.

The experiment loop should only edit matcher.py and invoke this evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping

from matcher import match_sample


def main() -> int:
    """CLI entry point for fixed matcher evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate matcher.py against a fixed golden benchmark.")
    parser.add_argument("--golden-set", required=True, help="Path to golden benchmark JSON file.")
    parser.add_argument("--split", choices=["search", "holdout"], default="search", help="Which split to evaluate.")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Optional matcher override in key=value format. Repeatable.",
    )
    parser.add_argument(
        "--config-json",
        default=None,
        help="Optional JSON object (or path to JSON file) with matcher config overrides.",
    )
    args = parser.parse_args()

    try:
        matcher_config = _parse_matcher_config(args.config, args.config_json)
        dataset = _load_golden_set(Path(args.golden_set))
        selected_samples = _select_split_samples(dataset, args.split)
        metrics = _evaluate_samples(selected_samples, matcher_config)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    score = 1.0 - metrics["macro_f1"]

    # Print this exact machine-parseable summary.
    print(f"score: {score:.6f}")
    print(f"macro_f1: {metrics['macro_f1']:.6f}")
    print(f"unmatched_rate: {metrics['unmatched_rate']:.6f}")
    print(f"wrong_class_rate: {metrics['wrong_class_rate']:.6f}")
    print(f"runtime_ms: {metrics['runtime_ms']:.6f}")
    print(f"num_samples: {metrics['num_samples']}")
    return 0


def _parse_matcher_config(config_items: list[str], config_json: str | None) -> dict[str, Any]:
    """Parse optional matcher override flags into a dictionary."""
    config: dict[str, Any] = {}

    if config_json:
        as_path = Path(config_json)
        raw_text = as_path.read_text(encoding="utf-8") if as_path.exists() else config_json
        parsed = json.loads(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError("--config-json must decode to a JSON object")
        config.update(parsed)

    for item in config_items:
        if "=" not in item:
            raise ValueError(f"Invalid --config entry '{item}'. Expected key=value format.")
        key, value_text = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --config entry '{item}'. Empty key.")
        config[key] = _parse_config_value(value_text.strip())

    return config


def _parse_config_value(raw: str) -> Any:
    """Parse scalar/list/object value using JSON semantics when possible."""
    # JSON parser gives robust typed parsing for numbers/bool/null/lists/dicts.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _load_golden_set(path: Path) -> dict[str, Any]:
    """Load and minimally validate golden benchmark JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Golden set file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = {"version", "task", "bbox_format", "class_map", "generation_config", "splits", "samples"}
    missing = sorted(required_keys - set(data.keys()))
    if missing:
        raise ValueError(f"Golden set missing required top-level keys: {', '.join(missing)}")

    if not isinstance(data.get("samples"), list):
        raise ValueError("Golden set 'samples' must be a list")
    if not isinstance(data.get("splits"), Mapping):
        raise ValueError("Golden set 'splits' must be an object")

    return data


def _select_split_samples(dataset: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    """Resolve split indices into sample objects in deterministic order."""
    split_indices = dataset.get("splits", {}).get(split)
    if not isinstance(split_indices, list):
        raise ValueError(f"Golden set split '{split}' is missing or not a list")

    all_samples = dataset.get("samples", [])
    selected: list[dict[str, Any]] = []
    for raw_idx in split_indices:
        idx = _safe_int_or_none(raw_idx)
        if idx is None:
            continue
        if 0 <= idx < len(all_samples):
            sample = all_samples[idx]
            if isinstance(sample, Mapping):
                selected.append(dict(sample))
    return selected


def _evaluate_samples(samples: list[dict[str, Any]], matcher_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Run matcher across samples and aggregate benchmark metrics."""
    class_stats: dict[int, dict[str, int]] = {}
    total_anns = 0
    unmatched_predictions = 0
    wrong_class_predictions = 0
    total_predicted_matches = 0
    matcher_runtime_s = 0.0

    for sample in samples:
        original_anns = sample.get("original_anns") or []
        detections = sample.get("detections") or []
        image_meta = sample.get("image") or {}
        gold_matches = sample.get("gold_matches") or []

        anns_sanitized, ann_norm_to_sid = _sanitize_annotations(original_anns)
        dets_sanitized, det_norm_to_sid, det_sid_to_category = _sanitize_detections(detections)
        gold_by_ann_sid = _build_gold_lookup(gold_matches, ann_norm_to_sid, det_norm_to_sid)

        t0 = time.perf_counter()
        predicted = match_sample(anns_sanitized, dets_sanitized, image_meta, config=matcher_config)
        matcher_runtime_s += time.perf_counter() - t0

        pred_by_ann_sid = _coerce_prediction_mapping(predicted)

        for ann in anns_sanitized:
            ann_sid = ann["ann_id"]
            ann_cat = _safe_int_or_none(ann.get("category_id"))
            pred_det_sid = pred_by_ann_sid.get(ann_sid)
            gold_det_sid = gold_by_ann_sid.get(ann_sid)

            total_anns += 1
            if pred_det_sid is None:
                unmatched_predictions += 1
            else:
                total_predicted_matches += 1
                det_cat = det_sid_to_category.get(pred_det_sid)
                if ann_cat is not None and det_cat is not None and ann_cat != det_cat:
                    wrong_class_predictions += 1

            if ann_cat is None:
                continue

            stats = class_stats.setdefault(ann_cat, {"tp": 0, "fp": 0, "fn": 0, "count": 0})
            stats["count"] += 1

            if gold_det_sid is not None and pred_det_sid == gold_det_sid:
                stats["tp"] += 1
            else:
                if pred_det_sid is not None:
                    stats["fp"] += 1
                if gold_det_sid is not None:
                    stats["fn"] += 1

    macro_f1 = _macro_f1(class_stats)
    unmatched_rate = (unmatched_predictions / total_anns) if total_anns else 0.0
    wrong_class_rate = (wrong_class_predictions / total_predicted_matches) if total_predicted_matches else 0.0

    return {
        "macro_f1": macro_f1,
        "unmatched_rate": unmatched_rate,
        "wrong_class_rate": wrong_class_rate,
        "runtime_ms": matcher_runtime_s * 1000.0,
        "num_samples": len(samples),
    }


def _sanitize_annotations(original_anns: list[Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build unique integer ann IDs for matcher execution and lookup."""
    sanitized: list[dict[str, Any]] = []
    ann_norm_to_sid: dict[str, int] = {}
    used_ids: set[int] = set()
    fallback_id = -1

    for ann in original_anns:
        if not isinstance(ann, Mapping):
            continue
        sid = _safe_int_or_none(ann.get("ann_id"))
        if sid is None or sid in used_ids:
            while fallback_id in used_ids:
                fallback_id -= 1
            sid = fallback_id
            fallback_id -= 1

        used_ids.add(sid)
        ann_norm = _normalize_id(ann.get("ann_id"))
        if ann_norm is None:
            ann_norm = str(sid)
        ann_norm_to_sid.setdefault(ann_norm, sid)

        ann_copy = dict(ann)
        ann_copy["ann_id"] = sid
        sanitized.append(ann_copy)

    return sanitized, ann_norm_to_sid


def _sanitize_detections(
    detections: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[int, int | None]]:
    """Build unique integer det IDs and extract optional detector class hints."""
    sanitized: list[dict[str, Any]] = []
    det_norm_to_sid: dict[str, int] = {}
    det_sid_to_category: dict[int, int | None] = {}
    used_ids: set[int] = set()
    fallback_id = -1

    for det in detections:
        if not isinstance(det, Mapping):
            continue
        sid = _safe_int_or_none(det.get("det_id"))
        if sid is None or sid in used_ids:
            while fallback_id in used_ids:
                fallback_id -= 1
            sid = fallback_id
            fallback_id -= 1

        used_ids.add(sid)
        det_norm = _normalize_id(det.get("det_id"))
        if det_norm is None:
            det_norm = str(sid)
        det_norm_to_sid.setdefault(det_norm, sid)

        det_copy = dict(det)
        det_copy["det_id"] = sid
        sanitized.append(det_copy)
        det_sid_to_category[sid] = _extract_detection_category(det)

    return sanitized, det_norm_to_sid, det_sid_to_category


def _build_gold_lookup(
    gold_matches: list[Any],
    ann_norm_to_sid: Mapping[str, int],
    det_norm_to_sid: Mapping[str, int],
) -> dict[int, int | None]:
    """Map sanitized ann IDs to sanitized gold det IDs (or None)."""
    gold_by_ann_sid: dict[int, int | None] = {ann_sid: None for ann_sid in ann_norm_to_sid.values()}

    for entry in gold_matches:
        if not isinstance(entry, Mapping):
            continue

        ann_norm = _normalize_id(entry.get("ann_id"))
        if ann_norm is None:
            continue
        ann_sid = ann_norm_to_sid.get(ann_norm)
        if ann_sid is None:
            continue

        det_norm = _normalize_id(entry.get("matched_det_id"))
        det_sid = det_norm_to_sid.get(det_norm) if det_norm is not None else None
        gold_by_ann_sid[ann_sid] = det_sid

    return gold_by_ann_sid


def _coerce_prediction_mapping(predicted: Mapping[Any, Any]) -> dict[int, int | None]:
    """Normalize matcher output to sanitized int IDs."""
    out: dict[int, int | None] = {}
    for ann_key, det_key in predicted.items():
        ann_sid = _safe_int_or_none(ann_key)
        if ann_sid is None:
            continue
        if det_key is None:
            out[ann_sid] = None
            continue
        det_sid = _safe_int_or_none(det_key)
        out[ann_sid] = det_sid
    return out


def _extract_detection_category(det: Mapping[str, Any]) -> int | None:
    """Extract detector-side category hints when present."""
    direct = _safe_int_or_none(det.get("category_id"))
    if direct is not None:
        return direct

    extra = det.get("extra")
    if not isinstance(extra, Mapping):
        return None

    for key in (
        "pred_category_id",
        "predicted_category_id",
        "COCO_pretrained_model_pred_category_id",
    ):
        if key in extra:
            parsed = _safe_int_or_none(extra.get(key))
            if parsed is not None:
                return parsed
    return None


def _macro_f1(class_stats: Mapping[int, Mapping[str, int]]) -> float:
    """Compute macro F1 over annotation categories."""
    if not class_stats:
        return 1.0

    f1_values: list[float] = []
    for stats in class_stats.values():
        tp = int(stats.get("tp", 0))
        fp = int(stats.get("fp", 0))
        fn = int(stats.get("fn", 0))
        count = int(stats.get("count", 0))

        denom = (2 * tp) + fp + fn
        if denom <= 0:
            f1 = 1.0 if count > 0 else 0.0
        else:
            f1 = (2.0 * tp) / denom
        f1_values.append(f1)

    if not f1_values:
        return 1.0
    return float(sum(f1_values) / len(f1_values))


def _normalize_id(value: Any) -> str | None:
    """Normalize IDs to canonical strings for robust lookup."""
    if value is None:
        return None
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return str(int(value))
        return str(value)

    text = str(value).strip()
    if not text:
        return None
    try:
        if any(ch in text for ch in (".", "e", "E")):
            num = float(text)
            if not math.isfinite(num):
                return text
            if num.is_integer():
                return str(int(num))
            return str(num)
        return str(int(text))
    except ValueError:
        return text


def _safe_int_or_none(value: Any) -> int | None:
    """Best-effort integer parsing for IDs and category fields."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value)

    text = str(value).strip()
    if not text:
        return None
    try:
        if any(ch in text for ch in (".", "e", "E")):
            numeric = float(text)
            if not math.isfinite(numeric):
                return None
            return int(numeric)
        return int(text)
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
