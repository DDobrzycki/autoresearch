"""Tunable Hungarian bbox matcher.

This is the only file that should be iteratively edited by the experiment loop.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class MatcherConfig:
    """Hyperparameters controlling cost construction and assignment behavior."""

    iou_weight: float = 0.7208003261911645
    center_weight: float = 0.1654450067360511
    size_weight: float = 0.028696543417234266
    score_weight: float = 0.08505812365555007
    unmatched_cost: float = 0.568929743995518
    min_iou_for_match: float = 0.06337313291259677
    min_detection_score: float = 0.0
    low_score_penalty: float = 0.34809839295390094
    class_aware: bool = False
    class_mismatch_penalty: float = 0.50
    class_mismatch_mode: str = "soft"  # "soft" or "hard"

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "MatcherConfig":
        """Create config from untyped override mapping."""
        if not config:
            return cls()

        data: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in config:
                data[f.name] = config[f.name]

        cfg = cls(
            iou_weight=_safe_float(data.get("iou_weight", cls.iou_weight), cls.iou_weight),
            center_weight=_safe_float(data.get("center_weight", cls.center_weight), cls.center_weight),
            size_weight=_safe_float(data.get("size_weight", cls.size_weight), cls.size_weight),
            score_weight=_safe_float(data.get("score_weight", cls.score_weight), cls.score_weight),
            unmatched_cost=_safe_float(data.get("unmatched_cost", cls.unmatched_cost), cls.unmatched_cost),
            min_iou_for_match=_safe_float(data.get("min_iou_for_match", cls.min_iou_for_match), cls.min_iou_for_match),
            min_detection_score=_safe_float(data.get("min_detection_score", cls.min_detection_score), cls.min_detection_score),
            low_score_penalty=_safe_float(data.get("low_score_penalty", cls.low_score_penalty), cls.low_score_penalty),
            class_aware=_safe_bool(data.get("class_aware", cls.class_aware), cls.class_aware),
            class_mismatch_penalty=_safe_float(data.get("class_mismatch_penalty", cls.class_mismatch_penalty), cls.class_mismatch_penalty),
            class_mismatch_mode=str(data.get("class_mismatch_mode", cls.class_mismatch_mode)),
        )
        if cfg.class_mismatch_mode not in {"soft", "hard"}:
            return MatcherConfig(**{**cfg.__dict__, "class_mismatch_mode": "soft"})
        return cfg


def match_sample(
    original_anns: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    image_meta: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[int, int | None]:
    """Match each original annotation to at most one detection.

    Args:
        original_anns: Annotation list with fields including ann_id, category_id, bbox_xywh.
        detections: Detection list with fields including det_id, bbox_xywh, score, and optional class hints.
        image_meta: Image metadata dictionary (width/height are used when available).
        config: Optional matcher config overrides.

    Returns:
        Mapping {ann_id: matched_det_id or None}.
    """
    cfg = MatcherConfig.from_mapping(config)
    anns = list(original_anns or [])
    dets = list(detections or [])

    if not anns:
        return {}

    ann_ids, ann_fallback = _unique_int_ids([ann.get("ann_id") for ann in anns], fallback_start=-1)
    if not dets:
        return {ann_id: None for ann_id in ann_ids}

    det_ids, _ = _unique_int_ids([det.get("det_id") for det in dets], fallback_start=ann_fallback)

    ann_boxes = np.vstack([_parse_bbox_xywh(ann.get("bbox_xywh")) for ann in anns])
    det_boxes = np.vstack([_parse_bbox_xywh(det.get("bbox_xywh")) for det in dets])

    iou = _pairwise_iou_xywh(ann_boxes, det_boxes)
    center_dist = _pairwise_center_dist_norm(ann_boxes, det_boxes, image_meta)
    log_area_delta = _pairwise_log_area_delta(ann_boxes, det_boxes)
    det_scores = np.array([_safe_float(det.get("score"), 0.0) for det in dets], dtype=np.float64)
    det_scores = np.clip(det_scores, 0.0, 1.0)

    cost = (
        cfg.iou_weight * (1.0 - iou)
        + cfg.center_weight * center_dist
        + cfg.size_weight * log_area_delta
        + cfg.score_weight * (1.0 - det_scores[np.newaxis, :])
    )

    if cfg.min_detection_score > 0.0:
        low_score_mask = det_scores < cfg.min_detection_score
        if np.any(low_score_mask):
            cost[:, low_score_mask] += cfg.low_score_penalty

    if cfg.class_aware:
        ann_cats = [_safe_int_or_none(ann.get("category_id")) for ann in anns]
        det_cats = [_extract_detection_category(det) for det in dets]
        if cfg.class_mismatch_mode == "hard":
            mismatch_penalty = 1e6
        else:
            mismatch_penalty = cfg.class_mismatch_penalty
        for i, ann_cat in enumerate(ann_cats):
            if ann_cat is None:
                continue
            for j, det_cat in enumerate(det_cats):
                if det_cat is None:
                    continue
                if ann_cat != det_cat:
                    cost[i, j] += mismatch_penalty

    num_anns = len(anns)
    unmatched_block = np.full((num_anns, num_anns), cfg.unmatched_cost, dtype=np.float64)
    full_cost = np.concatenate([cost, unmatched_block], axis=1)

    rows, cols = linear_sum_assignment(full_cost)

    result: dict[int, int | None] = {ann_id: None for ann_id in ann_ids}
    assigned_col = np.full(num_anns, -1, dtype=np.int64)
    assigned_col[rows] = cols

    for row_idx, ann_id in enumerate(ann_ids):
        col_idx = int(assigned_col[row_idx])
        if col_idx < 0 or col_idx >= len(dets):
            result[ann_id] = None
            continue
        if iou[row_idx, col_idx] < cfg.min_iou_for_match:
            result[ann_id] = None
            continue
        result[ann_id] = det_ids[col_idx]

    return result


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Parse a finite float, returning default on failure."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _safe_bool(value: Any, default: bool = False) -> bool:
    """Parse common bool-like values with a conservative fallback."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _safe_int_or_none(value: Any) -> int | None:
    """Best-effort integer parsing for IDs and categories."""
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
    if isinstance(value, str):
        text = value.strip()
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
    return None


def _unique_int_ids(raw_ids: Sequence[Any], fallback_start: int = -1) -> tuple[list[int], int]:
    """Coerce IDs to unique ints, using decreasing negative fallbacks when needed."""
    ids: list[int] = []
    used: set[int] = set()
    fallback = fallback_start

    for idx, raw_id in enumerate(raw_ids):
        parsed = _safe_int_or_none(raw_id)
        if parsed is None:
            parsed = fallback
        if parsed in used:
            parsed = fallback
        while parsed in used:
            fallback -= 1
            parsed = fallback
        used.add(parsed)
        ids.append(parsed)
        if parsed <= fallback:
            fallback = parsed - 1

    return ids, fallback


def _parse_bbox_xywh(raw_bbox: Any) -> np.ndarray:
    """Parse bbox into finite [x, y, w, h] with non-negative width/height."""
    arr = np.asarray(raw_bbox if raw_bbox is not None else [0.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(-1)
    if arr.size < 4:
        arr = np.pad(arr, (0, 4 - arr.size))
    x, y, w, h = (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
    if not math.isfinite(x):
        x = 0.0
    if not math.isfinite(y):
        y = 0.0
    if not math.isfinite(w):
        w = 0.0
    if not math.isfinite(h):
        h = 0.0
    w = max(w, 0.0)
    h = max(h, 0.0)
    return np.array([x, y, w, h], dtype=np.float64)


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [x, y, w, h] boxes to [x1, y1, x2, y2]."""
    out = boxes.copy()
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out


def _pairwise_iou_xywh(a_xywh: np.ndarray, b_xywh: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU for two xywh box arrays."""
    if a_xywh.size == 0 or b_xywh.size == 0:
        return np.zeros((a_xywh.shape[0], b_xywh.shape[0]), dtype=np.float64)

    a = _xywh_to_xyxy(a_xywh)
    b = _xywh_to_xyxy(b_xywh)

    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, a_xywh[:, 2] * a_xywh[:, 3])
    area_b = np.maximum(0.0, b_xywh[:, 2] * b_xywh[:, 3])
    union = area_a[:, None] + area_b[None, :] - inter

    iou = np.zeros_like(inter)
    valid = union > 0.0
    iou[valid] = inter[valid] / union[valid]
    return iou


def _pairwise_center_dist_norm(a_xywh: np.ndarray, b_xywh: np.ndarray, image_meta: Mapping[str, Any]) -> np.ndarray:
    """Compute center-to-center distance normalized by image diagonal."""
    if a_xywh.size == 0 or b_xywh.size == 0:
        return np.zeros((a_xywh.shape[0], b_xywh.shape[0]), dtype=np.float64)

    a_cx = a_xywh[:, 0] + 0.5 * a_xywh[:, 2]
    a_cy = a_xywh[:, 1] + 0.5 * a_xywh[:, 3]
    b_cx = b_xywh[:, 0] + 0.5 * b_xywh[:, 2]
    b_cy = b_xywh[:, 1] + 0.5 * b_xywh[:, 3]

    dx = a_cx[:, None] - b_cx[None, :]
    dy = a_cy[:, None] - b_cy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)

    width = _safe_float(image_meta.get("width"), 0.0)
    height = _safe_float(image_meta.get("height"), 0.0)
    if width <= 0.0 or height <= 0.0:
        max_x = float(np.max(np.concatenate([a_xywh[:, 0] + a_xywh[:, 2], b_xywh[:, 0] + b_xywh[:, 2]]), initial=1.0))
        max_y = float(np.max(np.concatenate([a_xywh[:, 1] + a_xywh[:, 3], b_xywh[:, 1] + b_xywh[:, 3]]), initial=1.0))
        width = max(width, max_x)
        height = max(height, max_y)

    diag = math.hypot(max(width, 1.0), max(height, 1.0))
    return dist / diag


def _pairwise_log_area_delta(a_xywh: np.ndarray, b_xywh: np.ndarray) -> np.ndarray:
    """Compute pairwise absolute log-area differences."""
    if a_xywh.size == 0 or b_xywh.size == 0:
        return np.zeros((a_xywh.shape[0], b_xywh.shape[0]), dtype=np.float64)

    eps = 1e-6
    area_a = np.maximum(a_xywh[:, 2] * a_xywh[:, 3], eps)
    area_b = np.maximum(b_xywh[:, 2] * b_xywh[:, 3], eps)
    return np.abs(np.log(area_a[:, None]) - np.log(area_b[None, :]))


def _extract_detection_category(det: Mapping[str, Any]) -> int | None:
    """Extract detector-side category hints when available."""
    cat = _safe_int_or_none(det.get("category_id"))
    if cat is not None:
        return cat

    extra = det.get("extra")
    if not isinstance(extra, Mapping):
        return None

    for key in (
        "pred_category_id",
        "predicted_category_id",
        "COCO_pretrained_model_pred_category_id",
    ):
        if key in extra:
            cat = _safe_int_or_none(extra.get(key))
            if cat is not None:
                return cat
    return None
