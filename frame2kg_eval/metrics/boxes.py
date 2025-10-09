"""Matched-pair IoU (box IoU) metrics.

Primary per-frame statistic: mean IoU across matched node pairs.

Per-frame outputs include:
- mean_iou: Average IoU across matched pairs (0.0 if none)
- median_iou: Median IoU across matched pairs (0.0 if none)
- std_iou: Standard deviation of IoU values (0.0 if none)
- min_iou, max_iou: Extremes (0.0 if none)
- count: Number of matched pairs

NB: Mean could in theory be too brittle; since IoU values are clamped
to [0,1] we still obtain relatively stable values even with outliers.
Median is also provided for robustness.

NB 2: Only matched boxes are considered - FP/FN boxes are captured via
precision/recall-style metrics elsewhere.

"""

from typing import Dict, List, Optional
import numpy as np
import warnings


def _warn_if_invalid_box(box, node_ctx: str) -> None:
    """Warn if a single box is invalid (x2<x1 or y2<y1).

    Args:
        box: Sequence like [x1, y1, x2, y2]
        node_ctx: Context string to identify the node (e.g., id/type)
    """
    if box is None:
        return
    try:
        x1, y1, x2, y2 = box  # type: ignore[misc]
    except Exception:
        return
    if (x2 < x1) or (y2 < y1):
        warnings.warn(
            f"Invalid bounding box detected for {node_ctx}: "
            f"x2<x1 or y2<y1 with box [{x1}, {y1}, {x2}, {y2}]",
            stacklevel=2,
        )


def box_iou_stats(
    p_nodes: List[Dict],
    g_nodes: List[Dict],
    mapping: Dict[int, int],
    iou_matrix: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute IoU statistics for matched node pairs.

    Args:
        p_nodes: Standardized predicted nodes (with `location` as [x1,y1,x2,y2] or None)
        g_nodes: Standardized GT nodes (with `location` as [x1,y1,x2,y2] or None)
        mapping: Dict mapping predicted indices to GT indices (from assignment)
        iou_matrix: Optional precomputed IoU matrix from matching stage

    Returns:
        Dict with keys: mean_iou, median_iou, std_iou, min_iou, max_iou, count, match_ious
    """
    ious: List[float] = []

    # Warn about any invalid boxes upfront (both preds and GT), once per node
    for idx, n in enumerate(p_nodes):
        _warn_if_invalid_box(n.get("location"), f"pred node '{n.get('id', idx)}'")
    for idx, n in enumerate(g_nodes):
        _warn_if_invalid_box(n.get("location"), f"GT node '{n.get('id', idx)}'")

    if iou_matrix is not None:
        for p_idx, g_idx in mapping.items():
            val = float(iou_matrix[p_idx, g_idx])
            if np.isfinite(val):
                ious.append(max(0.0, min(1.0, val)))
    else:
        # Fallback: compute IoU directly using iou matcher
        from frame2kg_eval.matching.iou import compute_iou
        for p_idx, g_idx in mapping.items():
            p_box = p_nodes[p_idx].get("location")
            g_box = g_nodes[g_idx].get("location")

            if p_box is not None and g_box is not None:
                iou = compute_iou(tuple(p_box), tuple(g_box))
                ious.append(iou)

    if not ious:
        return {
            "mean_iou": 0.0,
            "median_iou": 0.0,
            "std_iou": 0.0,
            "min_iou": 0.0,
            "max_iou": 0.0,
            "count": 0,
            "match_ious": (),
        }

    arr = np.asarray(ious, dtype=np.float32)
    return {
        "mean_iou": float(arr.mean()),
        "median_iou": float(np.median(arr)),
        "std_iou": float(arr.std(ddof=0)),
        "min_iou": float(arr.min()),
        "max_iou": float(arr.max()),
        "count": int(arr.size),
        "match_ious": tuple(float(v) for v in arr),
    }


def aggregate_iou_micro(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Matched-pair IoU micro-average (weighted by matched pair count).

    Args:
        stats_list: Per-frame stats from `box_iou_stats`

    Returns:
        Dict with mean_iou, median_iou, and total count
    """
    total_count = 0
    weighted_sum = 0.0
    all_ious: List[float] = []
    for s in stats_list:
        c = int(s.get("count", 0))
        m = float(s.get("mean_iou", 0.0))
        total_count += c
        weighted_sum += m * c
        if c > 0:
            match_ious = s.get("match_ious")
            if match_ious:
                all_ious.extend(float(v) for v in match_ious)
    mean = (weighted_sum / total_count) if total_count > 0 else 0.0
    median = float(np.median(np.asarray(all_ious, dtype=np.float32))) if all_ious else 0.0
    return {"mean_iou": mean, "median_iou": median, "count": total_count}


def aggregate_iou_macro(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Matched-pair IoU macro-average (unweighted mean of per-frame means).

    Args:
        stats_list: Per-frame stats from `box_iou_stats`

    Returns:
        Dict with mean_iou, median_iou, and frame_count (frames that had at least one match)
    """
    filtered = [s for s in stats_list if int(s.get("count", 0)) > 0]
    if not filtered:
        return {"mean_iou": 0.0, "median_iou": 0.0, "frame_count": 0}

    means = [float(s.get("mean_iou", 0.0)) for s in filtered]
    medians = [float(s.get("median_iou", 0.0)) for s in filtered]

    mean_value = float(sum(means) / len(means))
    median_value = float(np.median(np.asarray(medians, dtype=np.float32)))

    return {"mean_iou": mean_value, "median_iou": median_value, "frame_count": len(filtered)}
