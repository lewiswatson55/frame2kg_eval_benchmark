"""Bounding-box closeness metrics (IoU-based).

Primary metric: mean IoU across matched node pairs.

Per-frame outputs include:
- mean_iou: Average IoU across matched pairs (0.0 if none)
- std_iou: Standard deviation of IoU values (0.0 if none)
- min_iou, max_iou: Extremes (0.0 if none)
- count: Number of matched pairs
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
        Dict with keys: mean_iou, std_iou, min_iou, max_iou, count
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
        return {"mean_iou": 0.0, "std_iou": 0.0, "min_iou": 0.0, "max_iou": 0.0, "count": 0}

    arr = np.asarray(ious, dtype=np.float32)
    return {
        "mean_iou": float(arr.mean()),
        "std_iou": float(arr.std(ddof=0)),
        "min_iou": float(arr.min()),
        "max_iou": float(arr.max()),
        "count": int(arr.size),
    }


def aggregate_iou_micro(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Micro-average IoU across frames (weighted by matched count).

    Args:
        stats_list: Per-frame stats from `box_iou_stats`

    Returns:
        Dict with mean_iou and total count
    """
    total_count = 0
    weighted_sum = 0.0
    for s in stats_list:
        c = int(s.get("count", 0))
        m = float(s.get("mean_iou", 0.0))
        total_count += c
        weighted_sum += m * c
    mean = (weighted_sum / total_count) if total_count > 0 else 0.0
    return {"mean_iou": mean, "count": total_count}


def aggregate_iou_macro(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Macro-average IoU across frames (unweighted average of per-frame means).

    Args:
        stats_list: Per-frame stats from `box_iou_stats`

    Returns:
        Dict with mean_iou and frame_count (frames that had at least one match)
    """
    means = [float(s.get("mean_iou", 0.0)) for s in stats_list if int(s.get("count", 0)) > 0]
    if not means:
        return {"mean_iou": 0.0, "frame_count": 0}
    return {"mean_iou": float(sum(means) / len(means)), "frame_count": len(means)}
