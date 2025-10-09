"""Tests for box IoU closeness metrics."""

import numpy as np
from frame2kg_eval.metrics.boxes import (
    box_iou_stats,
    aggregate_iou_micro,
    aggregate_iou_macro,
)


def test_box_iou_stats_with_matrix():
    # Two matched pairs with known IoU values
    p_nodes = [
        {"id": "p1", "location": [0.0, 0.0, 0.5, 0.5]},
        {"id": "p2", "location": [0.5, 0.5, 1.0, 1.0]},
    ]
    g_nodes = [
        {"id": "g1", "location": [0.0, 0.0, 0.5, 0.5]},
        {"id": "g2", "location": [0.25, 0.25, 0.75, 0.75]},
    ]
    mapping = {0: 0, 1: 1}

    # Precompute IoU matrix manually
    iou_matrix = np.array([
        [1.0, 0.25],
        [0.0, 0.25],
    ], dtype=np.float32)

    stats = box_iou_stats(p_nodes, g_nodes, mapping, iou_matrix=iou_matrix)
    assert stats["count"] == 2
    assert abs(stats["mean_iou"] - 0.625) < 1e-6  # (1.0 + 0.25)/2
    assert abs(stats["median_iou"] - 0.625) < 1e-6
    assert abs(stats["min_iou"] - 0.25) < 1e-6
    assert abs(stats["max_iou"] - 1.0) < 1e-6
    assert stats["std_iou"] >= 0.0
    assert stats["match_ious"] == (1.0, 0.25)


def test_box_iou_stats_no_matches():
    p_nodes = [{"id": "p1", "location": [0.0, 0.0, 0.1, 0.1]}]
    g_nodes = [{"id": "g1", "location": [0.9, 0.9, 1.0, 1.0]}]
    mapping = {}

    stats = box_iou_stats(p_nodes, g_nodes, mapping, iou_matrix=None)
    assert stats == {
        "mean_iou": 0.0,
        "median_iou": 0.0,
        "std_iou": 0.0,
        "min_iou": 0.0,
        "max_iou": 0.0,
        "count": 0,
        "match_ious": (),
    }


def test_box_iou_stats_fallback_compute():
    # No iou_matrix provided -> compute from boxes directly
    p_nodes = [
        {"id": "p1", "location": [0.0, 0.0, 0.4, 0.4]},
        {"id": "p2", "location": [0.5, 0.5, 0.7, 0.7]},
    ]
    g_nodes = [
        {"id": "g1", "location": [0.2, 0.2, 0.6, 0.6]},  # partial overlap with p1
        {"id": "g2", "location": [0.5, 0.5, 0.7, 0.7]},  # perfect with p2
    ]
    mapping = {0: 0, 1: 1}

    stats = box_iou_stats(p_nodes, g_nodes, mapping)
    # Known values from test_iou.py partial overlap example ~0.1428 and 1.0
    assert stats["count"] == 2
    assert 0.55 < stats["mean_iou"] < 0.60
    assert 0.55 < stats["median_iou"] < 0.60
    assert stats["max_iou"] == 1.0
    assert stats["min_iou"] < 0.2
    assert len(stats["match_ious"]) == 2


def test_aggregate_iou_micro_macro():
    frames = [
        {"mean_iou": 0.8, "median_iou": 0.8, "count": 5, "match_ious": (0.8,) * 5},
        {"mean_iou": 0.4, "median_iou": 0.4, "count": 5, "match_ious": (0.4,) * 5},
        {"mean_iou": 0.0, "median_iou": 0.0, "count": 0, "match_ious": ()},  # ignored by macro
    ]

    micro = aggregate_iou_micro(frames)
    assert micro["count"] == 10
    assert abs(micro["mean_iou"] - 0.6) < 1e-6  # (0.8*5 + 0.4*5)/10
    assert abs(micro["median_iou"] - 0.6) < 1e-6  # Median of combined values

    macro = aggregate_iou_macro(frames)
    assert macro["frame_count"] == 2
    assert abs(macro["mean_iou"] - 0.6) < 1e-6  # (0.8 + 0.4)/2
    assert abs(macro["median_iou"] - 0.6) < 1e-6
