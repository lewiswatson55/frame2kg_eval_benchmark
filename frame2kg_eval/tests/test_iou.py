"""Tests for IoU computation."""

import pytest
import numpy as np
from frame2kg_eval.matching.iou import (
    compute_iou, compute_iou_matrix, filter_by_iou_threshold
)


class TestIoU:
    
    def test_compute_iou_perfect_overlap(self):
        # Same box should have IoU = 1.0
        box = (0.2, 0.3, 0.5, 0.6)
        assert compute_iou(box, box) == 1.0
        
        # Different position but same box
        box2 = (0.0, 0.0, 1.0, 1.0)
        assert compute_iou(box2, box2) == 1.0
    
    def test_compute_iou_no_overlap(self):
        # Completely separate boxes
        box1 = (0.0, 0.0, 0.2, 0.2)
        box2 = (0.5, 0.5, 0.7, 0.7)
        assert compute_iou(box1, box2) == 0.0
        
        # Adjacent boxes (touching but not overlapping)
        box3 = (0.0, 0.0, 0.5, 0.5)
        box4 = (0.5, 0.5, 1.0, 1.0)
        assert compute_iou(box3, box4) == 0.0
    
    def test_compute_iou_partial_overlap(self):
        # 50% overlap
        box1 = (0.0, 0.0, 0.4, 0.4)
        box2 = (0.2, 0.2, 0.6, 0.6)
        
        # Intersection = (0.2-0.4) x (0.2-0.4) = 0.2 * 0.2 = 0.04
        # Area1 = 0.4 * 0.4 = 0.16
        # Area2 = 0.4 * 0.4 = 0.16
        # Union = 0.16 + 0.16 - 0.04 = 0.28
        # IoU = 0.04 / 0.28 = 0.1428...
        iou = compute_iou(box1, box2)
        assert abs(iou - 0.1428) < 0.001
        
        # Contained box (box2 inside box1)
        box3 = (0.0, 0.0, 1.0, 1.0)
        box4 = (0.25, 0.25, 0.75, 0.75)
        # Area of box4 = 0.5 * 0.5 = 0.25
        # Area of box3 = 1.0 * 1.0 = 1.0
        # Intersection = 0.25 (box4 entirely inside)
        # Union = 1.0 (box3 contains box4)
        # IoU = 0.25 / 1.0 = 0.25
        assert compute_iou(box3, box4) == 0.25
    
    def test_compute_iou_edge_cases(self):
        # Zero-area boxes (lines)
        line_h = (0.1, 0.5, 0.9, 0.5)  # Horizontal line (zero height)
        line_v = (0.5, 0.1, 0.5, 0.9)  # Vertical line (zero width)
        assert compute_iou(line_h, line_v) == 0.0
        
        # Point (zero area)
        point = (0.5, 0.5, 0.5, 0.5)
        box = (0.0, 0.0, 1.0, 1.0)
        assert compute_iou(point, box) == 0.0
    
    def test_compute_iou_none_handling(self):
        box = (0.1, 0.1, 0.2, 0.2)
        assert compute_iou(None, box) == 0.0
        assert compute_iou(box, None) == 0.0
        assert compute_iou(None, None) == 0.0
    
    def test_compute_iou_matrix(self):
        pred_boxes = [
            (0.0, 0.0, 0.2, 0.2),  # Box 1
            (0.5, 0.5, 0.7, 0.7),  # Box 2
            None,                   # None
            (0.1, 0.1, 0.3, 0.3)   # Box 3
        ]
        gt_boxes = [
            (0.0, 0.0, 0.2, 0.2),  # Perfect match with pred[0]
            (0.1, 0.1, 0.3, 0.3),  # Perfect match with pred[3]
            (0.8, 0.8, 1.0, 1.0)   # No match
        ]
        
        matrix = compute_iou_matrix(pred_boxes, gt_boxes)
        
        # Check shape
        assert matrix.shape == (4, 3)
        assert matrix.dtype == np.float32
        
        # Check specific values
        assert matrix[0, 0] == 1.0  # pred[0] matches gt[0] perfectly
        assert matrix[0, 1] > 0 and matrix[0, 1] < 1  # Partial overlap
        assert matrix[1, 0] == 0.0  # No overlap
        assert matrix[2, :].sum() == 0.0  # None box has no overlaps
        assert matrix[3, 1] == 1.0  # pred[3] matches gt[1] perfectly
        
        # Check symmetry property (IoU(A,B) == IoU(B,A))
        matrix_T = compute_iou_matrix(gt_boxes, pred_boxes)
        assert np.allclose(matrix, matrix_T.T)
    
    def test_filter_by_iou_threshold(self):
        iou_matrix = np.array([
            [0.9, 0.2, 0.0],
            [0.1, 0.6, 0.3],
            [0.0, 0.0, 0.8]
        ], dtype=np.float32)
        
        # Threshold = 0.5
        mask = filter_by_iou_threshold(iou_matrix, 0.5)
        expected = np.array([
            [True, False, False],
            [False, True, False],
            [False, False, True]
        ])
        assert np.array_equal(mask, expected)
        
        # Threshold = 0.0 (all pass)
        mask_all = filter_by_iou_threshold(iou_matrix, 0.0)
        assert mask_all.all()
        
        # Threshold = 1.0 (none pass)  
        mask_none = filter_by_iou_threshold(iou_matrix, 1.0)
        assert not mask_none.any()


if __name__ == "__main__":
    pytest.main([__file__])
