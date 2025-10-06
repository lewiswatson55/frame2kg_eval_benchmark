"""Intersection over Union (IoU) utilities for bounding boxes."""

import numpy as np
from typing import List, Optional, Tuple


def compute_iou(box1: Tuple[float, float, float, float], 
                box2: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two bounding boxes.
    
    Args:
        box1: First box as (x1, y1, x2, y2)
        box2: Second box as (x1, y1, x2, y2)
    
    Returns:
        IoU score between 0 and 1
    """
    if box1 is None or box2 is None:
        return 0.0
    
    # Compute intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check for no overlap
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Compute areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute union
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    if union <= 0:
        return 0.0
    
    return intersection / union


def compute_iou_matrix(pred_boxes: List[Optional[Tuple]], 
                      gt_boxes: List[Optional[Tuple]]) -> np.ndarray:
    """Compute IoU matrix between all pairs of boxes.
    
    Args:
        pred_boxes: List of predicted boxes (can contain None)
        gt_boxes: List of ground truth boxes (can contain None)
    
    Returns:
        Matrix of shape (len(pred_boxes), len(gt_boxes)) with IoU scores
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
    
    for i in range(n_pred):
        for j in range(n_gt):
            if pred_boxes[i] is not None and gt_boxes[j] is not None:
                iou_matrix[i, j] = compute_iou(pred_boxes[i], gt_boxes[j])
    
    return iou_matrix


def filter_by_iou_threshold(iou_matrix: np.ndarray, 
                           threshold: float) -> np.ndarray:
    """Create binary mask for IoU values above threshold.
    
    Args:
        iou_matrix: IoU scores matrix
        threshold: Minimum IoU value to pass
    
    Returns:
        Boolean matrix of same shape indicating which pairs pass threshold
    """
    return iou_matrix >= threshold
