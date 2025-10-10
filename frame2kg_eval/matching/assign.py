"""Two-stage node matching with Hungarian assignment."""

from collections import defaultdict
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from scipy.optimize import linear_sum_assignment
from frame2kg_eval.matching.iou import compute_iou_matrix
from frame2kg_eval.matching.text import TextSimilarityComputer


def two_stage_node_match(
    p_nodes: List[Dict],
    g_nodes: List[Dict],
    *,
    tau: float,
    alpha: float,
    text_mode: str = "tfidf",
    text_fields: Tuple[str, ...] = ("id", "label"),
    text_floor: float = 0.25,
    sim_cache: Optional[Dict] = None
) -> Dict:
    """Two-stage node matching with IoU gating and text similarity.
    
    Args:
        p_nodes: List of predicted nodes
        g_nodes: List of ground truth nodes
        tau: IoU threshold for gating
        alpha: Blending weight for combining IoU and text similarity
        text_mode: Text similarity mode ("tfidf", "semantic", "hybrid")
        text_fields: Node fields to use for text similarity
        text_floor: Minimum text similarity to consider
        sim_cache: Optional cache for embeddings (passed to TextSimilarityComputer)
    
    Returns:
        Dictionary containing:
            - mapping: Dict[int, int] mapping pred indices to gt indices
            - unmatched_pred: Set of unmatched prediction indices
            - unmatched_gt: Set of unmatched ground truth indices
            - matrices: Dict with iou, text_sim, and score matrices
    """
    n_pred = len(p_nodes)
    n_gt = len(g_nodes)
    
    # Handle empty cases
    if n_pred == 0 or n_gt == 0:
        return {
            "mapping": {},
            "unmatched_pred": set(range(n_pred)),
            "unmatched_gt": set(range(n_gt)),
            "matrices": {
                "iou": np.zeros((n_pred, n_gt), dtype=np.float32),
                "text_sim": np.zeros((n_pred, n_gt), dtype=np.float32),
                "score": np.zeros((n_pred, n_gt), dtype=np.float32)
            }
        }
    
    # Extract bounding boxes
    pred_boxes = [n.get("location") for n in p_nodes]
    gt_boxes = [n.get("location") for n in g_nodes]
    
    # Compute IoU matrix
    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    
    # Compute text similarity matrix
    text_computer = TextSimilarityComputer(mode=text_mode)
    if sim_cache is not None and hasattr(text_computer, '_embedding_cache'):
        text_computer._embedding_cache = sim_cache
    
    text_sim_matrix = text_computer.compute_similarity_matrix(
        p_nodes, g_nodes, text_fields=text_fields
    )
    
    # Apply text floor
    text_mask = text_sim_matrix >= text_floor
    
    # Apply IoU gating
    iou_mask = iou_matrix >= tau
    
    # Combined mask: both conditions must be met
    valid_mask = iou_mask & text_mask
    
    # Compute blended scores
    score_matrix = np.full((n_pred, n_gt), -np.inf, dtype=np.float32)
    score_matrix[valid_mask] = (
        alpha * iou_matrix[valid_mask] + 
        (1 - alpha) * text_sim_matrix[valid_mask]
    )
    
    # Hungarian assignment
    mapping = {}
    if np.any(np.isfinite(score_matrix)):
        # Convert to cost matrix (negate scores)
        cost_matrix = -score_matrix
        
        # Replace inf with large value for scipy
        max_cost = np.abs(np.nanmax(cost_matrix[np.isfinite(cost_matrix)])) + 1
        cost_matrix[np.isinf(cost_matrix)] = max_cost
        
        # Run Hungarian algorithm
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out invalid assignments (inf score)
        for p_idx, g_idx in zip(pred_indices, gt_indices):
            if np.isfinite(score_matrix[p_idx, g_idx]) and score_matrix[p_idx, g_idx] > 0:
                mapping[int(p_idx)] = int(g_idx)
    
    # Identify unmatched nodes
    matched_pred = set(mapping.keys())
    matched_gt = set(mapping.values())
    unmatched_pred = set(range(n_pred)) - matched_pred
    unmatched_gt = set(range(n_gt)) - matched_gt
    
    return {
        "mapping": mapping,
        "unmatched_pred": unmatched_pred,
        "unmatched_gt": unmatched_gt,
        "matrices": {
            "iou": iou_matrix,
            "text_sim": text_sim_matrix,
            "score": score_matrix
        }
    }


def compute_edge_mapping(
    p_edges: List[Dict],
    g_edges: List[Dict],
    node_mapping: Dict[str, str],
    predicate_mode: str = "exact",
    *,
    semantic_threshold: float = 0.6,
    model_name: Optional[str] = None
) -> Dict[int, int]:
    """Map edges based on node mapping and predicate matching.
    
    Args:
        p_edges: Predicted edges
        g_edges: Ground truth edges
        node_mapping: Mapping from predicted node IDs to GT node IDs
        predicate_mode: How to match predicates ("exact" or "normalised")
    
    Returns:
        Dict mapping predicted edge indices to GT edge indices
    """
    edge_mapping: Dict[int, int] = {}

    if predicate_mode in {"exact", "normalised"}:
        # Build GT edge signatures for fast lookup
        gt_edge_sigs: Dict[Tuple[str, str, str], int] = {}
        for j, g_edge in enumerate(g_edges):
            if predicate_mode == "exact":
                pred_key = g_edge["predicate"]
            else:  # normalised
                from frame2kg_eval.utils.normalise import normalise_predicate
                pred_key = normalise_predicate(g_edge["predicate"])

            sig = (g_edge["source"], g_edge["target"], pred_key)
            gt_edge_sigs[sig] = j

        # Try to match each predicted edge
        for i, p_edge in enumerate(p_edges):
            p_src = p_edge["source"]
            p_tgt = p_edge["target"]

            if p_src not in node_mapping or p_tgt not in node_mapping:
                continue

            mapped_src = node_mapping[p_src]
            mapped_tgt = node_mapping[p_tgt]

            if predicate_mode == "exact":
                pred_key = p_edge["predicate"]
            else:
                from frame2kg_eval.utils.normalise import normalise_predicate
                pred_key = normalise_predicate(p_edge["predicate"])

            sig = (mapped_src, mapped_tgt, pred_key)
            if sig in gt_edge_sigs:
                edge_mapping[i] = gt_edge_sigs[sig]
    elif predicate_mode == "semantic":
        text_computer = TextSimilarityComputer(mode="semantic", model_name=model_name)

        pred_groups: Dict[Tuple[str, str], List[Tuple[int, str]]] = defaultdict(list)
        for i, p_edge in enumerate(p_edges):
            p_src = p_edge["source"]
            p_tgt = p_edge["target"]

            if p_src not in node_mapping or p_tgt not in node_mapping:
                continue

            mapped_src = node_mapping[p_src]
            mapped_tgt = node_mapping[p_tgt]
            pred_groups[(mapped_src, mapped_tgt)].append((i, p_edge["predicate"]))

        gt_groups: Dict[Tuple[str, str], List[Tuple[int, str]]] = defaultdict(list)
        for j, g_edge in enumerate(g_edges):
            gt_groups[(g_edge["source"], g_edge["target"])].append((j, g_edge["predicate"]))

        for key, preds in pred_groups.items():
            gt_candidates = gt_groups.get(key)
            if not gt_candidates:
                continue

            pred_texts = [pred for _, pred in preds]
            gt_texts = [pred for _, pred in gt_candidates]

            if not pred_texts or not gt_texts:
                continue

            similarity = text_computer.compute_semantic_similarity(pred_texts, gt_texts)
            if similarity.size == 0:
                continue

            cost_matrix = 1.0 - similarity
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r_idx, c_idx in zip(row_ind, col_ind):
                score = float(similarity[r_idx, c_idx])
                if score >= semantic_threshold:
                    pred_edge_idx = preds[r_idx][0]
                    gt_edge_idx = gt_candidates[c_idx][0]
                    edge_mapping[pred_edge_idx] = gt_edge_idx
    else:
        raise ValueError(f"Unsupported predicate_mode: {predicate_mode}")

    return edge_mapping
