"""Composite-aware diagnostic metrics for split/merge node mismatches.

This module provides diagnostic metrics to explain node FN/FP that arise from
compositional mismatches (e.g., "fruit_basket" vs "apple+banana+orange+basket").
These are DIAGNOSTIC ONLY and do not affect primary F1 scores.

Note: Directions (many→1 and 1→many) are evaluated independently, so a node
could theoretically be used as both anchor and candidate in different directions.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
from frame2kg_eval.matching.text import TextSimilarityComputer
from frame2kg_eval.matching.iou import compute_iou


def parse_location(location: Union[str, List, Tuple, None]) -> Optional[List[float]]:
    """Parse location from various formats to [x1, y1, x2, y2].
    
    Args:
        location: Location as string "x1,y1,x2,y2,conf" or list/tuple
    
    Returns:
        Parsed [x1, y1, x2, y2] or None if invalid
    """
    if location is None:
        return None
    
    if isinstance(location, str):
        # Parse "x1,y1,x2,y2,confidence" format
        parts = location.split(',')
        if len(parts) >= 4:
            try:
                x1, y1, x2, y2 = map(float, parts[:4])
                # Clamp to [0, 1] and ensure valid box
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                x1 = max(0.0, min(1.0, x1))
                y1 = max(0.0, min(1.0, y1))
                x2 = max(0.0, min(1.0, x2))
                y2 = max(0.0, min(1.0, y2))
                return [x1, y1, x2, y2]
            except (ValueError, TypeError):
                return None
    
    if isinstance(location, (list, tuple)) and len(location) >= 4:
        try:
            x1, y1, x2, y2 = float(location[0]), float(location[1]), float(location[2]), float(location[3])
            # Clamp and ensure valid
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            return [x1, y1, x2, y2]
        except (ValueError, TypeError):
            return None
    
    return None


def compute_box_union(boxes: List[Optional[List[float]]]) -> Optional[List[float]]:
    """Compute the axis-aligned bounding rectangle that contains all boxes.
    
    Note: This returns the minimal axis-aligned rectangle, not a polygonal union.
    
    Args:
        boxes: List of boxes in [x1, y1, x2, y2] format
    
    Returns:
        Bounding rectangle in [x1, y1, x2, y2] format, or None if no valid boxes
    """
    valid_boxes = [b for b in boxes if b is not None and len(b) == 4]
    if not valid_boxes:
        return None
    
    x1 = min(b[0] for b in valid_boxes)
    y1 = min(b[1] for b in valid_boxes)
    x2 = max(b[2] for b in valid_boxes)
    y2 = max(b[3] for b in valid_boxes)
    
    return [x1, y1, x2, y2]


def center_in_box(point_box: List[float], container_box: List[float]) -> bool:
    """Check if center of point_box is inside container_box.
    
    Args:
        point_box: Box whose center to check
        container_box: Container box
    
    Returns:
        True if center is inside
    """
    if point_box is None or container_box is None:
        return False
    
    center_x = (point_box[0] + point_box[2]) / 2
    center_y = (point_box[1] + point_box[3]) / 2
    
    return (container_box[0] <= center_x <= container_box[2] and
            container_box[1] <= center_y <= container_box[3])


def compute_composite_score(
    group_indices: List[int],
    anchor_idx: int,
    group_nodes: List[Dict],
    anchor_nodes: List[Dict],
    group_texts: List[str],
    anchor_texts: List[str],
    text_computer: TextSimilarityComputer,
    beta: float = 0.6
) -> float:
    """Compute composite matching score between a group and an anchor node.
    
    Args:
        group_indices: Indices of nodes in the group
        anchor_idx: Index of anchor node
        group_nodes: All candidate nodes
        anchor_nodes: All anchor nodes
        group_texts: Pre-extracted texts for group nodes
        anchor_texts: Pre-extracted texts for anchor nodes
        text_computer: Text similarity computer
        beta: Weight for IoU vs text similarity (default 0.6)
    
    Returns:
        Composite score in [0, 1]
    """
    # Get actual nodes and texts
    group = [group_nodes[i] for i in group_indices]
    anchor_node = anchor_nodes[anchor_idx]
    
    # Compute spatial score (IoU of union box with anchor)
    group_boxes = [parse_location(n.get("location")) for n in group]
    union_box = compute_box_union(group_boxes)
    anchor_box = parse_location(anchor_node.get("location"))
    
    iou_score = 0.0
    if union_box is not None and anchor_box is not None:
        iou_score = compute_iou(tuple(union_box), tuple(anchor_box))
    
    # Compute text similarity using pre-extracted texts
    group_text_list = [group_texts[i] for i in group_indices if group_texts[i]]
    anchor_text = anchor_texts[anchor_idx]
    
    text_score = 0.0
    if group_text_list and anchor_text:
        combined_group_text = " ".join(group_text_list)
        # Use semantic similarity for composite matching
        similarity_matrix = text_computer.compute_semantic_similarity(
            [combined_group_text], [anchor_text]
        )
        # Clamp cosine similarity to [-1, 1]
        text_score = float(np.clip(similarity_matrix[0, 0], -1.0, 1.0))
    
    # Blend scores
    return beta * iou_score + (1 - beta) * text_score


def find_composite_groups(
    anchor_nodes: List[Dict],
    candidate_nodes: List[Dict],
    unmatched_anchors: Set[int],
    unmatched_candidates: Set[int],
    text_computer: Optional[TextSimilarityComputer] = None,
    beta: float = 0.6,
    tau_c: float = 0.2,
    theta: float = 0.55,
    kmax: int = 4,
    eps: float = 0.01,
    text_fields: Tuple[str, ...] = ("label", "attributes"),
    max_candidates: int = 12
) -> Tuple[int, Dict[int, List[int]], Dict[str, float]]:
    """Find composite groups that explain unmatched nodes.
    
    Args:
        anchor_nodes: Nodes to find groups for
        candidate_nodes: Nodes to form groups from
        unmatched_anchors: Indices of unmatched anchor nodes
        unmatched_candidates: Indices of unmatched candidate nodes
        text_computer: Text similarity computer (will create if None)
        beta: Weight for IoU vs text similarity (default 0.6)
        tau_c: Spatial gating threshold (default 0.2)
        theta: Composite acceptance threshold (default 0.55)
        kmax: Maximum group size (default 4)
        eps: Minimum improvement threshold (default 0.01)
        text_fields: Fields to use for text extraction
        max_candidates: Max candidates after prefiltering (default 12)
    
    Returns:
        Tuple of (number of composite hits, mapping of anchor to group indices, statistics)
    """
    if text_computer is None:
        text_computer = TextSimilarityComputer(mode="semantic")
    
    # Pre-extract texts for all nodes by index
    anchor_texts = [text_computer.extract_node_text(node, text_fields) for node in anchor_nodes]
    candidate_texts = [text_computer.extract_node_text(node, text_fields) for node in candidate_nodes]
    
    used_candidates = set()
    hits = 0
    composite_mappings = {}
    group_sizes = []
    composite_scores = []
    
    for anchor_idx in unmatched_anchors:
        anchor = anchor_nodes[anchor_idx]
        anchor_box = parse_location(anchor.get("location"))
        
        # Find spatial candidates
        spatial_candidates = []
        for cand_idx in unmatched_candidates:
            if cand_idx in used_candidates:
                continue
            
            candidate = candidate_nodes[cand_idx]
            cand_box = parse_location(candidate.get("location"))
            
            # Check spatial overlap
            if anchor_box and cand_box:
                iou = compute_iou(tuple(cand_box), tuple(anchor_box))
                if iou >= tau_c or center_in_box(cand_box, anchor_box):
                    spatial_candidates.append((cand_idx, candidate, iou))
        
        # Cap candidates by text similarity for speed (keep top max_candidates)
        if len(spatial_candidates) > max_candidates:
            anchor_text = anchor_texts[anchor_idx]
            if anchor_text:
                # Batch compute similarities for all candidates at once
                cand_text_list = [candidate_texts[idx] for idx, _, _ in spatial_candidates]
                valid_indices = [i for i, t in enumerate(cand_text_list) if t]
                
                if valid_indices:
                    valid_texts = [cand_text_list[i] for i in valid_indices]
                    # Compute all similarities in one batch
                    sim_matrix = text_computer.compute_semantic_similarity(valid_texts, [anchor_text])
                    sims = np.clip(sim_matrix.ravel(), -1.0, 1.0)
                    
                    # Score candidates
                    scored_candidates = []
                    sim_idx = 0
                    for i, (cand_idx, candidate, iou) in enumerate(spatial_candidates):
                        if i in valid_indices:
                            text_sim = float(sims[sim_idx])
                            sim_idx += 1
                        else:
                            text_sim = 0.0
                        scored_candidates.append((cand_idx, candidate, text_sim))
                    
                    # Sort by text similarity, then by index for determinism
                    scored_candidates.sort(key=lambda x: (-x[2], x[0]))
                    candidates = [(c[0], c[1]) for c in scored_candidates[:max_candidates]]
                else:
                    # No valid texts, just take first max_candidates by index
                    candidates = [(c[0], c[1]) for c in sorted(spatial_candidates, key=lambda x: x[0])[:max_candidates]]
            else:
                # No anchor text, take first max_candidates by index  
                candidates = [(c[0], c[1]) for c in sorted(spatial_candidates, key=lambda x: x[0])[:max_candidates]]
        else:
            candidates = [(c[0], c[1]) for c in spatial_candidates]
        
        if not candidates:
            continue
        
        # Find best single candidate
        best_group = []
        best_score = -1.0
        best_indices = []
        
        for cand_idx, candidate in candidates:
            score = compute_composite_score(
                [cand_idx], anchor_idx, candidate_nodes, anchor_nodes,
                candidate_texts, anchor_texts, text_computer, beta
            )
            # Deterministic tie-breaking: prefer lower index
            if score > best_score or (abs(score - best_score) < 1e-9 and cand_idx < (best_indices[0] if best_indices else float('inf'))):
                best_group = [candidate]
                best_score = score
                best_indices = [cand_idx]
        
        # Greedy expansion
        remaining = set(c[0] for c in candidates) - set(best_indices)
        
        while len(best_group) < kmax and remaining:
            improved = False
            next_pick = None
            next_score = best_score
            
            for cand_idx in remaining:
                candidate = candidate_nodes[cand_idx]
                test_group = best_group + [candidate]
                
                test_indices = best_indices + [cand_idx]
                score = compute_composite_score(
                    test_indices, anchor_idx, candidate_nodes, anchor_nodes,
                    candidate_texts, anchor_texts, text_computer, beta
                )
                
                # Deterministic tie-breaking: prefer lower index (including candidate 0)
                if score > next_score + eps or (
                    abs(score - next_score - eps) < 1e-9
                    and cand_idx < (next_pick if next_pick is not None else float('inf'))
                ):
                    improved = True
                    next_pick = cand_idx
                    next_score = score
            
            if not improved:
                break
            
            best_group.append(candidate_nodes[next_pick])
            best_indices.append(next_pick)
            remaining.remove(next_pick)
            best_score = next_score
        
        # Check if group meets threshold
        if best_score >= theta and len(best_group) >= 2:
            hits += 1
            composite_mappings[anchor_idx] = best_indices
            group_sizes.append(len(best_indices))
            composite_scores.append(best_score)
            for idx in best_indices:
                used_candidates.add(idx)
    
    # Compute statistics
    stats = {
        "avg_group_size": np.mean(group_sizes) if group_sizes else 0.0,
        "mean_composite_score": np.mean(composite_scores) if composite_scores else 0.0,
    }
    
    return hits, composite_mappings, stats


def composite_diagnostics(
    p_nodes: List[Dict],
    g_nodes: List[Dict],
    mapping: Dict[int, int],
    text_computer: Optional[TextSimilarityComputer] = None,
    beta: float = 0.6,
    tau_c: float = 0.2,
    theta: float = 0.55,
    kmax: int = 4,
    eps: float = 0.01,
    text_fields: Tuple[str, ...] = ("label", "attributes"),
    max_candidates: int = 12
) -> Dict:
    """Compute composite diagnostic metrics.
    
    Args:
        p_nodes: Predicted nodes
        g_nodes: Ground truth nodes
        mapping: Node index mapping from primary matching
        text_computer: Text similarity computer (will create if None)
        beta: Weight for IoU vs text similarity (default 0.6)
        tau_c: Spatial gating threshold (default 0.2)
        theta: Composite acceptance threshold (default 0.55)
        kmax: Maximum group size (default 4)
        eps: Minimum improvement threshold (default 0.01)
        text_fields: Fields to use for text extraction (default: label, attributes)
        max_candidates: Max candidates to consider after text prefiltering (default 12)
    
    Returns:
        Dictionary with diagnostic metrics including:
        - composite_hits_gt_side: Unmatched GT explained by pred groups
        - composite_hits_pred_side: Unmatched pred explained by GT groups
        - composite_fn_explained_pct: % of FN explained
        - composite_fp_explained_pct: % of FP explained
        - composite_adjusted_recall: Diagnostic recall with composites
        - Statistics: avg group sizes and mean composite scores
    """
    # Identify unmatched nodes
    matched_pred = set(mapping.keys())
    matched_gt = set(mapping.values())
    unmatched_pred = set(range(len(p_nodes))) - matched_pred
    unmatched_gt = set(range(len(g_nodes))) - matched_gt
    
    # Original FN/FP counts
    original_fn = len(unmatched_gt)
    original_fp = len(unmatched_pred)
    
    # Reuse text computer if provided, otherwise create one
    if text_computer is None:
        text_computer = TextSimilarityComputer(mode="semantic")
    
    # Find composite groups: GT side (many preds → one GT)
    hits_gt, mappings_gt, stats_gt = find_composite_groups(
        g_nodes, p_nodes,
        unmatched_gt, unmatched_pred,
        text_computer, beta, tau_c, theta, kmax, eps, text_fields, max_candidates
    )
    
    # Find composite groups: Pred side (many GTs → one pred)  
    hits_pred, mappings_pred, stats_pred = find_composite_groups(
        p_nodes, g_nodes,
        unmatched_pred, unmatched_gt,
        text_computer, beta, tau_c, theta, kmax, eps, text_fields, max_candidates
    )
    
    # Calculate percentages
    pct_fn_explained = (hits_gt / original_fn * 100) if original_fn > 0 else 0.0
    pct_fp_explained = (hits_pred / original_fp * 100) if original_fp > 0 else 0.0
    
    # Compute composite-adjusted recall (diagnostic only)
    tp = len(mapping)
    composite_recall = ((tp + hits_gt) / len(g_nodes)) if len(g_nodes) > 0 else 0.0
    
    return {
        "composite_hits_gt_side": int(hits_gt),
        "composite_hits_pred_side": int(hits_pred),
        "composite_fn_explained_pct": float(pct_fn_explained),
        "composite_fp_explained_pct": float(pct_fp_explained),
        "composite_adjusted_recall": float(composite_recall),
        "composite_mappings_gt": mappings_gt,
        "composite_mappings_pred": mappings_pred,
        "avg_group_size_gt": float(stats_gt["avg_group_size"]),
        "avg_group_size_pred": float(stats_pred["avg_group_size"]),
        "mean_composite_score_gt": float(stats_gt["mean_composite_score"]),
        "mean_composite_score_pred": float(stats_pred["mean_composite_score"]),
    }
