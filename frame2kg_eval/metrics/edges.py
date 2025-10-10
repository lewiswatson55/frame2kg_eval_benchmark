"""Edge-level precision, recall, and F1 metrics."""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from scipy.optimize import linear_sum_assignment

from frame2kg_eval.matching.text import TextSimilarityComputer
from frame2kg_eval.utils.normalise import normalise_predicate


def edge_prf1(
    p_edges: List[Dict],
    g_edges: List[Dict],
    node_mapping: Dict[str, str],
    predicate_mode: str = "exact",
    *,
    semantic_threshold: float = 0.6,
    model_name: Optional[str] = None
) -> Dict:
    """Compute edge precision, recall, and F1.
    
    Edges match when both endpoints map correctly and predicates match.
    
    Args:
        p_edges: List of predicted edges
        g_edges: List of ground truth edges
        node_mapping: Mapping from predicted node IDs to GT node IDs
        predicate_mode: How to match predicates ("exact", "normalised", or "semantic")
        semantic_threshold: Minimum similarity required for semantic predicate matches
        model_name: Optional sentence transformer model name to override default
    
    Returns:
        Dictionary with metrics:
            - precision: Edge precision
            - recall: Edge recall
            - f1: Edge F1 score
            - tp: True positive count
            - fp: False positive count
            - fn: False negative count
            - support: Number of GT edges
    """
    # Build GT edge signatures for matching
    if predicate_mode not in {"exact", "normalised", "semantic"}:
        raise ValueError(f"Unsupported predicate_mode: {predicate_mode}")

    matched_pred_edges: Set[int]
    matched_gt_edges: Set[int]

    if predicate_mode in {"exact", "normalised"}:
        gt_edge_sigs = set()
        for g_edge in g_edges:
            if predicate_mode == "exact":
                pred_value = g_edge["predicate"]
            else:
                pred_value = normalise_predicate(g_edge["predicate"])

            sig = (g_edge["source"], g_edge["target"], pred_value)
            gt_edge_sigs.add(sig)

        matched_pred_edges = set()
        matched_gt_edges = set()

        for i, p_edge in enumerate(p_edges):
            p_src = p_edge["source"]
            p_tgt = p_edge["target"]

            if p_src not in node_mapping or p_tgt not in node_mapping:
                continue

            mapped_src = node_mapping[p_src]
            mapped_tgt = node_mapping[p_tgt]

            if predicate_mode == "exact":
                pred_value = p_edge["predicate"]
            else:
                pred_value = normalise_predicate(p_edge["predicate"])

            sig = (mapped_src, mapped_tgt, pred_value)
            if sig in gt_edge_sigs:
                matched_pred_edges.add(i)
        tp = len(matched_pred_edges)
        fp = len(p_edges) - tp
        fn = len(g_edges) - tp
    else:  # semantic predicate matching
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

        matched_pred_edges = set()
        matched_gt_edges = set()

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
                    matched_pred_edges.add(pred_edge_idx)
                    matched_gt_edges.add(gt_edge_idx)

        tp = len(matched_pred_edges)
        fp = len(p_edges) - tp
        fn = len(g_edges) - len(matched_gt_edges)
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": len(g_edges)
    }


def edge_by_label_baseline(
    p_edges: List[Dict],
    g_edges: List[Dict],
    p_nodes: List[Dict],
    g_nodes: List[Dict]
) -> Dict:
    """Edge-by-label baseline that ignores node IDs.
    
    Matches edges based on (source_label, target_label, predicate) triples.
    
    Args:
        p_edges: Predicted edges
        g_edges: Ground truth edges
        p_nodes: Predicted nodes (for label lookup)
        g_nodes: Ground truth nodes (for label lookup)
    
    Returns:
        Same metrics dictionary as edge_prf1
    """
    # Build node ID to label mappings
    p_id_to_label = {n["id"]: n.get("label", "") for n in p_nodes}
    g_id_to_label = {n["id"]: n.get("label", "") for n in g_nodes}
    
    # Build GT edge signatures by label
    gt_edge_sigs = []
    for g_edge in g_edges:
        src_label = g_id_to_label.get(g_edge["source"], "")
        tgt_label = g_id_to_label.get(g_edge["target"], "")
        pred = normalise_predicate(g_edge["predicate"])
        sig = (src_label, tgt_label, pred)
        gt_edge_sigs.append(sig)
    
    # Count matches using greedy assignment
    tp = 0
    used_gt = set()
    
    for p_edge in p_edges:
        src_label = p_id_to_label.get(p_edge["source"], "")
        tgt_label = p_id_to_label.get(p_edge["target"], "")
        pred = normalise_predicate(p_edge["predicate"])
        sig = (src_label, tgt_label, pred)
        
        # Find first unused matching GT edge
        for j, gt_sig in enumerate(gt_edge_sigs):
            if j not in used_gt and sig == gt_sig:
                tp += 1
                used_gt.add(j)
                break
    
    fp = len(p_edges) - tp
    fn = len(g_edges) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": len(g_edges)
    }


def edge_prf1_by_predicate(
    p_edges: List[Dict],
    g_edges: List[Dict],
    node_mapping: Dict[str, str],
    predicate_mode: str = "exact",
    *,
    semantic_threshold: float = 0.6,
    model_name: Optional[str] = None
) -> Dict[str, Dict]:
    """Compute per-predicate edge metrics.
    
    Args:
        p_edges: Predicted edges
        g_edges: Ground truth edges
        node_mapping: Node ID mapping
        predicate_mode: Predicate matching mode
        semantic_threshold: Minimum similarity required for semantic predicate matches
        model_name: Optional sentence transformer model name
    
    Returns:
        Dictionary mapping predicates to their metrics
    """
    per_predicate_metrics = {}
    
    # Group edges by predicate
    pred_by_predicate = {}
    for p_edge in p_edges:
        pred = p_edge["predicate"]
        if pred not in pred_by_predicate:
            pred_by_predicate[pred] = []
        pred_by_predicate[pred].append(p_edge)
    
    gt_by_predicate = {}
    for g_edge in g_edges:
        pred = g_edge["predicate"]
        if pred not in gt_by_predicate:
            gt_by_predicate[pred] = []
        gt_by_predicate[pred].append(g_edge)
    
    # Get all unique predicates
    all_predicates = set(pred_by_predicate.keys()) | set(gt_by_predicate.keys())
    
    for predicate in all_predicates:
        pred_edges = pred_by_predicate.get(predicate, [])
        gt_edges = gt_by_predicate.get(predicate, [])
        
        # Compute metrics for this predicate
        metrics = edge_prf1(
            pred_edges,
            gt_edges,
            node_mapping,
            predicate_mode,
            semantic_threshold=semantic_threshold,
            model_name=model_name,
        )
        per_predicate_metrics[predicate] = metrics
    
    return per_predicate_metrics
