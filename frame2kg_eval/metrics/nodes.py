"""Node-level precision, recall, and F1 metrics."""

from typing import Dict, List, Optional


def node_prf1(
    p_nodes: List[Dict],
    g_nodes: List[Dict],
    mapping: Dict[int, int]
) -> Dict:
    """Compute node precision, recall, and F1.
    
    Args:
        p_nodes: List of predicted nodes
        g_nodes: List of ground truth nodes
        mapping: Dictionary mapping prediction indices to GT indices
    
    Returns:
        Dictionary with metrics:
            - precision: Node precision
            - recall: Node recall
            - f1: Node F1 score
            - tp: True positive count
            - fp: False positive count
            - fn: False negative count
            - support: Number of GT nodes
    """
    n_pred = len(p_nodes)
    n_gt = len(g_nodes)
    
    # Count true positives (matched nodes)
    tp = len(mapping)
    
    # Count false positives (unmatched predictions)
    fp = n_pred - tp
    
    # Count false negatives (unmatched ground truth)
    fn = n_gt - tp
    
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
        "support": n_gt
    }


def node_prf1_by_label(
    p_nodes: List[Dict],
    g_nodes: List[Dict],
    mapping: Dict[int, int]
) -> Dict[str, Dict]:
    """Compute per-label node metrics.
    
    Args:
        p_nodes: List of predicted nodes
        g_nodes: List of ground truth nodes
        mapping: Dictionary mapping prediction indices to GT indices
    
    Returns:
        Dictionary mapping labels to their metrics
    """
    per_label_metrics = {}
    
    # Collect nodes by label
    pred_by_label = {}
    for i, node in enumerate(p_nodes):
        label = node.get("label", "unknown")
        if label not in pred_by_label:
            pred_by_label[label] = []
        pred_by_label[label].append(i)
    
    gt_by_label = {}
    for j, node in enumerate(g_nodes):
        label = node.get("label", "unknown")
        if label not in gt_by_label:
            gt_by_label[label] = []
        gt_by_label[label].append(j)
    
    # Get all unique labels
    all_labels = set(pred_by_label.keys()) | set(gt_by_label.keys())
    
    for label in all_labels:
        pred_indices = set(pred_by_label.get(label, []))
        gt_indices = set(gt_by_label.get(label, []))
        
        # Count matches for this label
        tp = 0
        for p_idx, g_idx in mapping.items():
            if p_idx in pred_indices and g_idx in gt_indices:
                # Check that labels actually match
                if (p_nodes[p_idx].get("label") == label and 
                    g_nodes[g_idx].get("label") == label):
                    tp += 1
        
        fp = len(pred_indices) - tp
        fn = len(gt_indices) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_label_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": len(gt_indices)
        }
    
    return per_label_metrics


def aggregate_micro(metrics_list: List[Dict]) -> Dict:
    """Aggregate metrics using micro-averaging (sum counts then compute).
    
    Args:
        metrics_list: List of metric dictionaries with tp/fp/fn counts
    
    Returns:
        Micro-averaged metrics
    """
    total_tp = sum(m.get("tp", 0) for m in metrics_list)
    total_fp = sum(m.get("fp", 0) for m in metrics_list)
    total_fn = sum(m.get("fn", 0) for m in metrics_list)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "support": sum(m.get("support", 0) for m in metrics_list)
    }


def aggregate_macro(metrics_list: List[Dict]) -> Dict:
    """Aggregate metrics using macro-averaging (average of metrics).
    
    Args:
        metrics_list: List of metric dictionaries
    
    Returns:
        Macro-averaged metrics
    """
    if not metrics_list:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "support": 0
        }
    
    # Filter out NaN values
    valid_precisions = [m["precision"] for m in metrics_list if m.get("precision") is not None]
    valid_recalls = [m["recall"] for m in metrics_list if m.get("recall") is not None]
    valid_f1s = [m["f1"] for m in metrics_list if m.get("f1") is not None]
    
    precision = sum(valid_precisions) / len(valid_precisions) if valid_precisions else 0.0
    recall = sum(valid_recalls) / len(valid_recalls) if valid_recalls else 0.0
    f1 = sum(valid_f1s) / len(valid_f1s) if valid_f1s else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": sum(m.get("tp", 0) for m in metrics_list),
        "fp": sum(m.get("fp", 0) for m in metrics_list),
        "fn": sum(m.get("fn", 0) for m in metrics_list),
        "support": sum(m.get("support", 0) for m in metrics_list)
    }
