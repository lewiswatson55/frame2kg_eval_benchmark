"""Main evaluation CLI command."""

import click
import csv
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from frame2kg_eval.io.preds import PredictionLoader
from frame2kg_eval.io.groundtruth import create_ground_truth_adapter
from frame2kg_eval.matching.assign import two_stage_node_match
from frame2kg_eval.metrics.nodes import node_prf1, aggregate_micro, aggregate_macro
from frame2kg_eval.metrics.edges import edge_prf1, edge_by_label_baseline
from frame2kg_eval.metrics.validity import compute_validity_from_directory
from frame2kg_eval.metrics.conformity import compute_conformity_from_directory, check_file_conformity
from frame2kg_eval.metrics.timing import manifest_timing
from frame2kg_eval.metrics.boxes import (box_iou_stats,aggregate_iou_micro,aggregate_iou_macro)
from frame2kg_eval.utils.logging import logger
from frame2kg_eval.utils.seeding import seed_matching, MATCHING_SEED


def _empty_prf(support: int, fp_penalty: int = 0) -> Dict[str, float]:
    """Build a zeroed precision/recall/F1 record with optional FP penalty."""

    tp = 0
    fp = fp_penalty
    fn = support

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
        "support": support,
    }


def _empty_box_stats() -> Dict[str, float]:
    """Return a zeroed-out box IoU stats record."""

    return {
        "mean_iou": 0.0,
        "median_iou": 0.0,
        "std_iou": 0.0,
        "min_iou": 0.0,
        "max_iou": 0.0,
        "count": 0,
        "match_ious": (),
    }


def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load configuration from file or use defaults."""
    default_config_path = Path(__file__).parent.parent / "config" / "defaults.yaml"
    
    if config_path and config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    elif default_config_path.exists():
        with open(default_config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Hardcoded defaults as fallback
        config = {
            "tau": 0.3,
            "alpha": 0.7,
            "text_mode": "semantic",
            "text_fields": ["id", "label"],
            "text_floor": 0.25,
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "predicate_mode": "exact"
        }
    
    return config


@click.command()
@click.option("--pred-dir", type=click.Path(exists=True, path_type=Path), required=True,
              help="Directory containing prediction files")
@click.option("--gt", type=str, required=True,
              help="Ground truth spec (hf:dataset:split or path)")
@click.option("--tau", type=float, default=None,
              help="IoU threshold for node matching")
@click.option("--alpha", type=float, default=None,
              help="Blending weight for IoU vs text similarity")
@click.option("--text-mode", type=click.Choice(["tfidf", "semantic", "hybrid"]), default=None,
              help="Text similarity mode")
@click.option("--text-fields", multiple=True, default=None,
              help="Node fields to use for text similarity")
@click.option("--text-floor", type=float, default=None,
              help="Minimum text similarity threshold")
@click.option("--out", type=click.Path(path_type=Path), required=True,
              help="Output file path (CSV or JSON)")
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=None,
              help="Configuration file path")
@click.option("--edge-baseline/--no-edge-baseline", default=False,
              help="Include edge-by-label baseline metrics")
@click.option("--verbose/--quiet", default=True,
              help="Verbose output")
def main(pred_dir, gt, tau, alpha, text_mode, text_fields, text_floor, out, config, 
         edge_baseline, verbose):
    """Evaluate Frame2KG predictions against ground truth."""
    
    # Load configuration
    cfg = load_config(config)
    
    # Override with CLI arguments
    if tau is not None:
        cfg["tau"] = tau
    if alpha is not None:
        cfg["alpha"] = alpha
    if text_mode is not None:
        cfg["text_mode"] = text_mode
    if text_fields:
        cfg["text_fields"] = list(text_fields)
    if text_floor is not None:
        cfg["text_floor"] = text_floor

    # Ensure deterministic behaviour across matching components
    seed_matching()

    if verbose:
        logger.info(f"Configuration: τ={cfg['tau']}, α={cfg['alpha']}, mode={cfg['text_mode']}")
        logger.info(f"Matching seed: {MATCHING_SEED}")
    
    # Load predictions and ground truth
    logger.info(f"Loading predictions from {pred_dir}")
    pred_loader = PredictionLoader(pred_dir)
    
    logger.info(f"Loading ground truth: {gt}")
    gt_adapter = create_ground_truth_adapter(gt)
    
    # Check validity statistics
    validity_stats = compute_validity_from_directory(pred_dir)
    logger.info(f"Validity: {validity_stats['valid_count']}/{validity_stats['total_count']} "
                f"({validity_stats['validity_rate']:.1f}%)")
    
    # Check schema conformity statistics
    conformity_stats = compute_conformity_from_directory(pred_dir)
    logger.info(f"Schema Conformity: {conformity_stats['conformant_count']}/{conformity_stats['total_count']} "
                f"({conformity_stats['conformity_rate_total']:.1f}%)")
    
    # Check for manifest timing
    manifest_path = pred_dir / "manifest.csv"
    timing_stats = manifest_timing(manifest_path) if manifest_path.exists() else None
    
    include_invalid = bool(cfg.get("include_invalid", True))
    strict_mode = bool(cfg.get("strict_mode", False))

    gt_graphs: Dict[Tuple[str, int], Dict] = {}
    for video_id, frame_no, graph in gt_adapter.iter_frames():
        gt_graphs[(video_id, frame_no)] = graph

    frame_results: List[Dict] = []
    all_node_metrics: List[Dict] = []
    all_edge_metrics: List[Dict] = []
    all_box_stats: List[Dict] = []

    pred_index = pred_loader.get_index()
    gt_keys = sorted(gt_graphs.keys())
    extra_predictions = sorted(set(pred_index.keys()) - set(gt_keys))
    total_frames = len(gt_keys)

    missing_predictions: List[Tuple[str, int]] = []
    unusable_predictions: List[Tuple[str, int]] = []

    if verbose:
        pbar = tqdm(total=total_frames, desc="Evaluating frames")

    for video_id, frame_no in gt_keys:
        gt_graph = gt_graphs[(video_id, frame_no)]
        g_nodes = gt_graph["nodes"]
        g_edges = gt_graph["edges"]

        pred_path = pred_index.get((video_id, frame_no))
        pred_graph = None
        parsed_json_flag = 0
        schema_conformant_flag = 0

        if pred_path is None:
            missing_predictions.append((video_id, frame_no))
        else:
            if pred_path.suffix == ".json":
                is_conformant, _ = check_file_conformity(pred_path)
                schema_conformant_flag = 1 if is_conformant else 0
                pred_graph = pred_loader.get_graph(video_id, frame_no)
                parsed_json_flag = 1 if pred_graph is not None else 0
                if pred_graph is None:
                    unusable_predictions.append((video_id, frame_no))
            else:
                unusable_predictions.append((video_id, frame_no))

        include_frame = True
        if pred_graph is None:
            include_frame = include_invalid or pred_path is None

        if not include_frame:
            if verbose:
                pbar.update(1)
            continue

        if pred_graph is None:
            p_nodes = []
            p_edges = []
            node_metrics = _empty_prf(len(g_nodes), len(g_nodes) if strict_mode else 0)
            edge_metrics = _empty_prf(len(g_edges), len(g_edges) if strict_mode else 0)
            box_stats = _empty_box_stats()
            edge_baseline_metrics = None
            if edge_baseline:
                edge_baseline_metrics = _empty_prf(len(g_edges), len(g_edges) if strict_mode else 0)
        else:
            p_nodes = pred_graph["nodes"]
            p_edges = pred_graph["edges"]

            match_result = two_stage_node_match(
                p_nodes, g_nodes,
                tau=cfg["tau"],
                alpha=cfg["alpha"],
                text_mode=cfg["text_mode"],
                text_fields=tuple(cfg["text_fields"]),
                text_floor=cfg["text_floor"]
            )

            node_metrics = node_prf1(p_nodes, g_nodes, match_result["mapping"])

            iou_matrix = match_result.get("matrices", {}).get("iou")
            box_stats = box_iou_stats(p_nodes, g_nodes, match_result["mapping"], iou_matrix=iou_matrix)

            node_id_mapping: Dict[str, str] = {}
            for p_idx, g_idx in match_result["mapping"].items():
                p_id = p_nodes[p_idx]["id"]
                g_id = g_nodes[g_idx]["id"]
                node_id_mapping[p_id] = g_id

            edge_metrics = edge_prf1(p_edges, g_edges, node_id_mapping, cfg.get("predicate_mode", "exact"))

            edge_baseline_metrics = None
            if edge_baseline:
                edge_baseline_metrics = edge_by_label_baseline(p_edges, g_edges, p_nodes, g_nodes)

        all_node_metrics.append(node_metrics)
        all_edge_metrics.append(edge_metrics)
        all_box_stats.append(box_stats)

        frame_result = {
            "video_id": video_id,
            "frame_no": frame_no,
            "parsed_json": parsed_json_flag,
            "schema_conformant": schema_conformant_flag,
            "node_tp": node_metrics["tp"],
            "node_fp": node_metrics["fp"],
            "node_fn": node_metrics["fn"],
            "node_precision": node_metrics["precision"],
            "node_recall": node_metrics["recall"],
            "node_f1": node_metrics["f1"],
            "edge_tp": edge_metrics["tp"],
            "edge_fp": edge_metrics["fp"],
            "edge_fn": edge_metrics["fn"],
            "edge_precision": edge_metrics["precision"],
            "edge_recall": edge_metrics["recall"],
            "edge_f1": edge_metrics["f1"],
            "box_mean_iou": box_stats["mean_iou"],
            "box_median_iou": box_stats["median_iou"],
            "box_std_iou": box_stats["std_iou"],
            "box_min_iou": box_stats["min_iou"],
            "box_max_iou": box_stats["max_iou"],
            "box_match_count": box_stats["count"],
        }

        if edge_baseline and edge_baseline_metrics:
            frame_result.update({
                "edge_baseline_precision": edge_baseline_metrics["precision"],
                "edge_baseline_recall": edge_baseline_metrics["recall"],
                "edge_baseline_f1": edge_baseline_metrics["f1"],
            })

        frame_results.append(frame_result)

        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()

    if extra_predictions:
        logger.warning(f"{len(extra_predictions)} prediction files have no matching ground truth and were ignored.")
    if missing_predictions:
        logger.warning(f"Missing predictions for {len(missing_predictions)} frames (treated as empty).")
    if unusable_predictions:
        logger.warning(f"Unusable prediction files for {len(unusable_predictions)} frames (treated as empty).")
    
    # Aggregate metrics
    node_micro = aggregate_micro(all_node_metrics)
    node_macro = aggregate_macro(all_node_metrics)
    edge_micro = aggregate_micro(all_edge_metrics)
    edge_macro = aggregate_macro(all_edge_metrics)
    box_micro = aggregate_iou_micro(all_box_stats)
    box_macro = aggregate_iou_macro(all_box_stats)
    
    # Write output
    output_path = Path(out)
    if output_path.suffix == ".json":
        # JSON output
        output_data = {
            "config": cfg,
            "validity": validity_stats,
            "conformity": conformity_stats,
            "timing": timing_stats,
            "aggregate": {
                "node_micro": node_micro,
                "node_macro": node_macro,
                "edge_micro": edge_micro,
                "edge_macro": edge_macro,
                "box_micro": box_micro,
                "box_macro": box_macro,
            },
            "frames": frame_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    else:
        # CSV output
        with open(output_path, 'w', newline='') as f:
            if frame_results:
                fieldnames = list(frame_results[0].keys())
                if timing_stats and timing_stats["n"] > 0:
                    fieldnames.append("gen_wall_time_s")
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Get per-frame timing if available
                frame_times = {}
                if manifest_path.exists():
                    from frame2kg_eval.metrics.timing import extract_timing_per_frame
                    frame_times = extract_timing_per_frame(manifest_path)
                
                # Write frame results
                for result in frame_results:
                    row = result.copy()
                    key = (result["video_id"], result["frame_no"])
                    if key in frame_times:
                        row["gen_wall_time_s"] = frame_times[key]
                    writer.writerow(row)
                
                # Write summary footer
                writer.writerow({})  # Empty row
                writer.writerow({
                    "video_id": "SUMMARY",
                    "frame_no": "MICRO",
                    "node_precision": node_micro["precision"],
                    "node_recall": node_micro["recall"],
                    "node_f1": node_micro["f1"],
                    "edge_precision": edge_micro["precision"],
                    "edge_recall": edge_micro["recall"],
                    "edge_f1": edge_micro["f1"],
                    "box_mean_iou": box_micro["mean_iou"],
                    "box_median_iou": box_micro["median_iou"],
                })
                writer.writerow({
                    "video_id": "SUMMARY",
                    "frame_no": "MACRO",
                    "node_precision": node_macro["precision"],
                    "node_recall": node_macro["recall"],
                    "node_f1": node_macro["f1"],
                    "edge_precision": edge_macro["precision"],
                    "edge_recall": edge_macro["recall"],
                    "edge_f1": edge_macro["f1"],
                    "box_mean_iou": box_macro["mean_iou"],
                    "box_median_iou": box_macro["median_iou"],
                })
    
    # Print summary
    logger.success(f"Evaluation complete! Results written to {output_path}")
    logger.info(f"Node F1: micro={node_micro['f1']:.3f}, macro={node_macro['f1']:.3f}")
    logger.info(f"Edge F1: micro={edge_micro['f1']:.3f}, macro={edge_macro['f1']:.3f}")
    logger.info(
        "Matched-pair IoU (box IoU): micro={micro:.3f} (weighted by matched pairs), "
        "macro={macro:.3f} (unweighted mean of per-frame means)"
        .format(micro=box_micro["mean_iou"], macro=box_macro["mean_iou"])
    )
    
    if timing_stats and timing_stats["n"] > 0:
        logger.info(f"Mean generation time: {timing_stats['mean']:.2f}s")


if __name__ == "__main__":
    main()
