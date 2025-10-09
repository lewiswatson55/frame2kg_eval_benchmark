"""Main evaluation CLI command."""

import click
import csv
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from frame2kg_eval.io.preds import PredictionLoader
from frame2kg_eval.io.groundtruth import create_ground_truth_adapter
from frame2kg_eval.matching.assign import two_stage_node_match, compute_edge_mapping
from frame2kg_eval.metrics.nodes import node_prf1, aggregate_micro, aggregate_macro
from frame2kg_eval.metrics.edges import edge_prf1, edge_by_label_baseline
from frame2kg_eval.metrics.validity import compute_validity_from_directory
from frame2kg_eval.metrics.conformity import compute_conformity_from_directory
from frame2kg_eval.metrics.timing import manifest_timing
from frame2kg_eval.metrics.boxes import (box_iou_stats,aggregate_iou_micro,aggregate_iou_macro)
from frame2kg_eval.utils.logging import logger


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
    
    if verbose:
        logger.info(f"Configuration: τ={cfg['tau']}, α={cfg['alpha']}, mode={cfg['text_mode']}")
    
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
    
    # Prepare output data
    frame_results = []
    all_node_metrics = []
    all_edge_metrics = []
    all_box_stats = []
    
    # Process each frame
    pred_index = pred_loader.get_index()
    total_frames = len(pred_index)
    
    if verbose:
        pbar = tqdm(total=total_frames, desc="Evaluating frames")
    
    for (video_id, frame_no), pred_path in sorted(pred_index.items()):
        # Get predictions
        pred_graph = pred_loader.get_graph(video_id, frame_no)
        
        # Get ground truth
        gt_graph = gt_adapter.get_graph(video_id, frame_no)
        
        if pred_graph is None or gt_graph is None:
            # Skip frames with missing data
            if verbose:
                pbar.update(1)
            continue
        
        # Extract nodes and edges
        p_nodes = pred_graph["nodes"]
        g_nodes = gt_graph["nodes"]
        p_edges = pred_graph["edges"]
        g_edges = gt_graph["edges"]
        
        # Node matching
        match_result = two_stage_node_match(
            p_nodes, g_nodes,
            tau=cfg["tau"],
            alpha=cfg["alpha"],
            text_mode=cfg["text_mode"],
            text_fields=tuple(cfg["text_fields"]),
            text_floor=cfg["text_floor"]
        )
        
        # Node metrics
        node_metrics = node_prf1(p_nodes, g_nodes, match_result["mapping"])
        all_node_metrics.append(node_metrics)

        # Box IoU closeness stats using precomputed IoU matrix
        iou_matrix = match_result.get("matrices", {}).get("iou")
        box_stats = box_iou_stats(p_nodes, g_nodes, match_result["mapping"], iou_matrix=iou_matrix)
        all_box_stats.append(box_stats)
        
        # Build node ID mapping for edges
        node_id_mapping = {}
        for p_idx, g_idx in match_result["mapping"].items():
            p_id = p_nodes[p_idx]["id"]
            g_id = g_nodes[g_idx]["id"]
            node_id_mapping[p_id] = g_id
        
        # Edge metrics
        edge_metrics = edge_prf1(p_edges, g_edges, node_id_mapping, cfg.get("predicate_mode", "exact"))
        all_edge_metrics.append(edge_metrics)
        
        # Optional edge baseline
        edge_baseline_metrics = None
        if edge_baseline:
            edge_baseline_metrics = edge_by_label_baseline(p_edges, g_edges, p_nodes, g_nodes)
        
        # Check schema conformity for this frame
        from frame2kg_eval.metrics.conformity import check_file_conformity
        is_conformant, _ = check_file_conformity(pred_path) if pred_path.suffix == ".json" else (False, None)
        
        # Store frame result
        frame_result = {
            "video_id": video_id,
            "frame_no": frame_no,
            "parsed_json": 1 if pred_path.suffix == ".json" else 0,
            "schema_conformant": 1 if is_conformant else 0,
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
            # Box closeness stats
            "box_mean_iou": box_stats["mean_iou"],
            "box_std_iou": box_stats["std_iou"],
            "box_min_iou": box_stats["min_iou"],
            "box_max_iou": box_stats["max_iou"],
            "box_match_count": box_stats["count"],
        }
        
        if edge_baseline_metrics:
            frame_result.update({
                "edge_baseline_precision": edge_baseline_metrics["precision"],
                "edge_baseline_recall": edge_baseline_metrics["recall"],
                "edge_baseline_f1": edge_baseline_metrics["f1"]
            })
        
        frame_results.append(frame_result)
        
        if verbose:
            pbar.update(1)
    
    if verbose:
        pbar.close()
    
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
                })
    
    # Print summary
    logger.success(f"Evaluation complete! Results written to {output_path}")
    logger.info(f"Node F1: micro={node_micro['f1']:.3f}, macro={node_macro['f1']:.3f}")
    logger.info(f"Edge F1: micro={edge_micro['f1']:.3f}, macro={edge_macro['f1']:.3f}")
    
    if timing_stats and timing_stats["n"] > 0:
        logger.info(f"Mean generation time: {timing_stats['mean']:.2f}s")


if __name__ == "__main__":
    main()
