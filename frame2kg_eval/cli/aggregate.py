"""Aggregate evaluation across multiple runs."""

import click
import csv
import json
import yaml
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from frame2kg_eval.io.preds import PredictionLoader
from frame2kg_eval.io.groundtruth import create_ground_truth_adapter
from frame2kg_eval.matching.assign import two_stage_node_match
from frame2kg_eval.metrics.nodes import node_prf1, aggregate_micro
from frame2kg_eval.metrics.edges import edge_prf1
from frame2kg_eval.metrics.validity import compute_validity_from_directory
from frame2kg_eval.metrics.timing import manifest_timing
from frame2kg_eval.utils.logging import logger


def evaluate_single_run(pred_dir: Path, gt_adapter, config: Dict) -> Dict:
    """Evaluate a single prediction run."""
    pred_loader = PredictionLoader(pred_dir)
    
    # Validity statistics
    validity_stats = compute_validity_from_directory(pred_dir)
    
    # Timing statistics
    manifest_path = pred_dir / "manifest.csv"
    timing_stats = manifest_timing(manifest_path) if manifest_path.exists() else None
    
    # Evaluate frames
    node_metrics_list = []
    edge_metrics_list = []
    
    for (video_id, frame_no), _ in pred_loader.get_index().items():
        pred_graph = pred_loader.get_graph(video_id, frame_no)
        gt_graph = gt_adapter.get_graph(video_id, frame_no)
        
        if not pred_graph or not gt_graph:
            continue
        
        # Node matching
        match_result = two_stage_node_match(
            pred_graph["nodes"], gt_graph["nodes"],
            tau=config["tau"],
            alpha=config["alpha"],
            text_mode=config["text_mode"],
            text_fields=tuple(config["text_fields"]),
            text_floor=config["text_floor"]
        )
        
        # Metrics
        node_metrics = node_prf1(
            pred_graph["nodes"], gt_graph["nodes"],
            match_result["mapping"]
        )
        node_metrics_list.append(node_metrics)
        
        # Edge metrics
        node_id_mapping = {
            pred_graph["nodes"][p_idx]["id"]: gt_graph["nodes"][g_idx]["id"]
            for p_idx, g_idx in match_result["mapping"].items()
        }
        
        edge_metrics = edge_prf1(
            pred_graph["edges"], gt_graph["edges"],
            node_id_mapping, config.get("predicate_mode", "exact")
        )
        edge_metrics_list.append(edge_metrics)
    
    # Aggregate
    node_micro = aggregate_micro(node_metrics_list)
    edge_micro = aggregate_micro(edge_metrics_list)
    
    return {
        "pred_dir": str(pred_dir),
        "validity_rate": validity_stats["validity_rate"],
        "valid_count": validity_stats["valid_count"],
        "invalid_count": validity_stats["invalid_count"],
        "node_f1": node_micro["f1"],
        "node_precision": node_micro["precision"],
        "node_recall": node_micro["recall"],
        "edge_f1": edge_micro["f1"],
        "edge_precision": edge_micro["precision"],
        "edge_recall": edge_micro["recall"],
        "mean_gen_time": timing_stats["mean"] if timing_stats else None,
        "num_frames": len(node_metrics_list)
    }


@click.command()
@click.option("--pred-root", type=click.Path(exists=True, path_type=Path), required=True,
              help="Root directory containing variant/index subdirectories")
@click.option("--gt", type=str, required=True,
              help="Ground truth spec (hf:dataset:split or path)")
@click.option("--tau", type=float, default=0.3,
              help="IoU threshold for node matching")
@click.option("--alpha", type=float, default=0.7,
              help="Blending weight for IoU vs text similarity")
@click.option("--text-mode", type=click.Choice(["tfidf", "semantic", "hybrid"]), default="semantic",
              help="Text similarity mode")
@click.option("--text-floor", type=float, default=0.25,
              help="Minimum text similarity threshold")
@click.option("--out", type=click.Path(path_type=Path), required=True,
              help="Output CSV file path")
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=None,
              help="Configuration file path")
@click.option("--pattern", type=str, default="*/*",
              help="Directory pattern for finding runs (e.g., '*/*' for variant/index)")
@click.option("--verbose/--quiet", default=True,
              help="Verbose output")
def main(pred_root, gt, tau, alpha, text_mode, text_floor, out, config, pattern, verbose):
    """Aggregate evaluation across multiple prediction runs."""
    
    # Load configuration
    cfg = {}
    if config and Path(config).exists():
        with open(config) as f:
            cfg = yaml.safe_load(f)
    
    # Override with CLI arguments
    cfg.update({
        "tau": tau,
        "alpha": alpha,
        "text_mode": text_mode,
        "text_floor": text_floor,
        "text_fields": cfg.get("text_fields", ["id", "label"])
    })
    
    logger.info(f"Configuration: τ={tau}, α={alpha}, mode={text_mode}")
    
    # Find all run directories
    pred_root = Path(pred_root)
    run_dirs = []
    
    for path in pred_root.glob(pattern):
        if path.is_dir():
            # Check if it contains prediction files
            json_files = list(path.glob("*.json"))
            if json_files:
                run_dirs.append(path)
    
    logger.info(f"Found {len(run_dirs)} run directories to evaluate")
    
    if not run_dirs:
        logger.error("No valid run directories found!")
        return
    
    # Load ground truth
    logger.info(f"Loading ground truth: {gt}")
    gt_adapter = create_ground_truth_adapter(gt)
    
    # Evaluate each run
    results = []
    
    for run_dir in tqdm(run_dirs, desc="Evaluating runs"):
        try:
            run_result = evaluate_single_run(run_dir, gt_adapter, cfg)
            
            # Parse variant and index from path
            path_parts = run_dir.relative_to(pred_root).parts
            if len(path_parts) >= 2:
                run_result["variant"] = path_parts[0]
                run_result["index"] = path_parts[1]
            elif len(path_parts) == 1:
                run_result["variant"] = path_parts[0]
                run_result["index"] = "0"
            else:
                run_result["variant"] = run_dir.name
                run_result["index"] = "0"
            
            results.append(run_result)
            
        except Exception as e:
            logger.warning(f"Failed to evaluate {run_dir}: {e}")
            continue
    
    # Sort by node F1
    results.sort(key=lambda x: x["node_f1"], reverse=True)
    
    # Write results
    output_path = Path(out)
    with open(output_path, 'w', newline='') as f:
        if results:
            fieldnames = ["variant", "index", "pred_dir", "validity_rate", 
                         "valid_count", "invalid_count",
                         "node_f1", "node_precision", "node_recall",
                         "edge_f1", "edge_precision", "edge_recall",
                         "mean_gen_time", "num_frames"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    # Print summary
    logger.success(f"Aggregation complete! Results written to {output_path}")
    
    if results:
        # Overall statistics
        avg_node_f1 = sum(r["node_f1"] for r in results) / len(results)
        avg_edge_f1 = sum(r["edge_f1"] for r in results) / len(results)
        avg_validity = sum(r["validity_rate"] for r in results) / len(results)
        
        logger.info(f"\nOverall statistics across {len(results)} runs:")
        logger.info(f"  Average Node F1: {avg_node_f1:.3f}")
        logger.info(f"  Average Edge F1: {avg_edge_f1:.3f}")
        logger.info(f"  Average Validity: {avg_validity:.1f}%")
        
        # Best run
        best = results[0]
        logger.info(f"\nBest run: {best['variant']}/{best['index']}")
        logger.info(f"  Node F1: {best['node_f1']:.3f}")
        logger.info(f"  Edge F1: {best['edge_f1']:.3f}")
        logger.info(f"  Validity: {best['validity_rate']:.1f}%")


if __name__ == "__main__":
    main()
