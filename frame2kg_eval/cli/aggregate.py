"""Aggregate evaluation across multiple runs."""

import click
import csv
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from frame2kg_eval.io.preds import PredictionLoader
from frame2kg_eval.io.groundtruth import create_ground_truth_adapter
from frame2kg_eval.matching.assign import two_stage_node_match
from frame2kg_eval.matching.text import TextSimilarityComputer, clear_text_computer_caches
from frame2kg_eval.metrics.nodes import node_prf1, aggregate_micro
from frame2kg_eval.metrics.edges import edge_prf1
from frame2kg_eval.metrics.validity import compute_validity_from_directory
from frame2kg_eval.metrics.conformity import compute_conformity_from_directory
from frame2kg_eval.metrics.timing import manifest_timing
from frame2kg_eval.metrics.boxes import (
    box_iou_stats,
    aggregate_iou_micro,
    IOU_COVERAGE_THRESHOLDS,
)
from frame2kg_eval.utils.logging import logger
from frame2kg_eval.utils.seeding import seed_matching, MATCHING_SEED


def _empty_prf(support: int, fp_penalty: int = 0) -> Dict[str, float]:
    """Return a zeroed metric record with optional FP penalty."""

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
    """Return zeroed matched-pair IoU stats."""

    stats = {
        "mean_iou": 0.0,
        "median_iou": 0.0,
        "std_iou": 0.0,
        "min_iou": 0.0,
        "max_iou": 0.0,
        "count": 0,
        "match_ious": (),
    }
    for _, key in IOU_COVERAGE_THRESHOLDS:
        stats[key] = 0.0
    return stats


def evaluate_single_run(pred_dir: Path, gt_graphs: Dict[Tuple[str, int], Dict], config: Dict) -> Dict:
    """Evaluate a single prediction run."""

    seed_matching()

    pred_loader = PredictionLoader(pred_dir)

    validity_stats = compute_validity_from_directory(pred_dir)
    conformity_stats = compute_conformity_from_directory(pred_dir)

    manifest_path = pred_dir / "manifest.csv"
    timing_stats = manifest_timing(manifest_path) if manifest_path.exists() else None

    include_invalid = bool(config.get("include_invalid", True))
    strict_mode = bool(config.get("strict_mode", False))
    text_mode = config.get("text_mode", "semantic")

    shared_text_computer = TextSimilarityComputer(
        mode=text_mode,
        model_name=config.get("model_name"),
    )
    semantic_text_computer = shared_text_computer if text_mode == "semantic" else None
    if config.get("predicate_mode") == "semantic" and semantic_text_computer is None:
        semantic_text_computer = TextSimilarityComputer(
            mode="semantic",
            model_name=config.get("model_name"),
        )

    node_metrics_list: List[Dict] = []
    edge_metrics_list: List[Dict] = []
    box_stats_list: List[Dict] = []

    pred_index = pred_loader.get_index()
    gt_keys = sorted(gt_graphs.keys())

    missing_predictions: List[Tuple[str, int]] = []
    unusable_predictions: List[Tuple[str, int]] = []
    extra_predictions = sorted(set(pred_index.keys()) - set(gt_keys))

    for video_id, frame_no in gt_keys:
        gt_graph = gt_graphs[(video_id, frame_no)]
        g_nodes = gt_graph["nodes"]
        g_edges = gt_graph["edges"]

        pred_path = pred_index.get((video_id, frame_no))
        pred_graph = None

        if pred_path is None:
            missing_predictions.append((video_id, frame_no))
        else:
            if pred_path.suffix == ".json":
                pred_graph = pred_loader.get_graph(video_id, frame_no)
                if pred_graph is None:
                    unusable_predictions.append((video_id, frame_no))
            else:
                unusable_predictions.append((video_id, frame_no))

        include_frame = True
        if pred_graph is None:
            include_frame = include_invalid or pred_path is None

        if not include_frame:
            continue

        if pred_graph is None:
            node_metrics = _empty_prf(len(g_nodes), len(g_nodes) if strict_mode else 0)
            edge_metrics = _empty_prf(len(g_edges), len(g_edges) if strict_mode else 0)
            box_stats = _empty_box_stats()
        else:
            p_nodes = pred_graph["nodes"]
            p_edges = pred_graph["edges"]

            match_result = two_stage_node_match(
                p_nodes, g_nodes,
                tau=config["tau"],
                alpha=config["alpha"],
                text_mode=text_mode,
                text_fields=tuple(config["text_fields"]),
                text_floor=config["text_floor"],
                text_computer=shared_text_computer,
            )

            node_metrics = node_prf1(p_nodes, g_nodes, match_result["mapping"])

            iou_matrix = match_result.get("matrices", {}).get("iou")
            box_stats = box_iou_stats(p_nodes, g_nodes, match_result["mapping"], iou_matrix=iou_matrix)

            node_id_mapping = {
                p_nodes[p_idx]["id"]: g_nodes[g_idx]["id"]
                for p_idx, g_idx in match_result["mapping"].items()
            }

            edge_metrics = edge_prf1(
                p_edges,
                g_edges,
                node_id_mapping,
                config.get("predicate_mode", "exact"),
                semantic_threshold=config.get("predicate_semantic_threshold", 0.6),
                model_name=config.get("model_name"),
                text_computer=semantic_text_computer,
            )

        node_metrics_list.append(node_metrics)
        edge_metrics_list.append(edge_metrics)
        box_stats_list.append(box_stats)

        clear_text_computer_caches(shared_text_computer, semantic_text_computer)

    if extra_predictions:
        logger.warning(f"[{pred_dir}] {len(extra_predictions)} prediction files have no matching ground truth and were ignored.")
    if missing_predictions:
        logger.warning(f"[{pred_dir}] Missing predictions for {len(missing_predictions)} frames (treated as empty).")
    if unusable_predictions:
        logger.warning(f"[{pred_dir}] Unusable prediction files for {len(unusable_predictions)} frames (treated as empty).")

    node_micro = aggregate_micro(node_metrics_list)
    edge_micro = aggregate_micro(edge_metrics_list)
    box_micro = aggregate_iou_micro(box_stats_list)

    return {
        "pred_dir": str(pred_dir),
        "validity_rate": validity_stats["validity_rate"],
        "valid_count": validity_stats["valid_count"],
        "invalid_count": validity_stats["invalid_count"],
        "conformity_rate": conformity_stats["conformity_rate_total"],
        "conformant_count": conformity_stats["conformant_count"],
        "non_conformant_count": conformity_stats["non_conformant_count"],
        "node_f1": node_micro["f1"],
        "node_precision": node_micro["precision"],
        "node_recall": node_micro["recall"],
        "edge_f1": edge_micro["f1"],
        "edge_precision": edge_micro["precision"],
        "edge_recall": edge_micro["recall"],
        "box_mean_iou": box_micro["mean_iou"],
        "box_median_iou": box_micro["median_iou"],
        **{key: box_micro.get(key, 0.0) for _, key in IOU_COVERAGE_THRESHOLDS},
        "mean_gen_time": timing_stats["mean"] if timing_stats else None,
        "num_frames": len(node_metrics_list),
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
    logger.info(f"Matching seed: {MATCHING_SEED}")
    
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
    gt_graphs: Dict[Tuple[str, int], Dict] = {}
    for video_id, frame_no, graph in gt_adapter.iter_frames():
        gt_graphs[(video_id, frame_no)] = graph
    
    # Evaluate each run
    results = []
    
    for run_dir in tqdm(run_dirs, desc="Evaluating runs"):
        try:
            run_result = evaluate_single_run(run_dir, gt_graphs, cfg)
            
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
            fieldnames = [
                "variant", "index", "pred_dir", "validity_rate", 
                "valid_count", "invalid_count",
                "node_f1", "node_precision", "node_recall",
                "edge_f1", "edge_precision", "edge_recall",
                "box_mean_iou", "box_median_iou",
                *[key for _, key in IOU_COVERAGE_THRESHOLDS],
                "mean_gen_time", "num_frames"
            ]
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
        avg_box_iou = sum(r.get("box_mean_iou", 0.0) for r in results) / len(results)
        avg_box_median = sum(r.get("box_median_iou", 0.0) for r in results) / len(results)
        avg_coverages = {
            key: sum(r.get(key, 0.0) for r in results) / len(results)
            for _, key in IOU_COVERAGE_THRESHOLDS
        }

        logger.info(f"\nOverall statistics across {len(results)} runs:")
        logger.info(f"  Average Node F1: {avg_node_f1:.3f}")
        logger.info(f"  Average Edge F1: {avg_edge_f1:.3f}")
        logger.info(f"  Average Validity: {avg_validity:.1f}%")
        logger.info(f"  Matched-pair IoU (box IoU) micro mean: {avg_box_iou:.3f}")
        logger.info(f"  Matched-pair IoU (box IoU) micro median: {avg_box_median:.3f}")
        for threshold, key in IOU_COVERAGE_THRESHOLDS:
            label = f"IoU@{threshold:.2f} coverage"
            logger.info(
                f"  {label}: {avg_coverages[key] * 100:.1f}%"
            )

        # Best run
        best = results[0]
        logger.info(f"\nBest run: {best['variant']}/{best['index']}")
        logger.info(f"  Node F1: {best['node_f1']:.3f}")
        logger.info(f"  Edge F1: {best['edge_f1']:.3f}")
        logger.info(f"  Validity: {best['validity_rate']:.1f}%")
        logger.info(
            f"  Matched-pair IoU (box IoU) micro mean: {best.get('box_mean_iou', 0.0):.3f}"
        )
        logger.info(
            f"  Matched-pair IoU (box IoU) micro median: {best.get('box_median_iou', 0.0):.3f}"
        )
        for threshold, key in IOU_COVERAGE_THRESHOLDS:
            label = f"IoU@{threshold:.2f} coverage"
            logger.info(
                f"  {label}: {best.get(key, 0.0) * 100:.1f}%"
            )


if __name__ == "__main__":
    main()
