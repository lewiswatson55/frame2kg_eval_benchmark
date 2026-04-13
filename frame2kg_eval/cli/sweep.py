"""Threshold sweep CLI command."""

import click
import csv
from pathlib import Path
from itertools import product
from typing import Dict
from tqdm import tqdm

from frame2kg_eval.cli.evaluate import load_config
from frame2kg_eval.io.preds import PredictionLoader
from frame2kg_eval.io.groundtruth import create_ground_truth_adapter
from frame2kg_eval.matching.assign import two_stage_node_match
from frame2kg_eval.matching.text import TextSimilarityComputer, clear_text_computer_caches
from frame2kg_eval.metrics.nodes import node_prf1, aggregate_micro
from frame2kg_eval.metrics.edges import edge_prf1
from frame2kg_eval.utils.logging import logger
from frame2kg_eval.utils.seeding import MATCHING_SEED, seed_matching


def _empty_metrics(support: int, strict_mode: bool) -> Dict[str, float]:
    """Mirror eval CLI behaviour for missing predictions."""
    fp_penalty = support if strict_mode else 0
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


@click.command()
@click.option("--pred-dir", type=click.Path(exists=True, path_type=Path), required=True,
              help="Directory containing prediction files")
@click.option("--gt", type=str, required=True,
              help="Ground truth spec (hf:dataset:split or path)")
@click.option("--taus", multiple=True, type=float, default=(),
              help="IoU thresholds to sweep (use multiple times: --taus 0.3 --taus 0.5)")
@click.option("--alphas", multiple=True, type=float, default=(),
              help="Alpha blending weights to sweep (use multiple times: --alphas 0.5 --alphas 0.7)")
@click.option("--text-mode", type=click.Choice(["tfidf", "semantic", "hybrid"]), default=None,
              help="Text similarity mode")
@click.option("--text-floor", type=float, default=None,
              help="Minimum text similarity threshold")
@click.option("--out", type=click.Path(path_type=Path), required=True,
              help="Output CSV file path")
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=None,
              help="Configuration file path")
@click.option("--verbose/--quiet", default=True,
              help="Verbose output")
def main(pred_dir, gt, taus, alphas, text_mode, text_floor, out, config,
         verbose):
    """Sweep τ and α parameters to find optimal thresholds."""
    cfg = load_config(config)

    tau_values = list(taus) if taus else cfg.get("default_taus") or [cfg.get("tau", 0.3)]
    alpha_values = list(alphas) if alphas else cfg.get("default_alphas") or [cfg.get("alpha", 0.7)]
    cfg_text_mode = text_mode if text_mode is not None else cfg.get("text_mode", "semantic")
    cfg_text_floor = text_floor if text_floor is not None else cfg.get("text_floor", 0.25)

    raw_text_fields = cfg.get("text_fields")
    if not raw_text_fields:
        raw_text_fields = ["id", "label"]
    text_fields = tuple(raw_text_fields)

    predicate_mode = cfg.get("predicate_mode", "normalised")
    predicate_semantic_threshold = cfg.get("predicate_semantic_threshold", 0.6)
    predicate_model_name = cfg.get("model_name")
    shared_text_computer = TextSimilarityComputer(
        mode=cfg_text_mode,
        model_name=predicate_model_name,
    )
    semantic_text_computer = shared_text_computer if cfg_text_mode == "semantic" else None
    if predicate_mode == "semantic" and semantic_text_computer is None:
        semantic_text_computer = TextSimilarityComputer(
            mode="semantic",
            model_name=predicate_model_name,
        )

    seed_matching()

    include_invalid = bool(cfg.get("include_invalid", True))
    strict_mode = bool(cfg.get("strict_mode", False))

    logger.info(f"Sweeping τ={tau_values}, α={alpha_values}")
    logger.info(f"Text mode: {cfg_text_mode}, floor: {cfg_text_floor}")
    logger.info(f"Matching seed: {MATCHING_SEED}")

    # Load data
    logger.info(f"Loading predictions from {pred_dir}")
    pred_loader = PredictionLoader(pred_dir)

    logger.info(f"Loading ground truth: {gt}")
    gt_adapter = create_ground_truth_adapter(gt)

    # Collect all frames to evaluate
    frames_to_eval = []
    gt_graphs = {}
    for video_id, frame_no, graph in gt_adapter.iter_frames():
        gt_graphs[(video_id, frame_no)] = graph

    for (video_id, frame_no), gt_graph in sorted(gt_graphs.items()):
        pred_graph = pred_loader.get_graph(video_id, frame_no)

        if pred_graph is None:
            if not include_invalid:
                continue
            frames_to_eval.append({
                "video_id": video_id,
                "frame_no": frame_no,
                "p_nodes": [],
                "g_nodes": gt_graph["nodes"],
                "p_edges": [],
                "g_edges": gt_graph["edges"],
                "missing_pred": True,
            })
            continue

        frames_to_eval.append({
            "video_id": video_id,
            "frame_no": frame_no,
            "p_nodes": pred_graph["nodes"],
            "g_nodes": gt_graph["nodes"],
            "p_edges": pred_graph["edges"],
            "g_edges": gt_graph["edges"],
            "missing_pred": False,
        })

    logger.info(f"Prepared {len(frames_to_eval)} frames for evaluation")
    
    # Run sweep
    results = []
    param_combinations = list(product(tau_values, alpha_values))
    
    for tau, alpha in tqdm(param_combinations, desc="Sweeping parameters"):
        # Evaluate all frames with these parameters
        node_metrics_list = []
        edge_metrics_list = []
        
        for frame in frames_to_eval:
            if frame.get("missing_pred"):
                node_metrics_list.append(_empty_metrics(len(frame["g_nodes"]), strict_mode))
                edge_metrics_list.append(_empty_metrics(len(frame["g_edges"]), strict_mode))
                continue

            match_result = two_stage_node_match(
                frame["p_nodes"], frame["g_nodes"],
                tau=tau,
                alpha=alpha,
                text_mode=cfg_text_mode,
                text_fields=text_fields,
                text_floor=cfg_text_floor,
                text_computer=shared_text_computer,
            )

            node_metrics = node_prf1(
                frame["p_nodes"], frame["g_nodes"],
                match_result["mapping"]
            )
            node_metrics_list.append(node_metrics)

            node_id_mapping = {}
            for p_idx, g_idx in match_result["mapping"].items():
                p_id = frame["p_nodes"][p_idx]["id"]
                g_id = frame["g_nodes"][g_idx]["id"]
                node_id_mapping[p_id] = g_id

            edge_metrics = edge_prf1(
                frame["p_edges"], frame["g_edges"],
                node_id_mapping,
                predicate_mode,
                semantic_threshold=predicate_semantic_threshold,
                model_name=predicate_model_name,
                text_computer=semantic_text_computer,
            )
            edge_metrics_list.append(edge_metrics)
        
        # Aggregate metrics
        node_micro = aggregate_micro(node_metrics_list)
        edge_micro = aggregate_micro(edge_metrics_list)
        
        # Store result
        results.append({
            "tau": tau,
            "alpha": alpha,
            "text_mode": cfg_text_mode,
            "text_floor": cfg_text_floor,
            "node_f1_micro": node_micro["f1"],
            "node_precision_micro": node_micro["precision"],
            "node_recall_micro": node_micro["recall"],
            "edge_f1_micro": edge_micro["f1"],
            "edge_precision_micro": edge_micro["precision"],
            "edge_recall_micro": edge_micro["recall"],
            "combined_f1": (node_micro["f1"] + edge_micro["f1"]) / 2,
            "num_frames": len(frames_to_eval)
        })

        clear_text_computer_caches(shared_text_computer, semantic_text_computer)
    
    # Sort by combined F1
    results.sort(key=lambda x: x["combined_f1"], reverse=True)
    
    # Write results
    output_path = Path(out)
    with open(output_path, 'w', newline='') as f:
        if results:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    # Print top results
    logger.success(f"Sweep complete! Results written to {output_path}")
    logger.info("\nTop 5 parameter combinations:")
    for i, result in enumerate(results[:5], 1):
        logger.info(f"{i}. τ={result['tau']:.2f}, α={result['alpha']:.2f}: "
                   f"Node F1={result['node_f1_micro']:.3f}, "
                   f"Edge F1={result['edge_f1_micro']:.3f}, "
                   f"Combined={result['combined_f1']:.3f}")
    
    # Best parameters
    best = results[0]
    logger.success(f"\nBest parameters: τ={best['tau']}, α={best['alpha']}")
    logger.success(f"Best combined F1: {best['combined_f1']:.3f}")


if __name__ == "__main__":
    main()
