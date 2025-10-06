"""Example usage of the Frame2KG evaluation toolkit."""

from pathlib import Path
from frame2kg_eval.io.preds import PredictionLoader
from frame2kg_eval.io.groundtruth import HFDatasetAdapter
from frame2kg_eval.matching.assign import two_stage_node_match
from frame2kg_eval.metrics.nodes import node_prf1, aggregate_micro
from frame2kg_eval.metrics.edges import edge_prf1
from frame2kg_eval.utils.logging import logger


def main():
    # Example 1: Load and validate predictions
    logger.info("Example 1: Loading predictions")
    
    # Replace with your prediction directory
    pred_dir = Path("./predictions/model_v1/run1")
    if pred_dir.exists():
        pred_loader = PredictionLoader(pred_dir)
        valid, invalid, total = pred_loader.count_valid()
        logger.info(f"Found {valid}/{total} valid predictions")
    
    # Example 2: Load ground truth from HuggingFace
    logger.info("\nExample 2: Loading ground truth")
    gt_adapter = HFDatasetAdapter("lewiswatson/Frame2KG-YC2", "validation_dev")
    logger.info(f"Loaded {gt_adapter.count_frames()} ground truth frames")
    
    # Example 3: Evaluate a single frame
    logger.info("\nExample 3: Single frame evaluation")
    
    # Get first available frame
    for video_id, frame_no, gt_graph in gt_adapter.iter_frames():
        logger.info(f"Evaluating {video_id} frame {frame_no}")
        
        # Mock prediction for demo (in practice, load from file)
        pred_graph = {
            "nodes": [
                {"id": "person1", "label": "person", "location": "0.1,0.2,0.3,0.4"},
                {"id": "ball1", "label": "ball", "location": "0.5,0.6,0.7,0.8"}
            ],
            "edges": [
                {"source": "person1", "target": "ball1", "predicate": "holding"}
            ]
        }
        
        # Perform node matching
        match_result = two_stage_node_match(
            pred_graph["nodes"],
            gt_graph["nodes"],
            tau=0.3,
            alpha=0.7,
            text_mode="tfidf",
            text_fields=("id", "label"),
            text_floor=0.25
        )
        
        # Calculate node metrics
        node_metrics = node_prf1(
            pred_graph["nodes"],
            gt_graph["nodes"],
            match_result["mapping"]
        )
        
        logger.info(f"Node metrics: P={node_metrics['precision']:.3f}, "
                   f"R={node_metrics['recall']:.3f}, F1={node_metrics['f1']:.3f}")
        
        # Build node ID mapping for edges
        node_id_mapping = {}
        for p_idx, g_idx in match_result["mapping"].items():
            p_id = pred_graph["nodes"][p_idx]["id"]
            g_id = gt_graph["nodes"][g_idx]["id"]
            node_id_mapping[p_id] = g_id
        
        # Calculate edge metrics
        edge_metrics = edge_prf1(
            pred_graph["edges"],
            gt_graph["edges"],
            node_id_mapping,
            "exact"
        )
        
        logger.info(f"Edge metrics: P={edge_metrics['precision']:.3f}, "
                   f"R={edge_metrics['recall']:.3f}, F1={edge_metrics['f1']:.3f}")
        
        break  # Just evaluate one frame for demo
    
    # Example 4: Batch evaluation with CLI
    logger.info("\nExample 4: CLI usage examples")
    logger.info("Run evaluation:")
    logger.info("  frame2kg-eval --pred-dir ./preds --gt hf:lewiswatson/Frame2KG-YC2:validation_dev --out results.csv")
    logger.info("\nParameter sweep:")
    logger.info("  frame2kg-sweep --pred-dir ./preds --gt hf:lewiswatson/Frame2KG-YC2:validation_dev --out sweep.csv")
    logger.info("\nAggregate multiple runs:")
    logger.info("  frame2kg-aggregate --pred-root ./all_preds --gt hf:lewiswatson/Frame2KG-YC2:validation_dev --out aggregate.csv")
    logger.info("\nSanity check:")
    logger.info("  frame2kg-doctor --pred-dir ./preds --gt hf:lewiswatson/Frame2KG-YC2:validation_dev")


if __name__ == "__main__":
    main()
