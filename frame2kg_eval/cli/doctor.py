"""Sanity check tool for Frame2KG evaluation."""

import click
import json
from pathlib import Path
from collections import Counter

from frame2kg_eval.io.preds import PredictionLoader
from frame2kg_eval.io.groundtruth import create_ground_truth_adapter
from frame2kg_eval.io.schema import validate_graph
from frame2kg_eval.utils.logging import logger


@click.command()
@click.option("--pred-dir", type=click.Path(exists=True, path_type=Path), default=None,
              help="Directory containing prediction files")
@click.option("--gt", type=str, default=None,
              help="Ground truth spec (hf:dataset:split or path)")
@click.option("--verbose/--quiet", default=True,
              help="Verbose output")
def main(pred_dir, gt, verbose):
    """Run sanity checks on predictions and ground truth data."""
    
    logger.info("Frame2KG Doctor - Sanity Check Tool")
    logger.info("=" * 50)
    
    issues_found = []
    warnings = []
    
    # Check predictions if provided
    if pred_dir:
        logger.info(f"\n📁 Checking predictions: {pred_dir}")
        pred_dir = Path(pred_dir)
        
        if not pred_dir.exists():
            issues_found.append(f"Prediction directory does not exist: {pred_dir}")
        else:
            # Count file types
            json_files = list(pred_dir.glob("*.json"))
            raw_files = list(pred_dir.glob("*.raw.txt"))
            other_files = len(list(pred_dir.glob("*"))) - len(json_files) - len(raw_files)
            
            logger.info(f"  Found {len(json_files)} JSON files")
            logger.info(f"  Found {len(raw_files)} raw text files")
            if other_files > 0:
                warnings.append(f"Found {other_files} unexpected files in prediction directory")
            
            # Check file naming
            from frame2kg_eval.utils.ids import parse_filename
            invalid_names = []
            for f in json_files + raw_files:
                if not parse_filename(f.name):
                    invalid_names.append(f.name)
            
            if invalid_names:
                issues_found.append(f"Invalid file names: {invalid_names[:5]}...")
            
            # Load predictions and check validity
            try:
                pred_loader = PredictionLoader(pred_dir)
                valid, invalid, total = pred_loader.count_valid()
                
                logger.info(f"  Valid graphs: {valid}/{total} ({valid/total*100:.1f}%)")
                
                if invalid > 0:
                    warnings.append(f"{invalid} files contain invalid JSON or graph structure")
                
                # Sample validation
                if verbose and json_files:
                    sample_file = json_files[0]
                    logger.info(f"\n  Sampling {sample_file.name}:")
                    try:
                        with open(sample_file) as f:
                            data = json.load(f)
                        
                        is_valid = validate_graph(data)
                        logger.info(f"    Valid graph: {is_valid}")
                        
                        if is_valid:
                            logger.info(f"    Nodes: {len(data.get('nodes', []))}")
                            logger.info(f"    Edges: {len(data.get('edges', []))}")
                            
                            # Check node structure
                            if data.get("nodes"):
                                node = data["nodes"][0]
                                logger.info(f"    Sample node keys: {list(node.keys())}")
                            
                            # Check edge structure
                            if data.get("edges"):
                                edge = data["edges"][0]
                                logger.info(f"    Sample edge keys: {list(edge.keys())}")
                    
                    except Exception as e:
                        issues_found.append(f"Failed to sample {sample_file.name}: {e}")
            
            except Exception as e:
                issues_found.append(f"Failed to load predictions: {e}")
            
            # Check for manifest
            manifest_path = pred_dir / "manifest.csv"
            if manifest_path.exists():
                logger.info(f"  ✓ Found manifest.csv")
                
                # Quick manifest check
                try:
                    import csv
                    with open(manifest_path) as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        logger.info(f"    Manifest entries: {len(rows)}")
                        
                        if rows and "gen_wall_time_s" in rows[0]:
                            logger.info(f"    ✓ Timing information available")
                
                except Exception as e:
                    warnings.append(f"Failed to read manifest: {e}")
            else:
                warnings.append("No manifest.csv found (timing metrics unavailable)")
    
    # Check ground truth if provided
    if gt:
        logger.info(f"\n🎯 Checking ground truth: {gt}")
        
        try:
            gt_adapter = create_ground_truth_adapter(gt)
            frame_count = gt_adapter.count_frames()
            logger.info(f"  Total frames: {frame_count}")
            
            # Sample some frames
            sample_count = 0
            node_labels = Counter()
            edge_predicates = Counter()
            
            for video_id, frame_no, graph in gt_adapter.iter_frames():
                if sample_count >= 10:  # Sample first 10 frames
                    break
                
                for node in graph.get("nodes", []):
                    node_labels[node.get("label", "unknown")] += 1
                
                for edge in graph.get("edges", []):
                    edge_predicates[edge.get("predicate", "unknown")] += 1
                
                sample_count += 1
            
            if node_labels:
                logger.info(f"  Top node labels: {node_labels.most_common(5)}")
            if edge_predicates:
                logger.info(f"  Top predicates: {edge_predicates.most_common(5)}")
        
        except Exception as e:
            issues_found.append(f"Failed to load ground truth: {e}")
    
    # Check overlap between predictions and ground truth
    if pred_dir and gt:
        logger.info("\n🔄 Checking data overlap:")
        
        try:
            pred_loader = PredictionLoader(pred_dir)
            gt_adapter = create_ground_truth_adapter(gt)
            
            pred_keys = set(pred_loader.get_index().keys())
            gt_keys = set()
            
            for video_id, frame_no, _ in gt_adapter.iter_frames():
                gt_keys.add((video_id, frame_no))
            
            overlap = pred_keys & gt_keys
            pred_only = pred_keys - gt_keys
            gt_only = gt_keys - pred_keys
            
            logger.info(f"  Overlapping frames: {len(overlap)}")
            logger.info(f"  Predictions only: {len(pred_only)}")
            logger.info(f"  Ground truth only: {len(gt_only)}")
            
            if pred_only:
                warnings.append(f"{len(pred_only)} predictions have no matching ground truth")
            if gt_only and len(gt_only) > len(overlap):
                warnings.append(f"{len(gt_only)} ground truth frames have no predictions")
        
        except Exception as e:
            issues_found.append(f"Failed to check overlap: {e}")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    
    if not issues_found and not warnings:
        logger.success("✅ All checks passed! Data looks good.")
    else:
        if warnings:
            logger.warning(f"⚠️  Found {len(warnings)} warnings:")
            for w in warnings:
                logger.warning(f"   - {w}")
        
        if issues_found:
            logger.error(f"❌ Found {len(issues_found)} issues:")
            for issue in issues_found:
                logger.error(f"   - {issue}")
    
    # Recommendations
    if issues_found or warnings:
        logger.info("\n💡 Recommendations:")
        
        if any("Invalid file names" in i for i in issues_found):
            logger.info("  - Ensure files follow naming pattern: <video_id>.<frame_no>.json")
        
        if any("invalid JSON" in w for w in warnings):
            logger.info("  - Check that predictions contain 'nodes' and 'edges' keys")
            logger.info("  - Validate JSON syntax in .raw.txt files")
        
        if any("no matching ground truth" in w for w in warnings):
            logger.info("  - Verify video IDs and frame numbers match dataset")


if __name__ == "__main__":
    main()
