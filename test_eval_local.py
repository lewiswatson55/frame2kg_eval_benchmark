#!/usr/bin/env python3
"""Test evaluation with local data (no network dependency)."""

import json
import tempfile
from pathlib import Path
import subprocess
import sys


def create_test_data():
    """Create test predictions and ground truth."""
    tmpdir = tempfile.mkdtemp(prefix="frame2kg_test_")
    tmppath = Path(tmpdir)
    
    # Create predictions directory
    pred_dir = tmppath / "predictions"
    pred_dir.mkdir()
    
    # Create ground truth directory
    gt_dir = tmppath / "ground_truth"
    gt_dir.mkdir()
    
    # Sample data for 3 frames
    frames = [
        {
            "video_id": "video1",
            "frame_no": 1,
            "pred": {
                "nodes": [
                    {"id": "p1", "label": "person", "location": "0.1,0.1,0.3,0.3"},
                    {"id": "b1", "label": "ball", "location": "0.5,0.5,0.7,0.7"}
                ],
                "edges": [
                    {"source": "p1", "target": "b1", "predicate": "holding"}
                ]
            },
            "gt": {
                "nodes": [
                    {"id": "g1", "label": "person", "location": "0.1,0.1,0.3,0.3"},
                    {"id": "g2", "label": "ball", "location": "0.5,0.5,0.7,0.7"},
                    {"id": "g3", "label": "dog", "location": "0.8,0.8,0.9,0.9"}  # FN
                ],
                "edges": [
                    {"source": "g1", "target": "g2", "predicate": "holding"},
                    {"source": "g1", "target": "g3", "predicate": "near"}  # FN
                ]
            }
        },
        {
            "video_id": "video1", 
            "frame_no": 2,
            "pred": {
                "nodes": [
                    {"id": "c1", "label": "car", "location": "0.2,0.2,0.5,0.5"},
                    {"id": "p2", "label": "person", "location": "0.6,0.6,0.8,0.8"}
                ],
                "edges": [
                    {"source": "p2", "target": "c1", "predicate": "driving"}
                ]
            },
            "gt": {
                "nodes": [
                    {"id": "v1", "label": "vehicle", "location": "0.2,0.2,0.5,0.5"},  # Similar to car
                    {"id": "h1", "label": "human", "location": "0.6,0.6,0.8,0.8"}  # Similar to person
                ],
                "edges": [
                    {"source": "h1", "target": "v1", "predicate": "driving"}
                ]
            }
        },
        {
            "video_id": "video2",
            "frame_no": 1,
            "pred": {
                "nodes": [
                    {"id": "t1", "label": "tree", "location": "0.3,0.3,0.4,0.4"}
                ],
                "edges": []
            },
            "gt": {
                "nodes": [
                    {"id": "plant1", "label": "tree", "location": "0.3,0.3,0.4,0.4"}
                ],
                "edges": []
            }
        }
    ]
    
    # Write prediction and ground truth files
    for frame in frames:
        # Prediction
        pred_file = pred_dir / f"{frame['video_id']}.{frame['frame_no']:03d}.json"
        with open(pred_file, 'w') as f:
            json.dump(frame['pred'], f, indent=2)
        
        # Ground truth
        gt_file = gt_dir / f"{frame['video_id']}.{frame['frame_no']:03d}.json"
        with open(gt_file, 'w') as f:
            json.dump(frame['gt'], f, indent=2)
    
    # Add one invalid prediction
    invalid_file = pred_dir / "video2.002.raw.txt"
    with open(invalid_file, 'w') as f:
        f.write("Failed to generate valid JSON")
    
    # Create manifest
    manifest_file = pred_dir / "manifest.csv"
    with open(manifest_file, 'w') as f:
        f.write("video_id,frame_number,gen_wall_time_s\n")
        f.write("video1,1,1.2\n")
        f.write("video1,2,1.5\n")
        f.write("video2,1,0.8\n")
        f.write("video2,2,2.1\n")
    
    return tmppath, pred_dir, gt_dir


def run_evaluation():
    """Run the evaluation pipeline."""
    print("=" * 60)
    print("Frame2KG Local Evaluation Test")
    print("=" * 60)
    
    # Create test data
    print("\n1. Creating test data...")
    tmppath, pred_dir, gt_dir = create_test_data()
    print(f"   ✓ Predictions: {pred_dir}")
    print(f"   ✓ Ground truth: {gt_dir}")
    
    # Run doctor check
    print("\n2. Running sanity check...")
    doctor_cmd = ["frame2kg-doctor", "--pred-dir", str(pred_dir), "--gt", str(gt_dir)]
    result = subprocess.run(doctor_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("   ✓ Sanity check passed")
    
    # Run evaluation
    print("\n3. Running evaluation...")
    output_file = tmppath / "results.csv"
    eval_cmd = [
        "frame2kg-eval",
        "--pred-dir", str(pred_dir),
        "--gt", str(gt_dir),
        "--tau", "0.5",
        "--alpha", "0.7", 
        "--text-mode", "tfidf",
        "--out", str(output_file)
    ]
    
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    if result.returncode == 0:
        print("   ✓ Evaluation completed")
        
        # Read and display results
        if output_file.exists():
            print("\n4. Results summary:")
            import csv
            with open(output_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                # Show per-frame results
                print("\n   Per-frame metrics:")
                for row in rows[:5]:  # Show first 5 rows
                    if 'video_id' in row and row['video_id']:
                        if row['video_id'] == 'SUMMARY':
                            print(f"\n   {row['frame_no']} Aggregation:")
                            print(f"     Node F1: {float(row.get('node_f1', 0)):.3f}")
                            print(f"     Edge F1: {float(row.get('edge_f1', 0)):.3f}")
                        else:
                            print(f"     {row['video_id']}.{row['frame_no']}: "
                                  f"Node F1={float(row.get('node_f1', 0)):.3f}, "
                                  f"Edge F1={float(row.get('edge_f1', 0)):.3f}")
    else:
        print(f"   ✗ Evaluation failed: {result.returncode}")
    
    # Run parameter sweep
    print("\n5. Running parameter sweep...")
    sweep_file = tmppath / "sweep.csv"
    sweep_cmd = [
        "frame2kg-sweep",
        "--pred-dir", str(pred_dir),
        "--gt", str(gt_dir),
        "--taus", "0.3", "--taus", "0.5", "--taus", "0.7",
        "--alphas", "0.5", "--alphas", "0.7", "--alphas", "0.9",
        "--out", str(sweep_file),
        "--text-mode", "tfidf"
    ]
    
    result = subprocess.run(sweep_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("   ✓ Sweep completed")
        
        # Show best parameters
        if sweep_file.exists():
            with open(sweep_file, 'r') as f:
                reader = csv.DictReader(f)
                best = next(reader)
                print(f"\n   Best parameters:")
                print(f"     τ={float(best['tau']):.2f}, α={float(best['alpha']):.2f}")
                print(f"     Combined F1: {float(best['combined_f1']):.3f}")
    else:
        print(f"   ✗ Sweep failed: {result.returncode}")
    
    print(f"\n✓ Test complete! Results saved to {tmppath}")
    print(f"  You can explore: {tmppath}")
    return tmppath


if __name__ == "__main__":
    tmppath = run_evaluation()
