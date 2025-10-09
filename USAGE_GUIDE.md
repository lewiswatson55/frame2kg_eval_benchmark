# Frame2KG Evaluation Toolkit - Usage Guide

## Installation

```bash
# Install the package
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## CLI Commands

The toolkit provides four main CLI commands:

### 1. frame2kg-eval - Main Evaluation

Evaluate predictions against ground truth with configurable thresholds.

```bash
# Basic usage with HuggingFace dataset
frame2kg-eval \
  --pred-dir ./predictions \
  --gt hf:lewiswatson/Frame2KG-YC2:validation_dev \
  --out results.csv

# With custom parameters
frame2kg-eval \
  --pred-dir ./predictions \
  --gt hf:lewiswatson/Frame2KG-YC2:validation_dev \
  --tau 0.3 \           # IoU threshold
  --alpha 0.7 \         # Blending weight (IoU vs text)
  --text-mode semantic \  # tfidf, semantic, or hybrid
  --text-floor 0.25 \   # Minimum text similarity
  --out results.csv

# With local ground truth
frame2kg-eval \
  --pred-dir ./predictions \
  --gt ./ground_truth_dir \
  --out results.csv
```

**Output**: CSV file with per-frame metrics and aggregated results (micro/macro).

Columns now include box closeness statistics:
- `box_mean_iou`: Mean IoU across matched node pairs (per frame, and summary rows)
- `box_std_iou`, `box_min_iou`, `box_max_iou`, `box_match_count`

### 2. frame2kg-sweep - Parameter Optimization

Find optimal τ (IoU threshold) and α (blending weight) parameters.

```bash
# Note: Use --taus and --alphas multiple times for multiple values
frame2kg-sweep \
  --pred-dir ./predictions \
  --gt hf:lewiswatson/Frame2KG-YC2:validation_dev \
  --taus 0.3 --taus 0.5 --taus 0.7 \
  --alphas 0.5 --alphas 0.7 --alphas 0.85 \
  --text-mode semantic \
  --out sweep_results.csv
```

**Output**: CSV with all parameter combinations ranked by combined F1 score.

### 3. frame2kg-aggregate - Multi-Run Analysis

Aggregate metrics across multiple prediction runs.

```bash
# Aggregate all runs in subdirectories
frame2kg-aggregate \
  --pred-root ./all_predictions \
  --gt hf:lewiswatson/Frame2KG-YC2:validation_dev \
  --pattern "*/*" \  # Directory pattern (e.g., variant/index)
  --tau 0.3 --alpha 0.7 \
  --out aggregate_results.csv
```

**Output**: CSV comparing all runs with their metrics.

### 4. frame2kg-doctor - Sanity Check

Check data integrity and identify issues.

```bash
# Check predictions only
frame2kg-doctor --pred-dir ./predictions

# Check both predictions and ground truth
frame2kg-doctor \
  --pred-dir ./predictions \
  --gt hf:lewiswatson/Frame2KG-YC2:validation_dev
```

**Output**: Console report with warnings and recommendations.

## Data Format

### Prediction Files

Files should be named: `<video_id>.<frame_number>.json`

Example: `video1.001.json`

```json
{
  "nodes": [
    {
      "id": "person1",
      "label": "person",
      "location": "0.1,0.2,0.3,0.4,0.9",
      "attributes": {"appearance": "blue shirt"}
    }
  ],
  "edges": [
    {
      "source": "person1",
      "target": "ball1",
      "predicate": "holding"
    }
  ]
}
```

### Location Format

Bounding boxes must be normalized coordinates in [0,1] and provided as a string with exactly 5 values:
- String only: `"x1,y1,x2,y2,confidence"`
- Constraints: x1 < x2, y1 < y2, and 0 ≤ confidence ≤ 1

### Required Schema Structure

**Nodes** must have:
- `id` (string): Unique identifier
- `label` (string): Node type/category
- `location` (string): "x1,y1,x2,y2,confidence" (floats), normalized to [0,1] with x1<x2 and y1<y2; confidence in [0,1]
- `attributes` (dict, optional): Additional properties

**Edges** must have:
- `source` (string): Source node ID
- `target` (string): Target node ID
- `predicate` (string): Relationship type

### Manifest (Optional)

For timing metrics, include `manifest.csv`:

```csv
video_id,frame_number,gen_wall_time_s
video1,1,1.5
video1,2,2.0
```

## Ground Truth Sources

### HuggingFace Dataset

```bash
# Format: hf:<dataset_name>:<split>
--gt hf:lewiswatson/Frame2KG-YC2:validation_dev
--gt hf:lewiswatson/Frame2KG-YC2:testing
```

### Local JSON Files

```bash
# Directory with same naming convention as predictions
--gt ./ground_truth_directory
```

## Configuration File

Create a YAML configuration file to set defaults:

```yaml
# config.yaml
tau: 0.3
alpha: 0.7
text_mode: semantic
text_fields: [id, label]
text_floor: 0.25
model_name: sentence-transformers/all-MiniLM-L6-v2
predicate_mode: exact
```

Use with: `--config config.yaml`

## Python API

```python
from frame2kg_eval.io.preds import PredictionLoader
from frame2kg_eval.io.groundtruth import create_ground_truth_adapter
from frame2kg_eval.matching.assign import two_stage_node_match
from frame2kg_eval.metrics.nodes import node_prf1, aggregate_micro

# Load data
pred_loader = PredictionLoader("./predictions")
gt_adapter = create_ground_truth_adapter("hf:lewiswatson/Frame2KG-YC2:validation_dev")

# Get specific frame
pred_graph = pred_loader.get_graph("video1", 1)
gt_graph = gt_adapter.get_graph("video1", 1)

# Node matching
match_result = two_stage_node_match(
    pred_graph["nodes"], gt_graph["nodes"],
    tau=0.3, alpha=0.7,
    text_mode="semantic"
)

# Compute metrics
    metrics = node_prf1(
        pred_graph["nodes"], 
        gt_graph["nodes"],
        match_result["mapping"]
    )

print(f"Node F1: {metrics['f1']:.3f}")

# Box IoU closeness
from frame2kg_eval.metrics.boxes import box_iou_stats
iou_matrix = match_result.get("matrices", {}).get("iou")
box_stats = box_iou_stats(pred_graph["nodes"], gt_graph["nodes"], match_result["mapping"], iou_matrix=iou_matrix)
print(f"Box IoU mean: {box_stats['mean_iou']:.3f} over {box_stats['count']} matches")
```

## Metrics Explained

### Node Metrics
- **Precision**: TP / (TP + FP) - Fraction of predicted nodes that are correct
- **Recall**: TP / (TP + FN) - Fraction of ground truth nodes that are found
- **F1**: Harmonic mean of precision and recall

### Box IoU Closeness
- **Mean IoU**: Average IoU across all matched node pairs in a frame
- **Count**: Number of matched node pairs in a frame

### Edge Metrics
- Edges match when both endpoints map correctly and predicates match
- Same PRF1 calculation as nodes

### JSON Validity
- **Validity Rate**: Percentage of files that are valid JSON
- Checks if files can be parsed as JSON and contain required structure

### Schema Conformity
- **Conformity Rate**: Percentage of files that conform to the expected schema
- Validates required fields (id, label, location for nodes; source, target, predicate for edges)
- Checks field types and structure
- Separate from JSON validity - valid JSON may not conform to schema

### Aggregation
- **Micro**: Sum all TP/FP/FN across frames, then compute metrics
- **Macro**: Compute metrics per frame, then average

### Two-Stage Matching
1. **IoU Gate**: Filter node pairs with IoU < τ
2. **Text Similarity**: Compute text similarity (TF-IDF or semantic)
3. **Blending**: Score = α × IoU + (1-α) × text_similarity
4. **Hungarian Assignment**: Optimal 1-to-1 matching

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=frame2kg_eval --cov-report=html

# Specific test file
pytest frame2kg_eval/tests/test_matching.py -v
```

## Troubleshooting

### Invalid JSON Files
- Check with `frame2kg-doctor`
- Ensure files contain `"nodes"` and `"edges"` keys
- Raw text files should have `.raw.txt` extension

### No Matches Found
- Lower IoU threshold (τ)
- Lower text floor threshold
- Check if bounding boxes are normalized [0,1]

### Memory Issues
- Use smaller batch sizes
- Clear embedding cache in long runs
- Consider TF-IDF instead of semantic mode

## Example Workflow

```bash
# 1. Check data integrity
frame2kg-doctor --pred-dir ./predictions

# 2. Find optimal parameters
frame2kg-sweep \
  --pred-dir ./predictions \
  --gt hf:lewiswatson/Frame2KG-YC2:validation_dev \
  --taus 0.2 --taus 0.3 --taus 0.4 --taus 0.5 \
  --alphas 0.5 --alphas 0.6 --alphas 0.7 --alphas 0.8 \
  --out sweep.csv

# 3. Run evaluation with best parameters
frame2kg-eval \
  --pred-dir ./predictions \
  --gt hf:lewiswatson/Frame2KG-YC2:validation_dev \
  --tau 0.3 --alpha 0.7 \
  --out final_results.csv

# 4. Compare multiple model versions
frame2kg-aggregate \
  --pred-root ./all_model_versions \
  --gt hf:lewiswatson/Frame2KG-YC2:validation_dev \
  --tau 0.3 --alpha 0.7 \
  --out comparison.csv
```
