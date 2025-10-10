# Frame2KG Evaluation Toolkit

Production-ready evaluation framework for Frame-to-Knowledge-Graph (Frame2KG) task assessment.

## Features

- **Deterministic evaluation**: Repeatable metrics from predictions and ground-truth splits
- **Two-stage node matching**: IoU gating with text similarity refinement using Hungarian assignment
- **Comprehensive metrics**: Node & edge precision/recall/F1 with TP/FP/FN counts (micro & macro)
- **Flexible similarity modes**: TF-IDF, semantic embeddings, or hybrid approaches
- **Threshold optimization**: Grid search over IoU (τ) and blending (α) parameters
- **JSON schema conformity**: Validates graph structure against expected schema format
- **Matched-pair IoU (box IoU)**: Per-frame mean IoU for matched nodes, with micro (match-weighted) and macro (per-frame mean) summaries
- **Robust I/O**: Handles various file formats and missing frames gracefully
- **HuggingFace integration**: Direct support for `lewiswatson/Frame2KG-YC2` dataset

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Single Run Evaluation

```bash
frame2kg-eval \
  --pred-dir ./predictions/model-v1/run1 \
  --gt hf:lewiswatson/Frame2KG-YC2:validation_dev \
  --tau 0.3 --alpha 0.7 \
  --text-mode semantic \
  --out results.csv
```

### Threshold Sweep

```bash
frame2kg-sweep \
  --pred-dir ./predictions/model-v1/run1 \
  --gt hf:lewiswatson/Frame2KG-YC2:validation_dev \
  --taus 0.3 0.5 0.7 \
  --alphas 0.5 0.7 0.85 \
  --out sweep_results.csv
```

### Aggregate Multiple Runs

```bash
frame2kg-aggregate \
  --pred-root ./predictions \
  --gt hf:lewiswatson/Frame2KG-YC2:validation_dev \
  --tau 0.3 --alpha 0.7 \
  --out aggregate_results.csv
```

### Sanity Check

```bash
frame2kg-doctor \
  --pred-dir ./predictions/model-v1/run1 \
  --gt hf:lewiswatson/Frame2KG-YC2:validation_dev
```

## Data Format

### Predictions
Place prediction files in a directory with naming convention:
```
<video_id>.<frame_number>.json
```

Each JSON file should conform to the following schema:
```json
{
  "nodes": [
    {
      "id": "string",           // Required: unique node identifier
      "label": "string",         // Required: node type/category
      "location": "x1,y1,x2,y2,confidence",  // Required string, normalized [0,1]
      "attributes": {            // Optional: additional properties
        "key": "value"
      }
    }
  ],
  "edges": [
    {
      "source": "nodeId",        // Required: source node ID
      "target": "nodeId",        // Required: target node ID
      "predicate": "string"      // Required: relationship type
    }
  ]
}
```

**Note**: The evaluation includes schema conformity checking that validates:
- Required fields are present (`id`, `label`, `location` for nodes; `source`, `target`, `predicate` for edges)
- Field types are correct (strings). `location` must be a string with exactly 5 numeric values (x1,y1,x2,y2,confidence) normalized to [0,1] and satisfy x1<x2, y1<y2
- Structure follows the expected schema

### Ground Truth
Supports two modes:
1. **HuggingFace dataset**: `hf:lewiswatson/Frame2KG-YC2:<split>`
2. **Local JSON files**: Directory with same naming convention as predictions

## Evaluation Methodology

### Node Matching
1. Compute IoU matrix between predicted and ground-truth bounding boxes
2. Calculate text similarity (TF-IDF or semantic embeddings)
3. Gate pairs by IoU ≥ τ threshold
4. Blend scores: `α * IoU + (1-α) * text_similarity`
5. Apply Hungarian algorithm for optimal 1-to-1 assignment

### Edge Scoring
Edges match when:
- Both endpoints map through node assignment
- Predicates are equal (exact or normalised match)

Normalised comparison lowercases and strips punctuation so common formatting variants (for example `next-to` vs `NEXT_TO`) line up, while deliberately avoiding broader semantic embeddings that tend to conflate genuinely distinct relation labels in this domain.

### Metrics
- **Node metrics**: Precision, Recall, F1, TP/FP/FN counts
- **Edge metrics**: Similar to nodes, with optional edge-by-label baseline
- **Validity**: JSON parsing success rate
- **Schema Conformity**: Validates structure against expected graph schema
- **Matched-pair IoU (box IoU)**: Mean/median IoU across matched node pairs (reported per-frame as `box_mean_iou` / `box_median_iou`). Micro = mean IoU across all matched pairs (weighted by match count); Macro = unweighted mean of per-frame mean IoU.
- **Timing**: Mean generation time from manifest.csv

Matching entry points seed Python, NumPy, and (when available) Torch RNGs to `42` so the assignment stage remains deterministic across runs; adjust via `seed_matching` if required.

Frames with missing or invalid predictions are evaluated as empty outputs so they contribute zero precision/recall; enable `strict_mode` to add an explicit false-positive penalty for those cases.

## Configuration

Default parameters in `config/defaults.yaml`:
```yaml
tau: 0.3
alpha: 0.7
text_mode: semantic
text_fields: [id, label]  # Add more fields (e.g. attributes, description) to include in similarity
text_floor: 0.25
model_name: sentence-transformers/all-MiniLM-L6-v2
predicate_mode: normalised
```

Specify as many `text_fields` as required, nested containers (like attribute dictionaries or lists of tags) are flattened before text similarity is computed.

## Python API

```python
from frame2kg_eval.matching import two_stage_node_match
from frame2kg_eval.metrics import node_prf1, edge_prf1

# Load predictions and ground truth
pred_nodes = [...]
gt_nodes = [...]

# Perform matching
result = two_stage_node_match(
    pred_nodes, gt_nodes,
    tau=0.3, alpha=0.7,
    text_mode="semantic"
)

# Calculate metrics
node_metrics = node_prf1(pred_nodes, gt_nodes, result["mapping"])
```

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{frame2kg_eval,
  title = {Frame2KG Evaluation Toolkit},
  author = {TBC},
  year = {2025},
  url = {TBC}
}
```

## License

TBC
