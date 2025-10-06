#!/bin/bash

echo "Frame2KG CLI Demo"
echo "================="

# Create sample prediction directory
mkdir -p sample_preds

# Create valid prediction
cat > sample_preds/video1.001.json << EOF
{
    "nodes": [
        {"id": "person1", "label": "person", "location": "0.1,0.2,0.3,0.4"},
        {"id": "ball1", "label": "ball", "location": "0.5,0.6,0.7,0.8"}
    ],
    "edges": [
        {"source": "person1", "target": "ball1", "predicate": "holding"}
    ]
}
EOF

# Create another valid prediction
cat > sample_preds/video1.002.json << EOF
{
    "nodes": [
        {"id": "car1", "label": "vehicle", "location": "0.2,0.2,0.5,0.5"}
    ],
    "edges": []
}
EOF

# Create invalid JSON (to test validity checking)
echo "invalid json {" > sample_preds/video1.003.raw.txt

# Create manifest for timing
cat > sample_preds/manifest.csv << EOF
video_id,frame_number,gen_wall_time_s
video1,1,1.5
video1,2,2.0
video1,3,3.5
EOF

echo ""
echo "✓ Created sample predictions in ./sample_preds/"
echo ""
echo "Available CLI commands:"
echo "----------------------"
echo ""
echo "1. Sanity check (frame2kg-doctor):"
echo "   frame2kg-doctor --pred-dir ./sample_preds"
echo ""
echo "2. Evaluation (frame2kg-eval):"
echo "   frame2kg-eval --pred-dir ./sample_preds --gt hf:lewiswatson/Frame2KG-YC2:validation_dev --out results.csv"
echo ""
echo "3. Parameter sweep (frame2kg-sweep):"
echo "   frame2kg-sweep --pred-dir ./sample_preds --gt hf:lewiswatson/Frame2KG-YC2:validation_dev --out sweep.csv"
echo ""
echo "4. Aggregate runs (frame2kg-aggregate):"
echo "   frame2kg-aggregate --pred-root . --pattern 'sample_preds' --gt hf:lewiswatson/Frame2KG-YC2:validation_dev --out aggregate.csv"
echo ""
echo "Note: For real evaluation, you'll need actual prediction files matching the HuggingFace dataset."
