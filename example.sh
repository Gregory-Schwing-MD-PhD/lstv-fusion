#!/bin/bash
# Example: Complete pipeline workflow

set -e

echo "=================================================="
echo "LSTV Fusion Pipeline - Example Workflow"
echo "=================================================="

# Setup paths
SPINEPS_DIR="spineps_output"
UNCERTAINTY_DIR="uncertainty_output"
OUTPUT_DIR="output"
MODELS_DIR="models"

echo ""
echo "Step 1: Run Fusion..."
python src/fuse_consensus.py \
    --spineps-dir $SPINEPS_DIR \
    --uncertainty-dir $UNCERTAINTY_DIR \
    --output-dir $OUTPUT_DIR \
    --valid-ids $MODELS_DIR/valid_id.npy \
    --entropy-threshold 5.0

echo ""
echo "✓ Fusion complete!"
echo "Results: $OUTPUT_DIR/fusion_summary.csv"

echo ""
echo "Step 2: Generate Interactive Viewer..."
python src/generate_viewer.py \
    --fusion-csv $OUTPUT_DIR/fusion_summary.csv \
    --spineps-dir $SPINEPS_DIR \
    --output-html viewer.html \
    --max-studies 50

echo ""
echo "✓ Viewer generated!"
echo "Open: firefox viewer.html"

echo ""
echo "=================================================="
echo "Pipeline complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Open viewer.html in browser"
echo "  2. Upload audit_queue.json to lstv-annotation-tool"
echo "  3. After clinical audit, run calculate_real_performance.py"
echo ""
