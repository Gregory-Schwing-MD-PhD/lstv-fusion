# LSTV Fusion Pipeline - Usage Guide

## Overview

This pipeline has **3 steps**:

1. **Fusion** - Combine SPINEPS + Uncertainty → CSV results
2. **Viewer** - Generate interactive HTML with embedded images
3. **Performance** - Calculate metrics after clinical audit

---

## Step 1: Run Fusion

### Prerequisites

You must already have:
- ✅ SPINEPS segmentations (`*_seg-vert.nii.gz`)
- ✅ Uncertainty metrics (`*_uncertainty_metrics.csv`)
- ✅ Valid IDs file (`valid_id.npy`)

### Run the Fusion

```bash
python src/fuse_consensus.py \
    --spineps-dir /path/to/spineps_output \
    --uncertainty-dir /path/to/uncertainty_output \
    --output-dir ./output \
    --valid-ids ./models/valid_id.npy \
    --entropy-threshold 5.0
```

### Output

```
output/
├── fusion_summary.csv          # Main results (all studies)
├── audit_queue/
│   └── audit_queue.json        # 60 cases for clinical review
├── relabeled_masks/
│   └── *.nii.gz                # Corrected segmentations
└── reports/
    └── detailed_report.json
```

---

## Step 2: Generate Interactive Viewer

### Run the Viewer Generator

```bash
python src/generate_viewer.py \
    --fusion-csv output/fusion_summary.csv \
    --spineps-dir /path/to/spineps_output \
    --output-html viewer.html \
    --max-studies 50
```

### Output

- `viewer.html` - Standalone HTML file (embeds all data)

### Open in Browser

```bash
firefox viewer.html
# or
open viewer.html  # macOS
```

### Viewer Features

**Interactive Navigation:**
- Dropdown selector to switch between studies
- LSTV status badge (⚠️ LSTV or ✓ Normal)

**Image Display:**
- Mid-sagittal slice from SPINEPS segmentation
- Hover over vertebrae → Tooltip with probability + entropy
- Cursor position tracking

**Sidebar Panels:**
- **Study Metrics:** LSTV status, entropy, confidence
- **Vertebrae List:** Per-vertebra entropy bars (green→yellow→red)
- Click vertebra in list to highlight (future feature)

**Export:**
- Download button for simple text report

---

## Step 3: Calculate Performance (After Clinical Audit)

### Prerequisites

You must have:
- ✅ Fusion results (`fusion_summary.csv`)
- ✅ Clinical audit results (`audit_results.json`)

The audit results come from the **lstv-annotation-tool** repo after radiologist review.

### Run Performance Calculation

```bash
python src/calculate_real_performance.py \
    --fusion-results output/fusion_summary.csv \
    --audit-results audit_results.json \
    --output-dir output/metrics
```

### Output

```
output/metrics/
├── performance_metrics.json    # Full metrics
├── summary_report.txt          # Human-readable
├── confusion_matrix.png
├── roc_curve.png
├── entropy_distribution.png
└── performance_by_confidence.png
```

---

## Docker Usage

### Build Container

```bash
cd docker
./build.sh
```

### Run Fusion in Container

```bash
# Using Docker
docker run --rm \
    -v $(pwd)/spineps_output:/data/spineps \
    -v $(pwd)/uncertainty_output:/data/uncertainty \
    -v $(pwd)/output:/data/output \
    -v $(pwd)/models:/data/models \
    go2432/lstv-fusion:latest \
    python /app/src/fuse_consensus.py \
    --spineps-dir /data/spineps \
    --uncertainty-dir /data/uncertainty \
    --output-dir /data/output \
    --valid-ids /data/models/valid_id.npy

# Using Singularity (HPC clusters)
singularity pull lstv-fusion.sif docker://go2432/lstv-fusion:latest

singularity exec \
    --bind $(pwd)/spineps_output:/data/spineps \
    --bind $(pwd)/uncertainty_output:/data/uncertainty \
    --bind $(pwd)/output:/data/output \
    --bind $(pwd)/models:/data/models \
    lstv-fusion.sif \
    python /app/src/fuse_consensus.py \
    --spineps-dir /data/spineps \
    --uncertainty-dir /data/uncertainty \
    --output-dir /data/output \
    --valid-ids /data/models/valid_id.npy
```

---

## Configuration Options

### Entropy Threshold

Controls LSTV detection sensitivity:

```bash
# More conservative (fewer false positives)
--entropy-threshold 5.5

# More sensitive (catch more LSTV cases)
--entropy-threshold 4.5

# Default (balanced)
--entropy-threshold 5.0
```

### Process Subset

For testing, process only a few studies:

```bash
# Add this flag to fuse_consensus.py
--max-studies 10
```

---

## Troubleshooting

### "FileNotFoundError: No such file"

Make sure paths are correct:
```bash
ls /path/to/spineps_output/*.nii.gz
ls /path/to/uncertainty_output/*.csv
ls ./models/valid_id.npy
```

### "No studies processed"

Check that study IDs match between:
- SPINEPS filenames: `1020394063_seg-vert.nii.gz`
- Uncertainty filenames: `1020394063_uncertainty_metrics.csv`
- Valid IDs in `valid_id.npy`

### Viewer shows "No image data"

- The viewer generator needs access to SPINEPS segmentations
- Make sure `--spineps-dir` points to correct directory
- Check that study IDs in `fusion_summary.csv` match filenames

### Performance calculation fails

- Make sure `audit_results.json` has correct format
- Check that study IDs in audit match those in fusion results

---

## Expected Runtime

| Task | Time | Output Size |
|------|------|-------------|
| Fusion (500 studies) | 2-4 hours | ~500 KB CSV |
| Viewer generation | 5-10 minutes | ~5-20 MB HTML |
| Performance calc | 1-2 minutes | ~2 MB (with plots) |

---

## Support

- **Email:** go2432@wayne.edu
- **GitHub Issues:** (your repo URL)
- **Documentation:** See docstrings in source code

---

## Quick Reference

```bash
# Complete workflow
cd lstv-fusion

# 1. Fusion
python src/fuse_consensus.py \
    --spineps-dir spineps_output \
    --uncertainty-dir uncertainty_output \
    --output-dir output \
    --valid-ids models/valid_id.npy

# 2. Viewer
python src/generate_viewer.py \
    --fusion-csv output/fusion_summary.csv \
    --spineps-dir spineps_output \
    --output-html viewer.html

# 3. Open viewer
firefox viewer.html

# 4. Performance (after clinical audit)
python src/calculate_real_performance.py \
    --fusion-results output/fusion_summary.csv \
    --audit-results audit_results.json \
    --output-dir output/metrics
```
