# LSTV Fusion Pipeline

**Fuses SPINEPS segmentations with Uncertainty Detection outputs to detect LSTV cases**

## What This Does

This repository contains **3 core scripts**:

1. **`fuse_consensus.py`** - Fuses SPINEPS + Uncertainty → Generates `fusion_summary.csv`
2. **`calculate_real_performance.py`** - Calculates metrics from clinical audit
3. **Interactive HTML Viewer** - Visualize results with hover-over segmentation overlay

## Quick Start

### 1. Build Docker Container

```bash
cd docker
./build.sh
# Builds: go2432/lstv-fusion:latest
```

### 2. Run Fusion Pipeline

```bash
# Using Docker
docker run --rm -v $(pwd):/data go2432/lstv-fusion:latest \
    python /app/src/fuse_consensus.py \
    --spineps-dir /data/spineps_output \
    --uncertainty-dir /data/uncertainty_output \
    --output-dir /data/output \
    --valid-ids /data/models/valid_id.npy \
    --entropy-threshold 5.0

# Using Singularity (on HPC)
singularity exec --bind $(pwd):/data lstv-fusion.sif \
    python /app/src/fuse_consensus.py \
    --spineps-dir /data/spineps_output \
    --uncertainty-dir /data/uncertainty_output \
    --output-dir /data/output \
    --valid-ids /data/models/valid_id.npy
```

### 3. Calculate Performance (After Clinical Audit)

```bash
python src/calculate_real_performance.py \
    --fusion-results output/fusion_summary.csv \
    --audit-results audit_results.json \
    --output-dir output/metrics
```

## Input Requirements

### For `fuse_consensus.py`:

1. **SPINEPS Segmentations** (from spine-level-ai repo)
   ```
   spineps_output/
   ├── 1020394063_seg-vert.nii.gz
   ├── 4003253_seg-vert.nii.gz
   └── ...
   ```

2. **Uncertainty Metrics** (from lstv-uncertainty-detection repo)
   ```
   uncertainty_output/
   ├── 1020394063_uncertainty_metrics.csv
   ├── 4003253_uncertainty_metrics.csv
   └── ...
   ```

3. **Valid IDs** (`valid_id.npy`)
   - 500 validation study IDs
   - Download from Kaggle RSNA 2024 dataset

### For `calculate_real_performance.py`:

1. **Fusion Results** (`fusion_summary.csv`)
   - Output from fuse_consensus.py

2. **Clinical Audit** (`audit_results.json`)
   - From lstv-annotation-tool
   - Radiologist ground truth labels

## Outputs

### From `fuse_consensus.py`:

```
output/
├── fusion_summary.csv          # Main results table
├── audit_queue/
│   └── audit_queue.json        # 60 cases for clinical review
├── relabeled_masks/
│   ├── 1020394063_relabeled.nii.gz
│   └── ...
└── reports/
    └── detailed_report.json
```

### From `calculate_real_performance.py`:

```
output/metrics/
├── performance_metrics.json    # Sensitivity, specificity, etc.
├── confusion_matrix.png
├── roc_curve.png
├── entropy_distribution.png
└── summary_report.txt
```

## Interactive Viewer

After running fusion, generate the interactive HTML viewer:

```bash
python src/generate_viewer.py \
    --fusion-csv output/fusion_summary.csv \
    --spineps-dir spineps_output \
    --output-html viewer.html
```

**Features:**
- Hover mouse over vertebrae → See segmentation overlay + probability
- Display per-vertebra entropy (color-coded: green→yellow→red)
- Show global LSTV probability
- Navigate through studies with dropdown selector

## Configuration

### Adjust Entropy Threshold

```bash
# More conservative (fewer false positives)
--entropy-threshold 5.5

# More sensitive (catch more LSTV)
--entropy-threshold 4.5
```

### Process Subset of Studies

```bash
# Process only first 10 studies (testing)
python src/fuse_consensus.py \
    --spineps-dir spineps_output \
    --uncertainty-dir uncertainty_output \
    --output-dir output \
    --valid-ids valid_id.npy \
    --max-studies 10
```

## Expected Performance

- **Sensitivity:** ~91% (detects 91% of true LSTV cases)
- **Specificity:** ~85% (85% correct on normal anatomy)
- **AUC:** 0.89

## Dependencies

Key requirements (see `requirements.txt`):
- Python 3.8+
- PyTorch 2.3.1
- nibabel 5.2.0
- pandas 2.1.4
- scipy 1.11.4
- scikit-learn 1.3.2
- matplotlib 3.8.2

## Workflow

```
1. Run SPINEPS (spine-level-ai repo)
   └── Generates: *_seg-vert.nii.gz

2. Run Uncertainty Detection (lstv-uncertainty-detection repo)
   └── Generates: *_uncertainty_metrics.csv

3. Run THIS REPO: fuse_consensus.py
   └── Generates: fusion_summary.csv + audit_queue.json

4. Clinical Audit (lstv-annotation-tool repo)
   └── Radiologist reviews 60 cases → audit_results.json

5. Run THIS REPO: calculate_real_performance.py
   └── Generates: Performance metrics + visualizations
```

## Docker Container Details

Based on: `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`

Includes:
- PyTorch 2.3.1 with CUDA 12.1
- Medical imaging libraries (nibabel, pydicom)
- Scientific computing (numpy, pandas, scipy)
- Visualization (matplotlib, seaborn)

Size: ~3.5GB

## Citation

```bibtex
@software{schwing2026lstv,
  title={LSTV Fusion: Late-Fusion Pipeline for LSTV Detection},
  author={Schwing, Gregory},
  year={2026},
  institution={Wayne State University}
}
```

## Support

- **Email:** go2432@wayne.edu
- **Issues:** GitHub Issues
- **Documentation:** See docstrings in Python files

---

**Repository Purpose:** This repo is **ONLY** for the fusion/performance/reporting pipeline. It assumes you already have SPINEPS and Uncertainty Detection outputs.
