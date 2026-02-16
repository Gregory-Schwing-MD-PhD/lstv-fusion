#!/usr/bin/env python3
"""
Unified LSTV Visualization - Shows Everything in One View

Overlays on DICOM viewer:
1. SPINEPS segmentation masks (colored by instance)
2. Centroids (marked with X)
3. Uncertainty values (color-coded: green=low, yellow=medium, red=high)
4. Predicted anatomical labels

Usage:
    python visualize_fusion.py \
        --study_id 1020394063 \
        --dicom_dir /path/to/train_images \
        --spineps_dir /path/to/spine-level-ai/results \
        --uncertainty_dir /path/to/lstv-uncertainty-detection/data/output/production \
        --output_dir ./visualizations
        
This creates interactive HTML with DICOM + all overlays!
"""

import argparse
import json
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import pydicom
from natsort import natsorted
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dicom_study(study_path: Path, series_id: str) -> tuple:
    """Load DICOM series and return middle slice + full volume"""
    series_path = study_path / series_id
    dicom_files = natsorted(list(series_path.glob('*.dcm')))
    
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files in {series_path}")
    
    # Load all slices
    slices = []
    for dcm_file in dicom_files:
        dcm = pydicom.dcmread(str(dcm_file))
        slices.append(dcm.pixel_array)
    
    volume = np.stack(slices)
    
    # Normalize to 8-bit
    vmin, vmax = volume.min(), volume.max()
    if vmax > vmin:
        volume = ((volume - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    
    # Get middle slice
    mid_idx = volume.shape[0] // 2
    mid_slice = volume[mid_idx]
    
    return mid_slice, volume, mid_idx


def load_spineps_segmentation(spineps_dir: Path, study_id: str) -> tuple:
    """Load SPINEPS segmentation and return middle slice"""
    seg_path = spineps_dir / "segmentations" / f"{study_id}_seg-vert.nii.gz"
    
    if not seg_path.exists():
        raise FileNotFoundError(f"SPINEPS segmentation not found: {seg_path}")
    
    seg_nii = nib.load(seg_path)
    seg_data = seg_nii.get_fdata().astype(int)
    
    # Get middle sagittal slice (left-right center)
    mid_idx = seg_data.shape[2] // 2
    mid_slice = seg_data[:, :, mid_idx]
    
    return mid_slice, seg_data, seg_nii.affine


def load_centroids(spineps_dir: Path, study_id: str) -> dict:
    """Load SPINEPS centroids"""
    centroid_path = spineps_dir / "centroids" / f"{study_id}_centroids.json"
    
    if not centroid_path.exists():
        logger.warning(f"Centroids not found: {centroid_path}")
        return {}
    
    with open(centroid_path, 'r') as f:
        centroids = json.load(f)
    
    return centroids


def load_uncertainty(uncertainty_dir: Path, study_id: str) -> pd.DataFrame:
    """Load uncertainty sampled at centroids"""
    uncertainty_path = uncertainty_dir / "centroid_uncertainty" / f"{study_id}_uncertainty_at_centroids.csv"
    
    if not uncertainty_path.exists():
        logger.warning(f"Uncertainty not found: {uncertainty_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(uncertainty_path)
    return df


def create_overlay_visualization(
    dicom_slice: np.ndarray,
    seg_slice: np.ndarray,
    centroids: dict,
    uncertainty_df: pd.DataFrame,
    output_path: Path
):
    """
    Create comprehensive visualization with all overlays
    
    Layout:
    - Background: DICOM image (grayscale)
    - Layer 1: SPINEPS masks (semi-transparent colored regions)
    - Layer 2: Centroids (X markers)
    - Layer 3: Uncertainty values (colored text: green/yellow/red)
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    
    # Background: DICOM
    ax.imshow(dicom_slice.T, cmap='gray', aspect='auto', origin='lower')
    
    # Layer 1: SPINEPS masks (semi-transparent)
    # Create colored mask overlay
    mask_overlay = np.zeros((*seg_slice.shape, 4))  # RGBA
    
    # Color map for instances
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    unique_instances = np.unique(seg_slice)
    unique_instances = unique_instances[unique_instances > 0]  # Exclude background
    
    for instance_id in unique_instances:
        mask = (seg_slice == instance_id)
        color_idx = (instance_id - 1) % 10
        mask_overlay[mask, :3] = colors[color_idx, :3]
        mask_overlay[mask, 3] = 0.3  # 30% opacity
    
    ax.imshow(mask_overlay.transpose(1, 0, 2), aspect='auto', origin='lower')
    
    # Layer 2 & 3: Centroids + Uncertainty
    if not uncertainty_df.empty:
        for _, row in uncertainty_df.iterrows():
            # Get centroid coordinates
            x = row['centroid_voxel_j']  # Anterior-posterior
            y = row['centroid_voxel_i']  # Superior-inferior
            
            entropy = row['entropy']
            instance_id = row['instance_id']
            most_likely_level = row['most_likely_level']
            
            # Color based on entropy
            if entropy > 5.5:
                color = 'red'
                marker_size = 150
            elif entropy > 4.5:
                color = 'orange'
                marker_size = 120
            else:
                color = 'lime'
                marker_size = 100
            
            # Plot centroid marker
            ax.scatter(x, y, c=color, marker='x', s=marker_size, 
                      linewidths=3, edgecolors='white', zorder=10)
            
            # Add text label
            label = f"V{instance_id}\n{most_likely_level}\nH={entropy:.2f}"
            ax.text(x + 15, y, label, 
                   fontsize=9, color=color, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor=color),
                   zorder=11)
    
    # Title and labels
    ax.set_title(f"LSTV Fusion Visualization\nStudy: {study_id}", 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Anterior ← → Posterior', fontsize=12)
    ax.set_ylabel('Inferior ← → Superior', fontsize=12)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='lime', label='Low Uncertainty (H < 4.5)'),
        mpatches.Patch(color='orange', label='Medium Uncertainty (4.5 < H < 5.5)'),
        mpatches.Patch(color='red', label='High Uncertainty (H > 5.5) - LSTV?'),
        mpatches.Patch(facecolor='blue', alpha=0.3, label='SPINEPS Segmentation Masks'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add entropy distribution subplot
    if not uncertainty_df.empty:
        # Inset axis for entropy histogram
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, width="25%", height="20%", loc='lower left',
                          bbox_to_anchor=(0.05, 0.05, 1, 1), bbox_transform=ax.transAxes)
        
        entropies = uncertainty_df['entropy'].values
        axins.hist(entropies, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
        axins.axvline(5.5, color='red', linestyle='--', linewidth=2, label='LSTV Threshold')
        axins.set_xlabel('Entropy', fontsize=8)
        axins.set_ylabel('Count', fontsize=8)
        axins.set_title('Uncertainty Distribution', fontsize=9)
        axins.tick_params(labelsize=7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved visualization: {output_path}")


def create_multi_view_visualization(
    study_id: str,
    dicom_dir: Path,
    spineps_dir: Path,
    uncertainty_dir: Path,
    output_path: Path,
    series_id: str = None
):
    """
    Create comprehensive multi-panel visualization
    
    Panels:
    1. DICOM + SPINEPS masks
    2. DICOM + Centroids + Uncertainty
    3. Entropy bar chart
    4. Summary statistics
    """
    
    logger.info(f"Creating visualization for study {study_id}...")
    
    # Load data
    study_path = dicom_dir / str(study_id)
    
    # Find sagittal T2 series if not provided
    if series_id is None:
        # Simple: use first directory
        series_dirs = [d for d in study_path.iterdir() if d.is_dir()]
        if not series_dirs:
            raise FileNotFoundError(f"No series found in {study_path}")
        series_id = series_dirs[0].name
    
    dicom_slice, dicom_volume, dicom_mid_idx = load_dicom_study(study_path, series_id)
    seg_slice, seg_volume, seg_affine = load_spineps_segmentation(spineps_dir, study_id)
    centroids = load_centroids(spineps_dir, study_id)
    uncertainty_df = load_uncertainty(uncertainty_dir, study_id)
    
    # Resize segmentation to match DICOM if needed
    if dicom_slice.shape != seg_slice.shape:
        logger.info(f"Resizing segmentation: {seg_slice.shape} → {dicom_slice.shape}")
        seg_slice = cv2.resize(seg_slice.astype(np.float32), 
                              (dicom_slice.shape[1], dicom_slice.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(int)
    
    # Create 2x2 subplot
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: DICOM + SPINEPS masks
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(dicom_slice.T, cmap='gray', aspect='auto', origin='lower')
    
    # Overlay masks
    mask_overlay = np.zeros((*seg_slice.shape, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    unique_instances = np.unique(seg_slice)
    unique_instances = unique_instances[unique_instances > 0]
    
    for instance_id in unique_instances:
        mask = (seg_slice == instance_id)
        color_idx = (instance_id - 1) % 10
        mask_overlay[mask, :3] = colors[color_idx, :3]
        mask_overlay[mask, 3] = 0.4
    
    ax1.imshow(mask_overlay.transpose(1, 0, 2), aspect='auto', origin='lower')
    ax1.set_title('Panel A: DICOM + SPINEPS Segmentation', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Anterior ← → Posterior')
    ax1.set_ylabel('Inferior ← → Superior')
    
    # Panel 2: DICOM + Centroids + Uncertainty
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(dicom_slice.T, cmap='gray', aspect='auto', origin='lower')
    
    if not uncertainty_df.empty:
        for _, row in uncertainty_df.iterrows():
            x = row['centroid_voxel_j']
            y = row['centroid_voxel_i']
            entropy = row['entropy']
            instance_id = row['instance_id']
            
            # Color based on entropy
            if entropy > 5.5:
                color, marker_size = 'red', 150
            elif entropy > 4.5:
                color, marker_size = 'orange', 120
            else:
                color, marker_size = 'lime', 100
            
            ax2.scatter(x, y, c=color, marker='x', s=marker_size, 
                       linewidths=3, edgecolors='white', zorder=10)
            ax2.text(x + 10, y, f"V{instance_id}\nH={entropy:.2f}", 
                    fontsize=8, color=color, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                    zorder=11)
    
    ax2.set_title('Panel B: Centroids + Uncertainty Values', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Anterior ← → Posterior')
    ax2.set_ylabel('Inferior ← → Superior')
    
    # Panel 3: Entropy bar chart
    ax3 = fig.add_subplot(gs[1, 0])
    
    if not uncertainty_df.empty:
        sorted_df = uncertainty_df.sort_values('instance_id')
        
        vertebrae = [f"V{row['instance_id']}\n{row['most_likely_level']}" 
                    for _, row in sorted_df.iterrows()]
        entropies = sorted_df['entropy'].values
        
        colors_bar = ['red' if e > 5.5 else 'orange' if e > 4.5 else 'lime' 
                     for e in entropies]
        
        bars = ax3.bar(range(len(vertebrae)), entropies, color=colors_bar, 
                       edgecolor='black', linewidth=1.5)
        ax3.axhline(y=5.5, color='red', linestyle='--', linewidth=2, label='LSTV Threshold (5.5)')
        ax3.axhline(y=4.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax3.set_xticks(range(len(vertebrae)))
        ax3.set_xticklabels(vertebrae, fontsize=9)
        ax3.set_ylabel('Shannon Entropy', fontsize=11, fontweight='bold')
        ax3.set_title('Panel C: Uncertainty by Vertebra', fontweight='bold', fontsize=12)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
    
    # Panel 4: Summary statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    summary_text = f"Study ID: {study_id}\n"
    summary_text += f"Series ID: {series_id}\n\n"
    summary_text += "=" * 40 + "\n"
    summary_text += "SUMMARY STATISTICS\n"
    summary_text += "=" * 40 + "\n\n"
    
    if not uncertainty_df.empty:
        summary_text += f"Number of vertebrae: {len(uncertainty_df)}\n\n"
        
        summary_text += f"Entropy Statistics:\n"
        summary_text += f"  Mean: {uncertainty_df['entropy'].mean():.3f}\n"
        summary_text += f"  Std:  {uncertainty_df['entropy'].std():.3f}\n"
        summary_text += f"  Min:  {uncertainty_df['entropy'].min():.3f}\n"
        summary_text += f"  Max:  {uncertainty_df['entropy'].max():.3f}\n\n"
        
        high_entropy = uncertainty_df[uncertainty_df['entropy'] > 5.5]
        summary_text += f"High Uncertainty (H > 5.5):\n"
        summary_text += f"  Count: {len(high_entropy)}\n"
        
        if len(high_entropy) > 0:
            summary_text += f"  LSTV DETECTED! ⚠️\n\n"
            summary_text += "  High uncertainty vertebrae:\n"
            for _, row in high_entropy.iterrows():
                summary_text += f"    • V{row['instance_id']} ({row['most_likely_level']}): H={row['entropy']:.3f}\n"
        else:
            summary_text += f"  Normal anatomy (no LSTV) ✓\n"
        
        summary_text += "\n" + "=" * 40 + "\n"
        summary_text += "INTERPRETATION\n"
        summary_text += "=" * 40 + "\n\n"
        
        if len(high_entropy) > 0:
            summary_text += "⚠️ High uncertainty detected!\n\n"
            summary_text += "This suggests possible LSTV\n"
            summary_text += "(Lumbosacral Transitional Vertebra)\n\n"
            summary_text += "Recommended action:\n"
            summary_text += "  → Manual review by radiologist\n"
            summary_text += "  → Verify vertebral count\n"
            summary_text += "  → Check for sacralization/lumbarization\n"
        else:
            summary_text += "✓ All entropy values below threshold\n\n"
            summary_text += "This suggests normal anatomy\n"
            summary_text += "with standard 5 lumbar vertebrae.\n"
    else:
        summary_text += "No uncertainty data available.\n"
    
    ax4.text(0.05, 0.95, summary_text, 
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Overall title
    fig.suptitle(f'LSTV Fusion Visualization - Study {study_id}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved multi-panel visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified LSTV Visualization - See Everything in One View'
    )
    
    parser.add_argument('--study_id', type=str, required=True,
                       help='Study ID to visualize')
    parser.add_argument('--dicom_dir', type=str, required=True,
                       help='DICOM directory (train_images)')
    parser.add_argument('--spineps_dir', type=str, required=True,
                       help='SPINEPS results directory')
    parser.add_argument('--uncertainty_dir', type=str, required=True,
                       help='Uncertainty detection output directory')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--series_id', type=str, default=None,
                       help='Specific series ID (optional, auto-detect if not provided)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualization
    output_path = output_dir / f"{args.study_id}_fusion_visualization.png"
    
    try:
        create_multi_view_visualization(
            study_id=args.study_id,
            dicom_dir=Path(args.dicom_dir),
            spineps_dir=Path(args.spineps_dir),
            uncertainty_dir=Path(args.uncertainty_dir),
            output_path=output_path,
            series_id=args.series_id
        )
        
        logger.info("="*60)
        logger.info("✓ VISUALIZATION COMPLETE!")
        logger.info("="*60)
        logger.info(f"Output: {output_path}")
        logger.info("")
        logger.info("The visualization shows:")
        logger.info("  Panel A: DICOM + SPINEPS segmentation masks")
        logger.info("  Panel B: Centroids with color-coded uncertainty")
        logger.info("  Panel C: Entropy bar chart")
        logger.info("  Panel D: Summary statistics and interpretation")
        logger.info("")
        logger.info("To view:")
        logger.info(f"  open {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
