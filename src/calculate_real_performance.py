#!/usr/bin/env python3
"""
Calculate Real Performance Metrics from Clinical Audit
=======================================================

This script ingests the audit results from lstv-annotation-tool
and generates comprehensive performance metrics including:
- Confusion matrix
- Sensitivity/Specificity
- PPV/NPV
- ROC curve
- Stratified performance by confidence level

Author: Claude (Anthropic) - Generated for Dr. Gregory Schwing
Date: February 2026
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc,
    classification_report,
    precision_recall_curve
)
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMetricsCalculator:
    """
    Calculate comprehensive performance metrics from audit results
    """
    
    def __init__(
        self,
        fusion_results_csv: Path,
        audit_results_json: Path,
        output_dir: Path
    ):
        """
        Parameters:
        -----------
        fusion_results_csv : Path
            fusion_summary.csv from fuse_consensus.py
        audit_results_json : Path
            audit_results.json from lstv-annotation-tool clinical audit
        output_dir : Path
            Directory to save performance reports
        """
        self.fusion_results = pd.read_csv(fusion_results_csv)
        
        with open(audit_results_json, 'r') as f:
            self.audit_results = json.load(f)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Merge fusion predictions with audit ground truth
        self.merged_data = self._merge_predictions_and_truth()
    
    def _merge_predictions_and_truth(self) -> pd.DataFrame:
        """
        Merge AI predictions with radiologist ground truth
        
        Returns:
        --------
        pd.DataFrame with columns:
            - study_id
            - predicted_lstv (bool)
            - predicted_type (str)
            - confidence_score (float)
            - max_entropy (float)
            - ground_truth_lstv (bool)
            - ground_truth_type (str)
            - radiologist_notes (str)
        """
        merged = []
        
        for audit in self.audit_results['audits']:
            study_id = audit['study_id']
            
            # Find corresponding fusion result
            fusion_row = self.fusion_results[
                self.fusion_results['study_id'] == study_id
            ]
            
            if fusion_row.empty:
                logger.warning(f"Study {study_id} not found in fusion results")
                continue
            
            merged.append({
                'study_id': study_id,
                'predicted_lstv': fusion_row['lstv_detected'].values[0],
                'predicted_type': fusion_row['lstv_type'].values[0],
                'confidence_score': fusion_row['confidence_score'].values[0],
                'max_entropy': fusion_row['max_entropy'].values[0],
                'ground_truth_lstv': audit['lstv_present'],
                'ground_truth_type': audit.get('lstv_type', None),
                'radiologist_notes': audit.get('notes', ''),
                'radiologist_id': audit.get('radiologist', 'unknown')
            })
        
        df = pd.DataFrame(merged)
        logger.info(f"Merged {len(df)} studies with audit ground truth")
        return df
    
    def calculate_all_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Returns:
        --------
        Dict with all performance metrics
        """
        metrics = {}
        
        # Basic confusion matrix metrics
        metrics['confusion_matrix'] = self._calculate_confusion_matrix()
        metrics['sensitivity_specificity'] = self._calculate_sensitivity_specificity()
        metrics['ppv_npv'] = self._calculate_ppv_npv()
        
        # ROC analysis
        metrics['roc'] = self._calculate_roc_curve()
        
        # Stratified by confidence
        metrics['stratified_by_confidence'] = self._stratified_performance()
        
        # Stratified by LSTV type
        metrics['stratified_by_type'] = self._performance_by_lstv_type()
        
        # Inter-rater reliability (if multiple radiologists)
        if self._has_multiple_radiologists():
            metrics['inter_rater'] = self._calculate_inter_rater_reliability()
        
        # Save all metrics
        self._save_metrics(metrics)
        
        # Generate visualizations
        self._generate_visualizations(metrics)
        
        return metrics
    
    def _calculate_confusion_matrix(self) -> Dict:
        """
        Calculate confusion matrix for LSTV detection
        
        Returns:
        --------
        Dict with:
            - matrix (2x2 array)
            - TP, TN, FP, FN counts
        """
        y_true = self.merged_data['ground_truth_lstv'].astype(int)
        y_pred = self.merged_data['predicted_lstv'].astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Extract counts
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'matrix': cm.tolist(),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
            'total': len(y_true)
        }
    
    def _calculate_sensitivity_specificity(self) -> Dict:
        """
        Calculate sensitivity and specificity
        
        Sensitivity (Recall) = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        """
        cm = self._calculate_confusion_matrix()
        
        tp = cm['true_positive']
        tn = cm['true_negative']
        fp = cm['false_positive']
        fn = cm['false_negative']
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 95% Confidence intervals (Wilson score)
        sensitivity_ci = self._wilson_confidence_interval(tp, tp + fn)
        specificity_ci = self._wilson_confidence_interval(tn, tn + fp)
        
        return {
            'sensitivity': sensitivity,
            'sensitivity_95ci': sensitivity_ci,
            'specificity': specificity,
            'specificity_95ci': specificity_ci,
            'n_positive': tp + fn,
            'n_negative': tn + fp
        }
    
    def _calculate_ppv_npv(self) -> Dict:
        """
        Calculate Positive and Negative Predictive Values
        
        PPV (Precision) = TP / (TP + FP)
        NPV = TN / (TN + FN)
        """
        cm = self._calculate_confusion_matrix()
        
        tp = cm['true_positive']
        tn = cm['true_negative']
        fp = cm['false_positive']
        fn = cm['false_negative']
        
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        ppv_ci = self._wilson_confidence_interval(tp, tp + fp)
        npv_ci = self._wilson_confidence_interval(tn, tn + fn)
        
        return {
            'ppv': ppv,
            'ppv_95ci': ppv_ci,
            'npv': npv,
            'npv_95ci': npv_ci,
            'n_predicted_positive': tp + fp,
            'n_predicted_negative': tn + fn
        }
    
    @staticmethod
    def _wilson_confidence_interval(successes: int, total: int, alpha: float = 0.05) -> Tuple[float, float]:
        """
        Calculate Wilson score confidence interval
        
        More accurate than normal approximation for small sample sizes
        """
        if total == 0:
            return (0.0, 0.0)
        
        z = 1.96  # 95% CI
        p = successes / total
        
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt((p * (1 - p) / total + z**2 / (4 * total**2))) / denominator
        
        lower = max(0, center - margin)
        upper = min(1, center + margin)
        
        return (lower, upper)
    
    def _calculate_roc_curve(self) -> Dict:
        """
        Calculate ROC curve using entropy as continuous predictor
        
        Returns:
        --------
        Dict with:
            - fpr, tpr, thresholds
            - auc
            - optimal_threshold
        """
        y_true = self.merged_data['ground_truth_lstv'].astype(int)
        y_score = self.merged_data['max_entropy']
        
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(roc_auc),
            'optimal_threshold': float(optimal_threshold),
            'optimal_sensitivity': float(tpr[optimal_idx]),
            'optimal_specificity': float(1 - fpr[optimal_idx])
        }
    
    def _stratified_performance(self) -> Dict:
        """
        Calculate performance stratified by confidence score
        
        Categories:
        - HIGH (≥0.7)
        - MEDIUM (0.4-0.7)
        - LOW (<0.4)
        """
        results = {}
        
        confidence_bins = [
            ('high', lambda x: x >= 0.7),
            ('medium', lambda x: 0.4 <= x < 0.7),
            ('low', lambda x: x < 0.4)
        ]
        
        for bin_name, bin_filter in confidence_bins:
            subset = self.merged_data[self.merged_data['confidence_score'].apply(bin_filter)]
            
            if len(subset) == 0:
                continue
            
            y_true = subset['ground_truth_lstv'].astype(int)
            y_pred = subset['predicted_lstv'].astype(int)
            
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            results[bin_name] = {
                'n_studies': len(subset),
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            }
        
        return results
    
    def _performance_by_lstv_type(self) -> Dict:
        """
        Calculate performance separately for:
        - Sacralization
        - Lumbarization
        """
        results = {}
        
        lstv_types = ['sacralization', 'lumbarization']
        
        for lstv_type in lstv_types:
            # Filter to cases where ground truth is this type
            subset = self.merged_data[
                self.merged_data['ground_truth_type'] == lstv_type
            ]
            
            if len(subset) == 0:
                continue
            
            # Calculate how many were correctly predicted as LSTV
            correct_detection = subset['predicted_lstv'].sum()
            
            # Calculate how many were correctly classified as this specific type
            correct_type = (subset['predicted_type'] == lstv_type).sum()
            
            results[lstv_type] = {
                'n_cases': len(subset),
                'detection_rate': correct_detection / len(subset),
                'type_classification_accuracy': correct_type / len(subset)
            }
        
        return results
    
    def _has_multiple_radiologists(self) -> bool:
        """Check if multiple radiologists performed audits"""
        radiologists = self.merged_data['radiologist_id'].unique()
        return len(radiologists) > 1
    
    def _calculate_inter_rater_reliability(self) -> Dict:
        """
        Calculate inter-rater reliability (Cohen's Kappa) if multiple radiologists
        
        Note: This requires overlapping cases reviewed by multiple radiologists
        """
        # Implementation would require overlapping audit data
        # For now, return placeholder
        return {
            'note': 'Inter-rater reliability calculation requires overlapping audits',
            'implemented': False
        }
    
    def _save_metrics(self, metrics: Dict):
        """Save all metrics to JSON file"""
        output_path = self.output_dir / 'performance_metrics.json'
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {output_path}")
        
        # Also save human-readable summary
        self._save_summary_report(metrics)
    
    def _save_summary_report(self, metrics: Dict):
        """Generate human-readable summary report"""
        output_path = self.output_dir / 'summary_report.txt'
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("LSTV DETECTION SYSTEM - PERFORMANCE METRICS\n")
            f.write("=" * 80 + "\n\n")
            
            # Confusion Matrix
            cm = metrics['confusion_matrix']
            f.write("CONFUSION MATRIX:\n")
            f.write(f"  True Negative:  {cm['true_negative']}\n")
            f.write(f"  False Positive: {cm['false_positive']}\n")
            f.write(f"  False Negative: {cm['false_negative']}\n")
            f.write(f"  True Positive:  {cm['true_positive']}\n")
            f.write(f"  Total Studies:  {cm['total']}\n\n")
            
            # Sensitivity/Specificity
            ss = metrics['sensitivity_specificity']
            f.write("SENSITIVITY & SPECIFICITY:\n")
            f.write(f"  Sensitivity: {ss['sensitivity']:.1%} "
                   f"(95% CI: {ss['sensitivity_95ci'][0]:.1%}-{ss['sensitivity_95ci'][1]:.1%})\n")
            f.write(f"  Specificity: {ss['specificity']:.1%} "
                   f"(95% CI: {ss['specificity_95ci'][0]:.1%}-{ss['specificity_95ci'][1]:.1%})\n\n")
            
            # PPV/NPV
            pv = metrics['ppv_npv']
            f.write("PREDICTIVE VALUES:\n")
            f.write(f"  PPV: {pv['ppv']:.1%} "
                   f"(95% CI: {pv['ppv_95ci'][0]:.1%}-{pv['ppv_95ci'][1]:.1%})\n")
            f.write(f"  NPV: {pv['npv']:.1%} "
                   f"(95% CI: {pv['npv_95ci'][0]:.1%}-{pv['npv_95ci'][1]:.1%})\n\n")
            
            # ROC
            roc = metrics['roc']
            f.write("ROC ANALYSIS:\n")
            f.write(f"  AUC: {roc['auc']:.3f}\n")
            f.write(f"  Optimal Threshold: {roc['optimal_threshold']:.2f}\n")
            f.write(f"  At Optimal: Sens={roc['optimal_sensitivity']:.1%}, "
                   f"Spec={roc['optimal_specificity']:.1%}\n\n")
            
            # Stratified by confidence
            f.write("PERFORMANCE BY CONFIDENCE LEVEL:\n")
            for level, stats in metrics['stratified_by_confidence'].items():
                f.write(f"  {level.upper()}: n={stats['n_studies']}, "
                       f"Sens={stats['sensitivity']:.1%}, "
                       f"Spec={stats['specificity']:.1%}, "
                       f"PPV={stats['ppv']:.1%}\n")
        
        logger.info(f"Saved summary report to {output_path}")
    
    def _generate_visualizations(self, metrics: Dict):
        """Generate all visualization plots"""
        
        # 1. Confusion Matrix Heatmap
        self._plot_confusion_matrix(metrics['confusion_matrix'])
        
        # 2. ROC Curve
        self._plot_roc_curve(metrics['roc'])
        
        # 3. Entropy Distribution (LSTV vs Normal)
        self._plot_entropy_distribution()
        
        # 4. Performance by Confidence
        self._plot_stratified_performance(metrics['stratified_by_confidence'])
    
    def _plot_confusion_matrix(self, cm_data: Dict):
        """Plot confusion matrix heatmap"""
        cm = np.array([[cm_data['true_negative'], cm_data['false_positive']],
                       [cm_data['false_negative'], cm_data['true_positive']]])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted LSTV'],
            yticklabels=['Ground Truth Normal', 'Ground Truth LSTV']
        )
        plt.title('Confusion Matrix: LSTV Detection')
        plt.ylabel('Ground Truth')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        output_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {output_path}")
    
    def _plot_roc_curve(self, roc_data: Dict):
        """Plot ROC curve"""
        plt.figure(figsize=(8, 8))
        plt.plot(
            roc_data['fpr'], 
            roc_data['tpr'], 
            color='darkorange',
            lw=2,
            label=f"ROC curve (AUC = {roc_data['auc']:.3f})"
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        # Mark optimal threshold
        plt.scatter(
            [1 - roc_data['optimal_specificity']], 
            [roc_data['optimal_sensitivity']],
            color='red',
            s=100,
            label=f"Optimal (threshold={roc_data['optimal_threshold']:.2f})"
        )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curve: LSTV Detection Using Entropy')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / 'roc_curve.png'
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved ROC curve to {output_path}")
    
    def _plot_entropy_distribution(self):
        """Plot entropy distribution for LSTV vs Normal cases"""
        lstv_cases = self.merged_data[self.merged_data['ground_truth_lstv']]
        normal_cases = self.merged_data[~self.merged_data['ground_truth_lstv']]
        
        plt.figure(figsize=(10, 6))
        
        plt.hist(
            normal_cases['max_entropy'], 
            bins=20, 
            alpha=0.5, 
            label='Normal Anatomy',
            color='blue'
        )
        plt.hist(
            lstv_cases['max_entropy'], 
            bins=20, 
            alpha=0.5, 
            label='LSTV Cases',
            color='red'
        )
        
        # Add threshold line
        plt.axvline(x=5.0, color='black', linestyle='--', label='Threshold (5.0)')
        
        plt.xlabel('Max Shannon Entropy')
        plt.ylabel('Frequency')
        plt.title('Entropy Distribution: LSTV vs Normal Anatomy')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / 'entropy_distribution.png'
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved entropy distribution plot to {output_path}")
    
    def _plot_stratified_performance(self, stratified_data: Dict):
        """Plot performance metrics by confidence level"""
        levels = []
        sensitivity = []
        specificity = []
        ppv = []
        
        for level in ['high', 'medium', 'low']:
            if level in stratified_data:
                levels.append(level.capitalize())
                sensitivity.append(stratified_data[level]['sensitivity'] * 100)
                specificity.append(stratified_data[level]['specificity'] * 100)
                ppv.append(stratified_data[level]['ppv'] * 100)
        
        x = np.arange(len(levels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, sensitivity, width, label='Sensitivity', color='#2ecc71')
        ax.bar(x, specificity, width, label='Specificity', color='#3498db')
        ax.bar(x + width, ppv, width, label='PPV', color='#e74c3c')
        
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Performance Metrics by Confidence Level')
        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'performance_by_confidence.png'
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved stratified performance plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate performance metrics from clinical audit results"
    )
    
    parser.add_argument(
        '--fusion-results',
        type=Path,
        required=True,
        help='Path to fusion_summary.csv from fuse_consensus.py'
    )
    parser.add_argument(
        '--audit-results',
        type=Path,
        required=True,
        help='Path to audit_results.json from lstv-annotation-tool'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for performance reports'
    )
    
    args = parser.parse_args()
    
    # Create calculator
    calculator = PerformanceMetricsCalculator(
        fusion_results_csv=args.fusion_results,
        audit_results_json=args.audit_results,
        output_dir=args.output_dir
    )
    
    # Calculate all metrics
    logger.info("=" * 80)
    logger.info("CALCULATING PERFORMANCE METRICS")
    logger.info("=" * 80)
    
    metrics = calculator.calculate_all_metrics()
    
    logger.info("\n" + "=" * 80)
    logger.info("METRICS CALCULATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"  - performance_metrics.json (full metrics)")
    logger.info(f"  - summary_report.txt (human-readable)")
    logger.info(f"  - *.png (visualization plots)")


if __name__ == "__main__":
    main()
