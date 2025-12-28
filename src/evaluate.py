"""
Evaluation and Reporting Module

Provides comprehensive evaluation metrics, comparison tools, and visualization
for baseline vs GA-optimized models.

Dataset: CICIDS2017
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)
import time
import json


class ModelEvaluator:
    """
    Comprehensive evaluation framework for intrusion detection models.
    """
    
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, model, X, y, model_name='Model', dataset_name='Test'):
        """
        Evaluate a model with comprehensive metrics.
        
        Args:
            model: Trained classifier
            X: Feature matrix
            y: True labels
            model_name: Name for identification
            dataset_name: Dataset identifier (e.g., 'Test', 'Validation')
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nEvaluating {model_name} on {dataset_name} set...")
        start_time = time.time()
        
        # Predictions
        y_pred = model.predict(X)
        
        # Probabilities (if available)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X)
        else:
            y_proba = None
        
        # Core metrics
        metrics = {
            'model_name': model_name,
            'dataset': dataset_name,
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'inference_time': time.time() - start_time,
            'n_features': X.shape[1]
        }
        
        # ROC-AUC if probabilities available
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y, y_proba)
            fpr, tpr, _ = roc_curve(y, y_proba)
            metrics['roc_curve'] = (fpr, tpr)
        else:
            metrics['roc_auc'] = None
            metrics['roc_curve'] = None
        
        # Confusion matrix breakdown
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        # Critical cybersecurity metrics
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        metrics['detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        
        # Store results
        self.results[f"{model_name}_{dataset_name}"] = metrics
        
        # Print summary
        self._print_metrics(metrics)
        
        return metrics
    
    def _print_metrics(self, metrics):
        """Print formatted metrics"""
        print(f"\n{'='*60}")
        print(f"Model: {metrics['model_name']} | Dataset: {metrics['dataset']}")
        print(f"{'='*60}")
        print(f"Features Used:     {metrics['n_features']}")
        print(f"Inference Time:    {metrics['inference_time']:.4f} seconds")
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:        {metrics['accuracy']:.4f}")
        print(f"  Precision:       {metrics['precision']:.4f}")
        print(f"  Recall:          {metrics['recall']:.4f}")
        print(f"  F1-Score:        {metrics['f1_score']:.4f}")
        if metrics['roc_auc'] is not None:
            print(f"  ROC-AUC:         {metrics['roc_auc']:.4f}")
        print(f"\nCybersecurity Metrics:")
        print(f"  Detection Rate:  {metrics['detection_rate']:.4f}")
        print(f"  False Pos. Rate: {metrics['fpr']:.4f}")
        print(f"  False Neg. Rate: {metrics['fnr']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"                   Predicted")
        print(f"                BENIGN  ATTACK")
        print(f"  Actual BENIGN   {metrics['true_negatives']:6d}  {metrics['false_positives']:6d}")
        print(f"         ATTACK   {metrics['false_negatives']:6d}  {metrics['true_positives']:6d}")
        print(f"{'='*60}")
    
    def compare_models(self, results_dict=None):
        """
        Compare multiple models side-by-side.
        
        Args:
            results_dict: Optional dict of results (uses stored results if None)
            
        Returns:
            DataFrame with comparison
        """
        if results_dict is None:
            results_dict = self.results
        
        if not results_dict:
            print("No results to compare")
            return None
        
        comparison_data = []
        for key, metrics in results_dict.items():
            comparison_data.append({
                'Model': metrics['model_name'],
                'Dataset': metrics['dataset'],
                'Features': metrics['n_features'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics.get('roc_auc', 'N/A'),
                'FPR': metrics['fpr'],
                'Detection Rate': metrics['detection_rate'],
                'Inference (s)': metrics['inference_time']
            })
        
        df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*100)
        print("MODEL COMPARISON")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100)
        
        return df
    
    def plot_confusion_matrices(self, results_dict=None, save_path=None):
        """
        Plot confusion matrices for multiple models.
        
        Args:
            results_dict: Optional dict of results
            save_path: Path to save figure
        """
        if results_dict is None:
            results_dict = self.results
        
        if not results_dict:
            print("No results to plot")
            return
        
        n_models = len(results_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (key, metrics) in zip(axes, results_dict.items()):
            cm = metrics['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['BENIGN', 'ATTACK'],
                       yticklabels=['BENIGN', 'ATTACK'],
                       cbar_kws={'label': 'Count'})
            
            ax.set_title(f"{metrics['model_name']}\n{metrics['dataset']} Set",
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=11)
            ax.set_xlabel('Predicted Label', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, results_dict=None, save_path=None):
        """
        Plot ROC curves for multiple models.
        
        Args:
            results_dict: Optional dict of results
            save_path: Path to save figure
        """
        if results_dict is None:
            results_dict = self.results
        
        plt.figure(figsize=(10, 8))
        
        for key, metrics in results_dict.items():
            if metrics['roc_curve'] is not None:
                fpr, tpr = metrics['roc_curve']
                auc = metrics['roc_auc']
                label = f"{metrics['model_name']} (AUC={auc:.4f})"
                plt.plot(fpr, tpr, linewidth=2, label=label)
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, results_dict=None, save_path=None):
        """
        Plot bar chart comparing key metrics across models.
        
        Args:
            results_dict: Optional dict of results
            save_path: Path to save figure
        """
        if results_dict is None:
            results_dict = self.results
        
        # Prepare data
        models = []
        metrics_data = {'Accuracy': [], 'Precision': [], 'Recall': [], 
                       'F1-Score': [], 'Detection Rate': []}
        
        for key, metrics in results_dict.items():
            models.append(f"{metrics['model_name']}\n({metrics['n_features']} features)")
            metrics_data['Accuracy'].append(metrics['accuracy'])
            metrics_data['Precision'].append(metrics['precision'])
            metrics_data['Recall'].append(metrics['recall'])
            metrics_data['F1-Score'].append(metrics['f1_score'])
            metrics_data['Detection Rate'].append(metrics['detection_rate'])
        
        # Create plot
        x = np.arange(len(models))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            offset = width * (i - 2)
            ax.bar(x + offset, values, width, label=metric_name, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.legend(fontsize=10, loc='lower right')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, output_file='reports/results/evaluation_report.txt'):
        """
        Generate a comprehensive text report.
        
        Args:
            output_file: Path to save report
        """
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CYBERSECURITY INTRUSION DETECTION - EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            for key, metrics in self.results.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"Model: {metrics['model_name']} | Dataset: {metrics['dataset']}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Features Used:     {metrics['n_features']}\n")
                f.write(f"Inference Time:    {metrics['inference_time']:.4f} seconds\n")
                f.write(f"\nPerformance Metrics:\n")
                f.write(f"  Accuracy:        {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision:       {metrics['precision']:.4f}\n")
                f.write(f"  Recall:          {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:        {metrics['f1_score']:.4f}\n")
                if metrics['roc_auc']:
                    f.write(f"  ROC-AUC:         {metrics['roc_auc']:.4f}\n")
                f.write(f"\nCybersecurity Metrics:\n")
                f.write(f"  Detection Rate:  {metrics['detection_rate']:.4f}\n")
                f.write(f"  False Pos. Rate: {metrics['fpr']:.4f}\n")
                f.write(f"  False Neg. Rate: {metrics['fnr']:.4f}\n")
                f.write(f"\nConfusion Matrix:\n")
                f.write(f"  TN: {metrics['true_negatives']:6d}  FP: {metrics['false_positives']:6d}\n")
                f.write(f"  FN: {metrics['false_negatives']:6d}  TP: {metrics['true_positives']:6d}\n")
                f.write("\n")
        
        print(f"\nReport saved to {output_file}")
    
    def save_results_json(self, output_file='reports/results/results.json'):
        """
        Save results to JSON file.
        
        Args:
            output_file: Path to save JSON
        """
        # Convert numpy types to native Python types
        results_serializable = {}
        for key, metrics in self.results.items():
            results_serializable[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else 
                   float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
                if k not in ['roc_curve', 'confusion_matrix']  # Skip complex objects
            }
        
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    print("Evaluation module loaded")
    print("Provides comprehensive metrics, visualizations, and reporting")