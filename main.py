"""
Main Execution Script
Cybersecurity Intrusion Detection Optimization using GA + Machine Learning

This script orchestrates the entire pipeline:
1. Data preprocessing
2. Baseline ML model training
3. GA-based feature selection
4. Final model training with selected features
5. Evaluation and comparison
6. Visualization and reporting

Run with: python main.py
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from src.preprocessing import CICIDS2017Preprocessor
from src.models import BaselineModels
from src.ga import GeneticFeatureSelector
from src.evaluate import ModelEvaluator


def print_header(text):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def main():
    """Main execution pipeline"""
    
    print_header("CYBERSECURITY INTRUSION DETECTION OPTIMIZATION")
    print("Using Genetic Algorithm + Machine Learning")
    print("Dataset: CICIDS2017")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_start = time.time()
    
    # =========================================================================
    # STEP 1: DATA PREPROCESSING
    # =========================================================================
    print_header("STEP 1: DATA PREPROCESSING")
    
    preprocessor = CICIDS2017Preprocessor(data_dir='data/')
    
    # For full dataset, remove max_rows_per_file parameter
    # For testing, use max_rows_per_file=50000
    processed_data = preprocessor.preprocess_pipeline(
        max_rows_per_file=50000  # Remove this line for full dataset
    )
    
    X_train = processed_data['X_train']
    X_val = processed_data['X_val']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_val = processed_data['y_val']
    y_test = processed_data['y_test']
    feature_names = processed_data['feature_names']
    
    print(f"\nPreprocessing complete!")
    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Number of features: {len(feature_names)}")
    
    # =========================================================================
    # STEP 2: BASELINE MODEL TRAINING
    # =========================================================================
    print_header("STEP 2: BASELINE MODEL TRAINING")
    
    baseline_models = BaselineModels()
    
    # Train Random Forest (primary baseline)
    print("\nTraining baseline Random Forest...")
    rf_baseline = baseline_models.train_random_forest(
        X_train, y_train,
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
    
    # Evaluate baseline on validation set
    evaluator = ModelEvaluator()
    baseline_val_metrics = evaluator.evaluate_model(
        rf_baseline, X_val, y_val,
        model_name='Baseline RF',
        dataset_name='Validation'
    )
    
    # Save baseline model
    baseline_models.save_model('random_forest', 'models/baseline_rf.pkl')
    
    print("\nBaseline training complete!")
    
    # =========================================================================
    # STEP 3: GENETIC ALGORITHM FEATURE SELECTION
    # =========================================================================
    print_header("STEP 3: GENETIC ALGORITHM FEATURE SELECTION")
    
    print("Initializing Genetic Algorithm...")
    print("\nGA Configuration:")
    print("  Chromosome: Binary vector (1=selected, 0=not selected)")
    print("  Fitness: α*F1_score - β*(k/d)")
    print("  Constraints: k_min <= selected_features <= k_max")
    
    ga_selector = GeneticFeatureSelector(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        alpha=1.0,           # Weight for F1 score
        beta=0.05,           # Weight for feature reduction
        k_min=10,            # Minimum features
        k_max=None,          # Maximum features (None = no limit)
        pop_size=30,         # Population size
        # pop_size=50, 
        n_generations=10,    # Number of generations
        # n_generations=30,
        cx_prob=0.7,         # Crossover probability
        mut_prob=0.2,        # Mutation probability
        tournament_size=3,   # Tournament selection size
        random_state=42
    )
    
    # Run GA
    best_individual, best_fitness = ga_selector.run(verbose=True)
    
    # Get selected features
    selected_mask = ga_selector.get_feature_mask()
    selected_features = ga_selector.get_selected_features(feature_names)
    
    print(f"\nGA Feature Selection Complete!")
    print(f"Selected {len(selected_features)} features out of {len(feature_names)}")
    print(f"Feature reduction: {(1 - len(selected_features)/len(feature_names))*100:.1f}%")
    
    # Plot GA results
    print("\nGenerating GA visualizations...")
    ga_selector.plot_fitness_evolution(save_path='reports/figures/ga_fitness_evolution.png')
    ga_selector.plot_generation_times(save_path='reports/figures/ga_generation_times.png')
    
    # Save selected features
    import pandas as pd
    selected_features_df = pd.DataFrame({
        'feature': selected_features,
        'index': [feature_names.index(f) for f in selected_features]
    })
    selected_features_df.to_csv('reports/results/selected_features.csv', index=False)
    print("Selected features saved to reports/results/selected_features.csv")
    
    # =========================================================================
    # STEP 4: FINAL MODEL TRAINING WITH SELECTED FEATURES
    # =========================================================================
    print_header("STEP 4: FINAL MODEL WITH GA-SELECTED FEATURES")
    
    # Apply feature selection to all datasets
    X_train_selected = X_train[:, selected_mask]
    X_val_selected = X_val[:, selected_mask]
    X_test_selected = X_test[:, selected_mask]
    
    print(f"Training Random Forest with {len(selected_features)} selected features...")
    
    # Train final model
    from sklearn.ensemble import RandomForestClassifier
    rf_optimized = RandomForestClassifier(
        n_estimators=300,    # More trees for final model
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_optimized.fit(X_train_selected, y_train)
    
    print("Final model training complete!")
    
    # =========================================================================
    # STEP 5: COMPREHENSIVE EVALUATION
    # =========================================================================
    print_header("STEP 5: COMPREHENSIVE EVALUATION")
    
    # Evaluate GA-optimized model on validation set
    ga_val_metrics = evaluator.evaluate_model(
        rf_optimized, X_val_selected, y_val,
        model_name='GA-Optimized RF',
        dataset_name='Validation'
    )
    
    # Evaluate both models on test set
    print("\n" + "-"*80)
    print("FINAL TEST SET EVALUATION")
    print("-"*80)
    
    baseline_test_metrics = evaluator.evaluate_model(
        rf_baseline, X_test, y_test,
        model_name='Baseline RF',
        dataset_name='Test'
    )
    
    ga_test_metrics = evaluator.evaluate_model(
        rf_optimized, X_test_selected, y_test,
        model_name='GA-Optimized RF',
        dataset_name='Test'
    )
    
    # =========================================================================
    # STEP 6: COMPARISON AND VISUALIZATION
    # =========================================================================
    print_header("STEP 6: RESULTS COMPARISON AND VISUALIZATION")
    
    # Compare models
    comparison_df = evaluator.compare_models()
    
    # Save comparison
    comparison_df.to_csv('reports/results/model_comparison.csv', index=False)
    print("\nComparison saved to reports/results/model_comparison.csv")
    
    # Generate visualizations
    print("\nGenerating comparison visualizations...")
    
    # Confusion matrices
    evaluator.plot_confusion_matrices(
        save_path='reports/figures/confusion_matrices.png'
    )
    
    # ROC curves
    evaluator.plot_roc_curves(
        save_path='reports/figures/roc_curves.png'
    )
    
    # Metrics comparison
    evaluator.plot_metrics_comparison(
        save_path='reports/figures/metrics_comparison.png'
    )
    
    # Generate text report
    evaluator.generate_report(
        output_file='reports/results/evaluation_report.txt'
    )
    
    # Save results as JSON
    evaluator.save_results_json(
        output_file='reports/results/results.json'
    )
    
    # =========================================================================
    # STEP 7: SUMMARY AND INSIGHTS
    # =========================================================================
    print_header("STEP 7: SUMMARY AND INSIGHTS")
    
    print("Key Findings:")
    print("-" * 80)
    
    # Performance improvement
    baseline_f1 = baseline_test_metrics['f1_score']
    ga_f1 = ga_test_metrics['f1_score']
    f1_improvement = ((ga_f1 - baseline_f1) / baseline_f1) * 100
    
    print(f"\n1. Model Performance:")
    print(f"   Baseline F1-Score:     {baseline_f1:.4f}")
    print(f"   GA-Optimized F1-Score: {ga_f1:.4f}")
    print(f"   Improvement:           {f1_improvement:+.2f}%")
    
    # Feature reduction
    feature_reduction = (1 - len(selected_features)/len(feature_names)) * 100
    print(f"\n2. Feature Reduction:")
    print(f"   Original features:     {len(feature_names)}")
    print(f"   Selected features:     {len(selected_features)}")
    print(f"   Reduction:             {feature_reduction:.1f}%")
    
    # False positive rate
    baseline_fpr = baseline_test_metrics['fpr']
    ga_fpr = ga_test_metrics['fpr']
    fpr_improvement = ((baseline_fpr - ga_fpr) / baseline_fpr) * 100
    
    print(f"\n3. False Positive Rate (Critical for Cybersecurity):")
    print(f"   Baseline FPR:          {baseline_fpr:.4f}")
    print(f"   GA-Optimized FPR:      {ga_fpr:.4f}")
    print(f"   Improvement:           {fpr_improvement:+.2f}%")
    
    # Inference speed
    baseline_time = baseline_test_metrics['inference_time']
    ga_time = ga_test_metrics['inference_time']
    time_improvement = ((baseline_time - ga_time) / baseline_time) * 100
    
    print(f"\n4. Inference Speed:")
    print(f"   Baseline time:         {baseline_time:.4f}s")
    print(f"   GA-Optimized time:     {ga_time:.4f}s")
    print(f"   Improvement:           {time_improvement:+.2f}%")
    
    # Total execution time
    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nAll results saved to 'reports/' directory:")
    print("  - Figures: reports/figures/")
    print("  - Results: reports/results/")
    print("  - Models: models/")
    print("\nYou can now review the results and prepare your coursework report.")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()