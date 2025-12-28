"""
Machine Learning Models Module

Implements baseline classifiers and evaluation metrics for intrusion detection.

Dataset: CICIDS2017
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import time
import pickle


class BaselineModels:
    """
    Collection of baseline ML models for intrusion detection.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_random_forest(self, X_train, y_train, n_estimators=200, max_depth=None, random_state=42):
        """
        Train Random Forest classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
            
        Returns:
            Trained model
        """
        print("\nTraining Random Forest...")
        start_time = time.time()
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state,
            class_weight='balanced'  # Handle class imbalance
        )
        
        rf.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        
        self.models['random_forest'] = rf
        return rf
    
    def train_logistic_regression(self, X_train, y_train, max_iter=1000, random_state=42):
        """
        Train Logistic Regression classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            max_iter: Maximum iterations
            random_state: Random seed
            
        Returns:
            Trained model
        """
        print("\nTraining Logistic Regression...")
        start_time = time.time()
        
        lr = LogisticRegression(
            max_iter=max_iter,
            n_jobs=-1,
            random_state=random_state,
            class_weight='balanced'
        )
        
        lr.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        
        self.models['logistic_regression'] = lr
        return lr
    
    def train_svm(self, X_train, y_train, kernel='rbf', random_state=42):
        """
        Train SVM classifier (use with caution on large datasets).
        
        Args:
            X_train: Training features
            y_train: Training labels
            kernel: Kernel type
            random_state: Random seed
            
        Returns:
            Trained model
        """
        print("\nTraining SVM...")
        print("Warning: SVM training may be slow on large datasets")
        start_time = time.time()
        
        svm = SVC(
            kernel=kernel,
            random_state=random_state,
            class_weight='balanced',
            probability=True  # For ROC-AUC
        )
        
        svm.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        
        self.models['svm'] = svm
        return svm
    
    def evaluate_model(self, model, X, y, model_name='Model'):
        """
        Evaluate model performance with comprehensive metrics.
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            model_name: Name for reporting
            
        Returns:
            Dictionary of metrics
        """
        print(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X)
        
        # Probabilities for ROC-AUC
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)[:, 1]
        else:
            y_proba = model.decision_function(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred, 
                                                          target_names=['BENIGN', 'ATTACK'],
                                                          digits=4)
        }
        
        # False Positive Rate (critical for cybersecurity)
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  FPR:       {metrics['fpr']:.4f}")
        print(f"  FNR:       {metrics['fnr']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  TN: {tn:6d}  FP: {fp:6d}")
        print(f"  FN: {fn:6d}  TP: {tp:6d}")
        
        return metrics
    
    def compare_models(self, X_train, y_train, X_val, y_val):
        """
        Train and compare multiple baseline models.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Dictionary of results for each model
        """
        results = {}
        
        # Random Forest (primary baseline)
        rf = self.train_random_forest(X_train, y_train)
        results['random_forest'] = self.evaluate_model(rf, X_val, y_val, 'Random Forest')
        
        # Logistic Regression (fast baseline)
        lr = self.train_logistic_regression(X_train, y_train)
        results['logistic_regression'] = self.evaluate_model(lr, X_val, y_val, 'Logistic Regression')
        
        self.results = results
        return results
    
    def save_model(self, model_name, filepath):
        """
        Save trained model to file.
        
        Args:
            model_name: Name of model to save
            filepath: Path to save file
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model


class LightweightClassifier:
    """
    Lightweight classifier for use in GA fitness evaluation.
    Uses Logistic Regression for speed.
    """
    
    def __init__(self, max_iter=1000, random_state=42):
        self.model = LogisticRegression(
            max_iter=max_iter,
            n_jobs=1,  # Single job for GA parallelization
            random_state=random_state,
            class_weight='balanced',
            solver='lbfgs'
        )
    
    def fit(self, X, y):
        """Train the model"""
        return self.model.fit(X, y)
    
    def predict(self, X):
        """Predict labels"""
        return self.model.predict(X)
    
    def score(self, X, y):
        """Return F1 score"""
        y_pred = self.predict(X)
        return f1_score(y, y_pred, zero_division=0)


def get_feature_importance(model, feature_names, top_n=20):
    """
    Get feature importance from trained model.
    
    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importance
    """
    import pandas as pd
    
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature importance")
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)


if __name__ == "__main__":
    # Example usage
    print("ML Models module loaded")
    print("Available models:")
    print("  - Random Forest")
    print("  - Logistic Regression")
    print("  - SVM (use with caution on large datasets)")

# import joblib
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score
# )


# class BaselineModels:
#     """
#     Baseline machine learning models for intrusion detection
#     """

#     def __init__(self):
#         self.models = {}

#     def train_random_forest(
#         self,
#         X_train,
#         y_train,
#         n_estimators=200,
#         max_depth=None,
#         random_state=42
#     ):
#         """
#         Train Random Forest baseline model
#         """
#         rf = RandomForestClassifier(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             n_jobs=-1,
#             random_state=random_state,
#             class_weight='balanced'
#         )
#         rf.fit(X_train, y_train)
#         self.models['random_forest'] = rf
#         return rf

#     def train_logistic_regression(
#         self,
#         X_train,
#         y_train,
#         max_iter=1000
#     ):
#         """
#         Train Logistic Regression baseline
#         """
#         lr = LogisticRegression(
#             max_iter=max_iter,
#             n_jobs=-1,
#             class_weight='balanced'
#         )
#         lr.fit(X_train, y_train)
#         self.models['logistic_regression'] = lr
#         return lr

#     def evaluate(
#         self,
#         model,
#         X,
#         y
#     ):
#         """
#         Compute basic evaluation metrics
#         """
#         y_pred = model.predict(X)
#         return {
#             'accuracy': accuracy_score(y, y_pred),
#             'precision': precision_score(y, y_pred, zero_division=0),
#             'recall': recall_score(y, y_pred, zero_division=0),
#             'f1_score': f1_score(y, y_pred, zero_division=0)
#         }

#     def save_model(self, model_name, path):
#         """
#         Save trained model to disk
#         """
#         if model_name not in self.models:
#             raise ValueError(f"Model '{model_name}' not found")
#         joblib.dump(self.models[model_name], path)

#     def load_model(self, path):
#         """
#         Load a saved model
#         """
#         return joblib.load(path)


# class LightweightClassifier:
#     """
#     Fast classifier used inside GA fitness evaluation
#     """

#     def __init__(self, random_state=42):
#         self.model = RandomForestClassifier(
#             n_estimators=50,
#             max_depth=None,
#             n_jobs=-1,
#             random_state=random_state,
#             class_weight='balanced'
#         )

#     def fit(self, X, y):
#         self.model.fit(X, y)

#     def predict(self, X):
#         return self.model.predict(X)

#     def f1_score(self, X, y):
#         y_pred = self.predict(X)
#         return f1_score(y, y_pred, zero_division=0)
