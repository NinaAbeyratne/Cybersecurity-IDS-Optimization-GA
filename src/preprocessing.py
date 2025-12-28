"""
Data Preprocessing Module for CICIDS2017 Dataset

Handles loading, cleaning, encoding, and splitting of the intrusion detection data.
This module is critical for preparing high-dimensional network traffic data for ML.

Dataset: CICIDS2017
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import glob
import os

class CICIDS2017Preprocessor:
    """
    Preprocessor for CICIDS2017 intrusion detection dataset.
    
    Handles:
    - Loading multiple CSV files
    - Cleaning infinite/NaN values
    - Label encoding (binary: benign vs attack)
    - Feature scaling
    - Train/validation/test splitting
    """
    
    def __init__(self, data_dir='data/'):
        """
        Initialize preprocessor.
        
        Args:
            data_dir: Directory containing CICIDS2017 CSV files
        """
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def load_data(self, sample_frac=None, max_rows_per_file=None):
        """
        Load all CICIDS2017 CSV files from data directory.
        
        Args:
            sample_frac: Fraction of data to sample (for faster testing)
            max_rows_per_file: Maximum rows to load per file
            
        Returns:
            DataFrame with all loaded data
        """
        print("Loading CICIDS2017 dataset...")
        csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        dataframes = []
        for file in csv_files:
            print(f"Loading {os.path.basename(file)}...")
            try:
                if max_rows_per_file:
                    df = pd.read_csv(file, nrows=max_rows_per_file, encoding='utf-8', low_memory=False)
                else:
                    df = pd.read_csv(file, encoding='utf-8', low_memory=False)
                
                # Sample if requested
                if sample_frac and sample_frac < 1.0:
                    df = df.sample(frac=sample_frac, random_state=42)
                
                dataframes.append(df)
                print(f"  Loaded {len(df)} rows")
            except Exception as e:
                print(f"  Error loading {file}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No data was successfully loaded")
        
        # Combine all dataframes
        data = pd.concat(dataframes, ignore_index=True)
        print(f"\nTotal rows loaded: {len(data)}")
        print(f"Total columns: {len(data.columns)}")
        
        return data
    
    def clean_data(self, data):
        """
        Clean the dataset by handling missing values, infinities, and duplicates.
        
        Args:
            data: Raw DataFrame
            
        Returns:
            Cleaned DataFrame, label column name
        """
        print("\nCleaning data...")
        initial_rows = len(data)
        
        # Strip whitespace from column names
        data.columns = data.columns.str.strip()
        
        # Identify label column (usually 'Label' or ' Label')
        label_cols = [col for col in data.columns if 'label' in col.lower()]
        if not label_cols:
            raise ValueError("Could not find label column in dataset")
        
        label_col = label_cols[0]
        print(f"Using label column: '{label_col}'")
        
        # Replace infinite values with NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with any NaN values (conservative approach)
        data = data.dropna()
        print(f"Removed {initial_rows - len(data)} rows with missing/infinite values")
        
        # Remove duplicate rows
        duplicates = data.duplicated().sum()
        data = data.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows")
        
        print(f"Clean dataset: {len(data)} rows")
        
        return data, label_col
    
    def encode_labels(self, data, label_col):
        """
        Encode labels as binary: BENIGN (0) vs ATTACK (1).
        
        Args:
            data: DataFrame with raw labels
            label_col: Name of label column
            
        Returns:
            Series of binary labels, Series of original labels
        """
        print("\nEncoding labels...")
        
        # Get unique attack types
        unique_labels = data[label_col].unique()
        print(f"Found {len(unique_labels)} unique labels:")
        for label in sorted(unique_labels):
            count = (data[label_col] == label).sum()
            print(f"  {label}: {count} samples")
        
        # Create binary labels: BENIGN = 0, all attacks = 1
        y = (data[label_col].str.upper() != 'BENIGN').astype(int)
        
        benign_count = (y == 0).sum()
        attack_count = (y == 1).sum()
        print(f"\nBinary encoding:")
        print(f"  BENIGN: {benign_count} samples ({benign_count/len(y)*100:.2f}%)")
        print(f"  ATTACK: {attack_count} samples ({attack_count/len(y)*100:.2f}%)")
        
        return y, data[label_col].copy()
    
    def prepare_features(self, data, label_col):
        """
        Prepare feature matrix by removing non-feature columns.
        
        Args:
            data: DataFrame with all columns
            label_col: Name of label column to exclude
            
        Returns:
            Feature matrix (DataFrame)
        """
        print("\nPreparing features...")
        
        # Columns to exclude
        exclude_cols = [label_col]
        
        # Also exclude any timestamp or IP address columns if present
        for col in data.columns:
            col_lower = col.lower().strip()
            if any(x in col_lower for x in ['timestamp', 'time', 'date', 'ip', 'port']):
                if col not in exclude_cols:
                    exclude_cols.append(col)
                    print(f"  Excluding column: {col}")
        
        # Select feature columns
        X = data.drop(columns=exclude_cols, errors='ignore')
        
        # Convert all features to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Drop any columns that couldn't be converted
        X = X.dropna(axis=1, how='all')
        
        self.feature_names = X.columns.tolist()
        print(f"Feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X
    
    def scale_features(self, X_train, X_val, X_test):
        """
        Scale features using StandardScaler fitted on training data.
        
        Args:
            X_train, X_val, X_test: Feature matrices
            
        Returns:
            Scaled feature matrices (numpy arrays)
        """
        print("\nScaling features...")
        
        # Fit scaler on training data only
        self.scaler.fit(X_train)
        
        # Transform all sets
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Scaling complete")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def split_data(self, X, y, test_size=0.15, val_size=0.15, random_state=42):
        """
        Split data into train/validation/test sets with stratification.
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Fraction for test set
            val_size: Fraction for validation set (from remaining data)
            random_state: Random seed
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print(f"\nSplitting data (test={test_size}, val={val_size})...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=random_state
        )
        
        print(f"Training set:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"Test set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_pipeline(self, sample_frac=None, max_rows_per_file=None):
        """
        Execute full preprocessing pipeline.
        
        Args:
            sample_frac: Fraction of data to sample
            max_rows_per_file: Maximum rows per CSV file
            
        Returns:
            Dictionary with all processed data
        """
        # Load data
        data = self.load_data(sample_frac=sample_frac, max_rows_per_file=max_rows_per_file)
        
        # Clean data
        data, label_col = self.clean_data(data)
        
        # Encode labels
        y, y_original = self.encode_labels(data, label_col)
        
        # Prepare features
        X = self.prepare_features(data, label_col)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(X_train, X_val, X_test)
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train.values,
            'y_val': y_val.values,
            'y_test': y_test.values,
            'feature_names': self.feature_names,
            'scaler': self.scaler
        }


if __name__ == "__main__":
    # Example usage
    preprocessor = CICIDS2017Preprocessor(data_dir='data/')
    
    # For testing, use a sample
    # Remove sample_frac parameter for full dataset
    processed_data = preprocessor.preprocess_pipeline(max_rows_per_file=50000)
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
    print(f"Training samples: {len(processed_data['y_train'])}")
    print(f"Features: {len(processed_data['feature_names'])}")