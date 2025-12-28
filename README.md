# Cybersecurity Intrusion Detection Optimization
## Using Genetic Algorithm + Machine Learning

**Dataset:** CICIDS2017  
**Technique:** Genetic Algorithm for Feature Selection + Random Forest Classifier

---

## 📋 Project Overview

This project demonstrates the application of Evolutionary Computing (specifically Genetic Algorithms) combined with Machine Learning to optimize intrusion detection in cybersecurity. The solution addresses the high-dimensional, complex nature of network traffic data where traditional optimization methods struggle.

### Problem Statement

Network intrusion detection faces two major challenges:
1. **High dimensionality**: Network traffic datasets contain 70+ features, many of which may be redundant or irrelevant
2. **Real-time constraints**: Security systems must detect threats quickly with minimal false positives

### Why Genetic Algorithms?

- **Combinatorial complexity**: With 78 features, there are 2^78 possible feature subsets—computationally infeasible for exhaustive search
- **Multi-objective optimization**: Balance between model performance (high F1-score) and feature reduction (faster inference)
- **Non-linear search space**: No gradient information available; GA's evolutionary approach explores effectively
- **Constraint handling**: GA naturally handles minimum/maximum feature constraints

---

## 🏗️ Solution Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA INGESTION MODULE                     │
│                     (preprocessing.py)                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ • Load CICIDS2017 CSV files                         │    │
│  │ • Clean: Handle NaN, infinite values               │    │
│  │ • Encode: BENIGN=0, ATTACK=1                       │    │
│  │ • Scale: StandardScaler normalization              │    │
│  │ • Split: Train/Validation/Test (70/15/15)          │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              BASELINE ML MODULE (models.py)                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ • Random Forest (n_estimators=200)                  │    │
│  │ • Logistic Regression                               │    │
│  │ • Evaluation: Accuracy, Precision, Recall, F1, FPR  │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         GENETIC ALGORITHM MODULE (ga.py)                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ CHROMOSOME: Binary vector [g₁, g₂, ..., gₙ]        │    │
│  │   gᵢ = 1 → feature i selected                       │    │
│  │   gᵢ = 0 → feature i excluded                       │    │
│  │                                                      │    │
│  │ FITNESS FUNCTION:                                    │    │
│  │   fitness = α·F1_score - β·(k/d)                    │    │
│  │   where:                                             │    │
│  │     k = number of selected features                  │    │
│  │     d = total features                               │    │
│  │     α = 1.0 (performance weight)                     │    │
│  │     β = 0.05 (complexity penalty)                    │    │
│  │                                                      │    │
│  │ CONSTRAINTS:                                         │    │
│  │   • k_min ≤ k ≤ k_max                               │    │
│  │   • k_min = 10 (minimum features)                   │    │
│  │                                                      │    │
│  │ OPERATORS:                                           │    │
│  │   • Selection: Tournament (size=3)                   │    │
│  │   • Crossover: Uniform (prob=0.7, indpb=0.5)        │    │
│  │   • Mutation: Bit flip (prob=0.2, indpb=0.02)       │    │
│  │                                                      │    │
│  │ PARAMETERS:                                          │    │
│  │   • Population size: 50                              │    │
│  │   • Generations: 30                                  │    │
│  │   • Elitism: Best individuals preserved              │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          OPTIMIZED MODEL MODULE (models.py)                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ • Random Forest with selected features               │    │
│  │ • n_estimators=300 (enhanced for final model)        │    │
│  │ • Trained on GA-selected feature subset              │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          EVALUATION MODULE (evaluate.py)                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ METRICS:                                             │    │
│  │   • Classification: Accuracy, Precision, Recall, F1  │    │
│  │   • Cybersecurity: FPR, FNR, Detection Rate          │    │
│  │   • Performance: ROC-AUC, Inference Time             │    │
│  │                                                      │    │
│  │ VISUALIZATIONS:                                      │    │
│  │   • Confusion matrices                               │    │
│  │   • ROC curves                                       │    │
│  │   • Metrics comparison charts                        │    │
│  │   • GA fitness evolution plots                       │    │
│  │                                                      │    │
│  │ REPORTS:                                             │    │
│  │   • Detailed text report                             │    │
│  │   • JSON results export                              │    │
│  │   • CSV comparison table                             │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup Steps

1. **Clone/Download the project**
   ```bash
   cd cybersecurity-intrusion-detection
   ```

2. **Create project structure**
   ```bash
   python create_structure.py
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download CICIDS2017 dataset**
   - Download `MachineLearningCSV.zip` from the CICIDS2017 website
   - Extract all CSV files to the `data/` directory
   - Files should include:
     - Monday-WorkingHours.pcap_ISCX.csv
     - Tuesday-WorkingHours.pcap_ISCX.csv
     - Wednesday-workingHours.pcap_ISCX.csv
     - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
     - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
     - Friday-WorkingHours-Morning.pcap_ISCX.csv
     - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
     - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

---

## 🚀 Usage

### Quick Start (Full Pipeline)

Run the complete pipeline:
```bash
python main.py
```

This will execute all steps:
1. Data preprocessing
2. Baseline model training
3. GA feature selection
4. Optimized model training
5. Comprehensive evaluation
6. Report generation

### Testing with Sample Data

For faster testing (useful during development):

Edit `main.py` line ~40:
```python
processed_data = preprocessor.preprocess_pipeline(
    max_rows_per_file=50000  # Loads 50K rows per file
)
```

For full dataset, remove the parameter:
```python
processed_data = preprocessor.preprocess_pipeline()
```

### Individual Modules

You can also run modules independently:

**Preprocessing only:**
```bash
python src/preprocessing.py
```

**Train baseline models:**
```python
from src.preprocessing import CICIDS2017Preprocessor
from src.models import BaselineModels

preprocessor = CICIDS2017Preprocessor()
data = preprocessor.preprocess_pipeline()

models = BaselineModels()
models.compare_models(data['X_train'], data['y_train'], 
                     data['X_val'], data['y_val'])
```

**Run GA feature selection:**
```python
from src.ga import GeneticFeatureSelector

ga = GeneticFeatureSelector(
    X_train, y_train, X_val, y_val,
    pop_size=50, n_generations=30
)
best_individual, fitness = ga.run()
```

---

## 📊 Output Files

After running the pipeline, results are saved to:

```
reports/
├── figures/
│   ├── confusion_matrices.png      # Side-by-side confusion matrices
│   ├── roc_curves.png               # ROC curve comparison
│   ├── metrics_comparison.png       # Bar chart of metrics
│   ├── ga_fitness_evolution.png     # GA fitness over generations
│   └── ga_generation_times.png      # GA execution time per generation
│
└── results/
    ├── evaluation_report.txt        # Detailed text report
    ├── results.json                 # Machine-readable results
    ├── model_comparison.csv         # Tabular comparison
    └── selected_features.csv        # List of GA-selected features

models/
└── baseline_rf.pkl                  # Saved baseline model
```

---

## 🧬 Genetic Algorithm Details

### Chromosome Representation

Each individual is a binary vector of length `d` (number of features):

```
Chromosome = [g₁, g₂, g₃, ..., gₐ]

Example (d=10):
[1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
 ↑     ↑  ↑        ↑     ↑
Selected features: 1, 3, 4, 7, 9
```

### Fitness Function

The fitness function balances model performance with feature reduction:

```
fitness(individual) = α · F1_score - β · (k/d)

where:
  - F1_score: Harmonic mean of precision and recall
  - k: Number of selected features (sum of 1s in chromosome)
  - d: Total number of features
  - α: Performance weight (1.0)
  - β: Complexity penalty (0.05)
```

**Example calculation:**
```
Suppose:
- F1_score = 0.95
- k = 30 selected features
- d = 78 total features
- α = 1.0, β = 0.05

fitness = 1.0 × 0.95 - 0.05 × (30/78)
        = 0.95 - 0.0192
        = 0.9308
```

### Constraints

1. **Minimum features**: `k ≥ k_min` (default: 10)
   - Prevents overly sparse feature sets
   - Ensures minimum information for classification

2. **Maximum features**: `k ≤ k_max` (optional)
   - Can limit computational cost
   - Default: no limit (k_max = d)

### Genetic Operators

1. **Selection**: Tournament Selection (size=3)
   - Randomly select 3 individuals
   - Choose the one with highest fitness
   - Balances exploration and exploitation

2. **Crossover**: Uniform Crossover (prob=0.7)
   - Each gene has 50% chance to come from either parent
   - Creates diverse offspring

3. **Mutation**: Bit Flip Mutation (prob=0.2)
   - Each gene has 2% chance to flip (0→1 or 1→0)
   - Maintains genetic diversity

### Evolution Parameters

```python
pop_size = 50           # Population size
n_generations = 30      # Number of generations
cx_prob = 0.7          # Crossover probability
mut_prob = 0.2         # Mutation probability
tournament_size = 3    # Selection tournament size
```

---

## 📈 Expected Results

### Baseline Model Performance
- **Accuracy**: ~0.98-0.99
- **F1-Score**: ~0.97-0.99
- **False Positive Rate**: ~0.01-0.03
- **Features**: All 78 features

### GA-Optimized Model Performance
- **Accuracy**: Similar or better (~0.98-0.99)
- **F1-Score**: Comparable or improved
- **False Positive Rate**: Potentially reduced
- **Features**: 20-40 features (40-50% reduction)
- **Inference Time**: Significantly faster

### Key Improvements
1. ✅ **Feature Reduction**: 40-60% fewer features
2. ✅ **Faster Inference**: 30-50% speed improvement
3. ✅ **Maintained/Improved Accuracy**: No performance loss
4. ✅ **Lower False Positives**: Critical for cybersecurity
5. ✅ **Better Interpretability**: Fewer features to analyze

---

## 🎯 Coursework Deliverables

This code provides all components needed for the coursework:

### Task 1: Industry Application ✅
- **Application**: Network intrusion detection
- **Complexity**: High-dimensional feature space (78 features)
- **Scale**: CICIDS2017 dataset with multiple attack types

### Task 2: Justification for GA ✅
- **Combinatorial explosion**: 2^78 possible feature subsets
- **Multi-objective**: Performance vs. feature reduction
- **Non-convex search space**: No gradient-based solution
- **Constraint handling**: Minimum/maximum features

### Task 3: Modularized Diagram ✅
- See "Solution Architecture" section above
- Also available in the code documentation

### Task 4: Methodology ✅
- **Chromosome**: Binary vector detailed above
- **Fitness**: α·F1 - β·(k/d) equation provided
- **Constraints**: k_min and k_max defined

### Task 5: Implementation ✅
- Complete working prototype
- Visualizations and reports
- Ready for demonstration and viva

---

## 🔧 Customization

### Adjust GA Parameters

Edit `main.py` around line 130:

```python
ga_selector = GeneticFeatureSelector(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    alpha=1.0,           # ← Adjust fitness weights
    beta=0.05,           # ← Adjust complexity penalty
    k_min=10,            # ← Change minimum features
    pop_size=50,         # ← Increase for better exploration
    n_generations=30,    # ← More generations = better solution
    cx_prob=0.7,         # ← Crossover probability
    mut_prob=0.2         # ← Mutation probability
)
```

### Use Different Classifier

Edit `src/ga.py`, `_evaluate_individual` method:

```python
# Replace LogisticRegression with:
from sklearn.svm import SVC
clf = SVC(kernel='rbf', random_state=42)

# Or use XGBoost:
from xgboost import XGBClassifier
clf = XGBClassifier(n_estimators=100)
```

### Enable SMOTE for Class Balance

Edit `src/preprocessing.py`:

```python
from imblearn.over_sampling import SMOTE

def apply_smote(self, X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled
```

---

## 📝 Report Writing Tips

### Problem Definition
- Emphasize the **high-dimensional** nature of network traffic
- Highlight **real-time detection** requirements
- Discuss the **cost of false positives** in cybersecurity

### Justification for GA
- Explain why **exhaustive search** is impossible (2^78 combinations)
- Describe how **traditional optimization** fails on non-convex spaces
- Justify the **multi-objective** nature (performance + efficiency)

### Methodology
- Include the **chromosome representation** diagram
- Show the **fitness function** equation with clear explanation
- Document the **constraints** and their cybersecurity rationale
- Describe **genetic operators** with examples

### Results
- Present **comparison tables** (baseline vs. GA)
- Show **visualization plots** from reports/figures/
- Highlight **key improvements**: feature reduction, FPR, speed
- Discuss **trade-offs** if any

### Effectiveness
- Quantify **feature reduction** percentage
- Show **maintained or improved** classification performance
- Demonstrate **faster inference** for real-time deployment
- Discuss **scalability** and practical deployment considerations

---

## 🐛 Troubleshooting

### Memory Issues
If you run out of memory:
```python
# Use smaller sample in main.py
processed_data = preprocessor.preprocess_pipeline(
    max_rows_per_file=30000  # Reduce this number
)
```

### Slow GA Execution
If GA is too slow:
```python
# Reduce GA parameters
ga_selector = GeneticFeatureSelector(
    pop_size=30,         # Smaller population
    n_generations=20     # Fewer generations
)
```

### Installation Errors
If pip install fails:
```bash
# Upgrade pip first
pip install --upgrade pip

# Install packages one by one
pip install numpy pandas scikit-learn
pip install deap matplotlib seaborn
```

---

## 📚 References

### Dataset
- **CICIDS2017**: Canadian Institute for Cybersecurity Intrusion Detection System
- Source: https://www.unb.ca/cic/datasets/ids-2017.html

### Libraries
- **scikit-learn**: Machine learning algorithms
- **DEAP**: Distributed Evolutionary Algorithms in Python
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization

### Theory
- Holland, J. H. (1992). Genetic Algorithms. Scientific American.
- Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization and Machine Learning.
- Guyon, I. & Elisseeff, A. (2003). An Introduction to Variable and Feature Selection.

---

## 📞 Support

For coursework-related questions:
- Check the module Moodle page
- Contact module coordinator: Fathima Farhath
- Refer to the coursework specification document

For technical issues:
- Review the troubleshooting section above
- Check console output for error messages
- Verify all CSV files are in the `data/` directory

---

## ✅ Checklist Before Submission

- [ ] All CSV files placed in `data/` directory
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Main pipeline runs successfully (`python main.py`)
- [ ] All output files generated in `reports/`
- [ ] Code is well-commented and readable
- [ ] Report written covering all 5 tasks
- [ ] Figures included in report from `reports/figures/`
- [ ] Results tables included from `reports/results/`
- [ ] Demonstration prepared (ready to show working prototype)
- [ ] Viva preparation (understand GA, fitness function, results)

---

## 📄 License

This project is for educational purposes as part of CM 4607 coursework.

---

**Good luck with your coursework! 🚀**