# Cybersecurity Intrusion Detection Optimization
## Using Genetic Algorithm + Machine Learning

**Dataset:** CICIDS2017  
**Technique:** Genetic Algorithm for Feature Selection + Random Forest Classifier

***

## Project Overview

Complete CM4607 Computational Intelligence coursework project. **Only 3 production-ready Jupyter notebooks needed** for full workflow.

***

## Quick Start - 3 Essential Notebooks Only

**⚠️ USE ONLY THESE 3 ROOT-LEVEL NOTEBOOKS (complete final workflow):**

1. **[Preprocessing_CI_IDS.ipynb](Preprocessing_CI_IDS.ipynb)**  
   Data loading → cleaning → SMOTE → train/val/test splits

2. **[Feature_Selection_GA_using_Pro.ipynb](Feature_Selection_GA_using_Pro.ipynb)**  
   Complete GA feature selection → saves optimal feature mask

3. **[Model_Training_+_Evaluation.ipynb](Model_Training_+_Evaluation.ipynb)**  
   Baseline RF → GA-optimized RF → full evaluation + visualizations

**Run in this exact order: 1→2→3.** That's it. Complete pipeline.

***

## Repository Structure

```
Cybersecurity-IDS-Optimization-GA/
├── **3 WORKING NOTEBOOKS** ← ONLY USE THESE
│   ├── Preprocessing_CI_IDS.ipynb              # 1. Data prep
│   ├── Feature_Selection_GA_using_Pro.ipynb    # 2. GA feature selection
│   └── Model_Training_+_Evaluation.ipynb       # 3. Model training + results
│
├── notebooks/           # IGNORE - old/broken files
├── src/                 # IGNORE - abandoned Python package
├── Nina-CI-CW-Report.pdf  # FINAL COURSEWORK REPORT
├── data/                   # CICIDS2017 CSV files
└── reports/                # Generated results
```

**Other notebooks/files = development artifacts. IGNORE completely.**

***

## Execution (3 Simple Steps)

```bash
# Google Colab / Jupyter - Root directory
1. Preprocessing_CI_IDS.ipynb          # ~10-15 min
2. Feature_Selection_GA_using_Pro.ipynb # ~60 min  
3. Model_Training_+_Evaluation.ipynb    # ~20 min
```

**Total runtime:** ~75 min (full dataset) | ~20 min (sample data)

***

## Dataset Setup

Download CICIDS2017 → extract `MachineLearningCSV.zip` → copy **8 CSV files** to `./data/`:

```
Monday-WorkingHours.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv  
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

***

## Key Results

| Model | Features | Macro F1 | FPR | Train Time |
|-------|----------|----------|-----|------------|
| **Baseline RF** | 77 | 0.9372 | 0.0009 | 781s |
| **GA-Optimized** | **33 (-57%)** | **0.9696 (+3.5%)** | **0.0005 (-44%)** | **582s (-25%)** |

**Generated:** Confusion matrices, ROC curves, GA fitness plots, feature charts.

***

##  Tasks Covered

| Task | Status | Notebook |
|------|----------|----------|
| **1. Industry App** | Complete | Report + Preprocessing |
| **2. GA Justification** | Complete | Report + Feature_Selection |
| **3. Architecture** | Complete | Report + README |
| **4. Methodology** | Complete | Feature_Selection_GA_using_Pro.ipynb |
| **5. Implementation** | Complete | All 3 notebooks |

***

## Dependencies (Colab-Ready)

```
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn deap joblib
```

***

## Final Report

**[Nina-CI-CW-Report.pdf](Nina-CI-CW-Report.pdf)** = Complete coursework submission:
- All 5 tasks documented
- Architecture diagrams
- 57% feature reduction proof
- Full results analysis
- Ready to submit

***

**Highlight:**
```
• 77→33 features (-57%)
• F1: 0.9372→0.9696 (+3.5%)
• FPR: 0.0009→0.0005 (-44%)
```

***

## Submission Package

```
 3x Working notebooks (root level)
 Nina-CI-CW-Report.pdf
 Reproducible results
```
