# 🛡️ Cyber IIoT Intrusion Detection System

> A **multi-class network intrusion detection pipeline** for Industrial Internet of Things (IIoT) environments, built on the **WUSTL-IIoT-2021** dataset. The system uses a 5-stage bio-inspired feature selection pipeline, SMOTE-based class balancing, Bayesian hyperparameter optimisation, and an adaptive weighted ensemble of XGBoost, LightGBM, and Random Forest.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Feature Selection Pipeline](#feature-selection-pipeline)
- [Models & Ensemble](#models--ensemble)
- [Key Design Principles](#key-design-principles)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)

---

## Overview

This project implements an **end-to-end intrusion detection system (IDS)** designed for IIoT networks. It classifies network traffic into multiple attack categories using a modular, cell-based notebook architecture. The pipeline places strong emphasis on:

- **Leakage-free evaluation** — all preprocessing fits on training data only
- **Bio-inspired feature selection** — Red Ant Colony optimisation for intelligent subset search
- **Class imbalance handling** — SMOTE oversampling with balanced class weights
- **Automated tuning** — Optuna Bayesian (BO-TPE) hyperparameter optimisation
- **Explainability** — SHAP-based model interpretability

---

## Dataset

| Property | Details |
|----------|---------|
| **Name** | WUSTL-IIoT-2021 |
| **Source** | Washington University in St. Louis |
| **Domain** | Industrial IoT network traffic |
| **Task** | Multi-class classification (attack type detection) |
| **Target Column** | `Traffic` (multiple attack categories) |

The dataset is loaded from `wustl_iiot_2021/wustl_iiot_2021.csv`. Non-numeric identifier and timestamp columns are dropped during preprocessing.

---

## Pipeline Architecture

The pipeline is organised as numbered Python cells, executed sequentially:

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: DATA PREPARATION                                  │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐  ┌───────────┐  │
│  │ GPU     │→ │ Data     │→ │ Train/Test │→ │ Feature   │  │
│  │ Check   │  │ Loading  │  │ Split      │  │ Selection │  │
│  └─────────┘  └──────────┘  └────────────┘  └───────────┘  │
├─────────────────────────────────────────────────────────────┤
│  PHASE 2: FEATURE SELECTION (5-Stage, Train-Only)           │
│  ┌────────┐ ┌───────┐ ┌──────┐ ┌─────────┐ ┌───────────┐  │
│  │ RF     │→│ Corr  │→│ RFE  │→│ Red Ant │→│ Aggregate │  │
│  │ Import.│ │Filter │ │XGBoost│ │ Colony  │ │ Voting    │  │
│  └────────┘ └───────┘ └──────┘ └─────────┘ └───────────┘  │
├─────────────────────────────────────────────────────────────┤
│  PHASE 3: MODEL TRAINING                                    │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────────────┐ │
│  │ SMOTE    │→ │ Model     │→ │ Optuna HPO              │ │
│  │ Balancing│  │ Training  │  │ (Bayesian Optimisation)  │ │
│  └──────────┘  └───────────┘  └──────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  PHASE 4: ENSEMBLE & EVALUATION                             │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────────────┐ │
│  │ SHAP     │→ │ Adaptive  │→ │ Comprehensive            │ │
│  │ Analysis │  │ Ensemble  │  │ Evaluation Suite          │ │
│  └──────────┘  └───────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Feature Selection Pipeline

A multi-stage, voting-based feature selection strategy operating **exclusively on training data** to prevent data leakage:

| Stage | Method | Purpose |
|-------|--------|---------|
| **1** | Random Forest Importance | Preliminary importance ranking |
| **2** | Correlation Filtering | Redundancy removal (threshold 0.95) |
| **3** | RFE with XGBoost (GPU) | Recursive subset optimisation |
| **4** | Red Ant Colony Algorithm | Swarm-intelligence subset search |
| **5** | Aggregation (Voting) | Consensus-based final feature set |

### Red Ant Colony Algorithm

A novel bio-inspired optimisation method where:
- **Ants** probabilistically select feature subsets based on pheromone trails and RF-importance heuristics
- **Fitness** = `Balanced_Accuracy − λ × (n_selected / n_total)` — balances performance vs. parsimony
- **Pheromone update** reinforces features found in high-fitness subsets

---

## Models & Ensemble

### Base Classifiers

| Model | Key Configuration |
|-------|-------------------|
| **XGBoost** | GPU-accelerated (`hist` + CUDA), 500 trees, max_depth=12, L1/L2 regularisation |
| **LightGBM** | Class-weighted (`balanced`), 500 trees, 63 leaves, max_depth=12 |
| **Random Forest** | Unlimited depth, `balanced_subsample` weighting, 500 trees |

### Hyperparameter Optimisation

- **Framework**: Optuna (TPE Sampler — Bayesian Optimisation)
- **Trials**: 20 trials on a 200K subsample
- **Objective**: Maximise 3-fold cross-validated balanced accuracy
- **Scope**: XGBoost hyperparameters (n_estimators, max_depth, learning_rate, regularisation, etc.)

### Adaptive Weighted Soft Voting Ensemble

The final classifier is a **soft voting ensemble** where each model's weight is proportional to its validation balanced accuracy:

```
Weight_i = BalancedAccuracy_i / Σ(BalancedAccuracy_all)
```

This ensures stronger models have greater influence on predictions.

---

## Key Design Principles

- **🔒 Leakage-Free**: Train/test split occurs _before_ any feature selection or scaling. StandardScaler is fit only on training data.
- **⚖️ Class-Aware**: SMOTE oversampling + balanced class weights handle minority attack types.
- **🧬 Bio-Inspired**: Red Ant Colony algorithm adds swarm-intelligence-based feature search beyond traditional methods.
- **📊 Explainable**: SHAP values provide feature-level interpretability for the trained models.
- **🔬 Reproducible**: `RANDOM_SEED = 42` is used across all stochastic components.

---

## Project Structure

```
cyberr/
├── cybersecuritywork_local.ipynb  # Main notebook (local execution)
├── cybersecuritywork.ipynb        # Main notebook (Colab version)
├── run_notebook.bat               # Batch runner script
├── cells/                         # Modular pipeline cells
│   ├── cell_00_gpu_check.py       # GPU availability check
│   ├── cell_01_file_listing.py    # Dataset file discovery
│   ├── cell_02_imports.py         # Library imports & seed
│   ├── cell_03_data_loading.py    # Load & preprocess WUSTL-IIoT-2021
│   ├── cell_03b_train_test_split.py  # Early train/test split (leakage fix)
│   ├── cell_04_fs_header.md       # Feature selection documentation
│   ├── cell_05_fs_stage1_rf.py    # Stage 1: RF importance
│   ├── cell_06_fs_stage2_corr.py  # Stage 2: Correlation filtering
│   ├── cell_07_fs_stage3_rfe.py   # Stage 3: RFE with XGBoost
│   ├── cell_08_fs_stage4_redant.py # Stage 4: Red Ant Colony
│   ├── cell_09_fs_stage5_agg.py   # Stage 5: Voting aggregation
│   ├── cell_10_apply_features.py  # Apply selected features + scale
│   ├── cell_11_train_test_split.py # Scaling (train-fitted only)
│   ├── cell_12_smote.py           # SMOTE oversampling
│   ├── cell_13_model_training.py  # Train XGBoost, LightGBM, RF
│   ├── cell_13b_hpo_optuna.py     # Optuna hyperparameter optimisation
│   ├── cell_14_shap.py            # SHAP explainability analysis
│   ├── cell_15_feature_weighting.py # Feature importance weighting
│   ├── cell_16_ensemble.py        # Adaptive weighted voting ensemble
│   ├── cell_17_evaluation.py      # Primary model evaluation
│   ├── cell_18_eval_header.md     # Evaluation section header
│   ├── cell_19_eval_classdist.py  # Class distribution analysis
│   ├── cell_20_eval_metrics.py    # Detailed per-class metrics
│   ├── cell_21_eval_confusion.py  # Confusion matrix visualisation
│   ├── cell_22_eval_cv.py         # Stratified K-fold cross-validation
│   ├── cell_23_eval_baselines.py  # Baseline comparison (SVM, KNN, etc.)
│   └── cell_24_eval_chart.py      # Model comparison charts
├── wustl_iiot_2021/               # Dataset directory
├── class_distribution.png         # Generated: class distribution plot
├── confusion_matrix.png           # Generated: confusion matrix plot
└── model_comparison.png           # Generated: model comparison chart
```

---

## Requirements

### Core Dependencies

```
numpy
pandas
scikit-learn
xgboost (GPU build)
lightgbm
imbalanced-learn
shap
optuna
matplotlib
seaborn
```

### Hardware

- **GPU**: NVIDIA CUDA-compatible GPU recommended (XGBoost GPU acceleration)
- **RAM**: 16 GB+ recommended for full dataset processing
- **CUDA**: CUDA toolkit matching your XGBoost GPU build

### Install

```bash
pip install numpy pandas scikit-learn xgboost lightgbm imbalanced-learn shap optuna matplotlib seaborn
```

---

## Usage

### 1. Place the dataset

Download the WUSTL-IIoT-2021 dataset and place it in:
```
wustl_iiot_2021/wustl_iiot_2021.csv
```

### 2. Run the notebook

**Jupyter Notebook:**
```bash
jupyter notebook cybersecuritywork_local.ipynb
```

**Batch execution (Windows):**
```bash
run_notebook.bat
```

Run cells sequentially from `cell_00` through `cell_24`. The pipeline will:
1. Verify GPU availability
2. Load and preprocess the dataset
3. Split data into train/test (before any feature engineering)
4. Run the 5-stage feature selection pipeline
5. Apply SMOTE and train the models
6. Optimise XGBoost with Optuna
7. Build the adaptive weighted ensemble
8. Generate comprehensive evaluation outputs

---

## Evaluation Metrics

The evaluation suite (cells 17–24) produces:

| Output | Description |
|--------|-------------|
| **Classification Report** | Per-class precision, recall, F1-score |
| **Balanced Accuracy** | Accounts for class imbalance |
| **ROC-AUC** | Multi-class One-vs-Rest AUC |
| **Confusion Matrix** | Visual heatmap of predictions |
| **Cross-Validation** | Stratified K-fold with multiple metrics |
| **Baseline Comparison** | SVM, KNN, Decision Tree vs. ensemble |
| **Model Comparison Chart** | Bar chart comparing all approaches |

---

## License

This project is developed for academic research purposes.

---

<p align="center">
  <i>Built with ❤️ for IIoT security research</i>
</p>
