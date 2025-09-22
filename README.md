# Causal Representation Learning from Multimodal Clinical Records under Non-Random Modality Missingness

This repository contains the implementation of our causal representation learning framework for handling multimodal clinical data with Missing-Not-At-Random (MMNAR) patterns. Our approach addresses the challenge of missing modalities in clinical data, which is often determined by physician decision-making and patient conditions rather than random processes.

**Note**: All code in this repository is developed and tested using the following publicly available datasets:

* MIMIC-IV v3.1
* MIMIC-IV-Note: Deidentified free-text clinical notes v2.2
* Generalized Image Embeddings for the MIMIC Chest X-Ray dataset v1.0

## Overview

Our framework consists of three key components:

1. **MMNAR-Aware Modality Fusion**: Integrates representations from structured data, medical imaging, and clinical text using large language models and modality-specific encoders.
2. **Representation Balancing Module**: Encourages generalization across varying modality missing patterns.
3. **Multitask Outcome Prediction Module**: Jointly models multiple clinical outcomes and applies a rectifier to correct residual biases.

## Framework Workflow

Before running any steps below, please **first perform data preprocessing** using the provided `data_preprocessing.ipynb` notebook to prepare and align the multimodal data.

The implementation consists of two main Python scripts that work sequentially:

1. `multimodal_missingness.py` - Implements the MMNAR-Aware Modality Fusion and Representation Balancing Module
2. `apply_rectifier.py` - Implements the Rectifier Correction for the Multitask Outcome Prediction Module

## Prerequisites

```
Python 3.8.10
numpy 1.23.5
pandas 1.5.2
scipy 1.9.3
scikit-learn 1.1.3
torch 1.13.1+cu116
torchvision 0.14.1+cu116
matplotlib 3.6.2
seaborn 0.12.1
tqdm 4.64.1
```

## Usage Instructions

### Step 0: Data Preprocessing

Before any training, please run the data preprocessing notebook to generate harmonized input files:

```bash
jupyter notebook data_preprocessing.ipynb
```

This notebook will:

* Parse and clean structured EHR data
* Process and align clinical notes
* Load image embeddings
* Generate the intermediate data required by subsequent training scripts

### Step 1: Train the Multimodal Model

Then, run the `multimodal_missingness.py` script to train the model with MMNAR-aware fusion and representation balancing:

```bash
python multimodal_missingness.py
```

This script will:

* Load multimodal data (structured EHR, chest X-rays, clinical notes, radiology reports)
* Train the RB model with MMNAR-aware fusion and representation balancing
* Save the best model as `best_rb_model.pt` in the `outputs_model/models` directory
* Generate predictions in `rb_predictions.csv` in the `outputs_model/results` directory

### Step 2: Apply the Rectifier

After obtaining the prediction CSV file, run the `apply_rectifier.py` script to correct residual biases:

```bash
python apply_rectifier.py --csv_path path/to/rb_predictions.csv --selective --output_prefix enhanced_results
```

Parameters:

* `--csv_path`: Path to the predictions CSV file (output from Step 1)
* `--selective`: Use selective correction (recommended)
* `--output_prefix`: Prefix for output files

This script will:

* Load the predictions from the CSV file
* Apply the rectifier correction to adjust for modality-specific biases
* Generate enhanced prediction results in `enhanced_results_selective_results.csv`
* Create performance comparison charts (`enhanced_results_roc.png` and `enhanced_results_pr.png`)
