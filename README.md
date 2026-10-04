# Causal Representation Learning from Multimodal Clinical Records under Non-Random Modality Missingness

This repository contains the implementation of our causal representation learning framework for handling multimodal clinical data with Missing-Not-At-Random (MMNAR) patterns. The pipeline integrates structured EHR features, MIMIC-CXR image embeddings, and MIMIC-IV clinical notes, then applies a rectifier to reduce residual modality-specific bias.

All code in this repository is developed and tested with publicly available PhysioNet datasets:

* MIMIC-IV v3.1
* MIMIC-IV-Note: Deidentified free-text clinical notes v2.2
* Generalized Image Embeddings for the MIMIC Chest X-Ray dataset v1.0

Access to these datasets requires the appropriate PhysioNet credentialing and data-use agreements.

## Overview

The framework consists of three key components:

1. **MMNAR-Aware Modality Fusion**: integrates representations from structured EHR data, chest X-ray embeddings, discharge notes, and radiology reports.
2. **Representation Balancing Module**: encourages generalization across different modality missingness patterns.
3. **Outcome Prediction and Rectifier Module**: trains separate readmission, ICU, and mortality models, with auxiliary outcome relationships for post-discharge tasks, and applies a validation-fitted rectifier to correct residual bias.

The pipeline prepares two cohorts: a post-discharge cohort for 30-day readmission and ICU admission within 90 days after discharge, and an independent ICU-landmark cohort for in-hospital mortality prediction at ICU admission + 24 hours. Mortality uses early structured features, CXR embeddings, and radiology reports; discharge notes are excluded from this task.

## Data Setup

Before running the pipeline, prepare the local data directories below. By default, `mimiciv_cxr/` and `mimiciv_note/` are sibling directories of this repository.

### Structured MIMIC-IV Tables

`preprocess.py` downloads the required MIMIC-IV v3.1 structured tables from the BigQuery datasets under `physionet-data`. Configure Google Cloud application-default credentials with access to PhysioNet BigQuery before running the script.

Replace both `YOUR_GCP_PROJECT_ID` placeholders in `preprocess.py` with your billing project ID, or set both `gcp_project` values to `None` to use the project from your default credentials. Configuration is defined in the script; there is no `--gcp-project` command-line option.

### MIMIC-CXR Image Embeddings

Download **Generalized Image Embeddings for the MIMIC Chest X-Ray dataset v1.0** from:

https://physionet.org/content/image-embeddings-mimic-cxr/1.0/

After extraction, move or copy the `p*` folders from the downloaded `files/` directory into `../mimiciv_cxr/`. The expected layout is:

```text
../mimiciv_cxr/
  p10/
    p10000032/
      s.../
        *.tfrecord
  ...
  p19/
```

The CXR preprocessing code expects paths of the form `../mimiciv_cxr/pXX/p<subject_id>/<study>/*.tfrecord`. Also place `mimic-cxr-2.0.0-metadata.csv.gz` (or its uncompressed CSV) under this directory. If the metadata lacks subject/study identifiers, include `mimic-cxr-2.0.0-split.csv.gz` (or its uncompressed CSV) as well. These files provide acquisition times and subject/study mapping; preprocessing selects frontal images within the cohort-specific time windows.

### MIMIC-IV-Note

Download **MIMIC-IV-Note v2.2** from:

https://physionet.org/content/mimic-iv-note/2.2/

Extract the note CSV files and place them directly in `../mimiciv_note/`. The pipeline uses:

```text
../mimiciv_note/
  discharge.csv
  radiology.csv
```

## Expected Directory Layout

After preprocessing and training, the expected directory layout is:

```text
<parent>/
  mimiciv_cxr/
  mimiciv_note/
  CRL-MMNAR/
    preprocess.py
    preprocess_common.py
    preprocess_mortality.py
    preprocess_post_discharge.py
    preprocess_cxr.py
    multimodal_missingness.py
    rectifier.py
    data/
      mortality/
      post_discharge/
    outputs/
      model/
      rectifier/
```

`preprocess.py` generates the following main inputs:

| Directory | Main feature and embedding files |
| --- | --- |
| `data/mortality/` | `patient_features_early.csv`, `patient_features_discharge.csv`, `patient_text_embeddings.csv`, `cxr_embeddings_early_aggregated.csv`, `cxr_embeddings_aggregated.csv` |
| `data/post_discharge/` | `patient_features_discharge_complete.csv`, `patient_text_embeddings_post.csv`, `cxr_embeddings_aggregated.csv` |

Both directories also contain `selected_subject_ids.csv`, `analysis_cohort.csv`, downloaded structured tables, artifact metadata (`*.meta.json`), and `preflight_report.json`. Keep the metadata alongside the CSVs; training checks cohort fingerprints and view compatibility.

## Prerequisites

The current code requires Python 3.10 or newer (the rectifier uses `int.bit_count()`). Core dependencies are:

```text
numpy
pandas
scipy
scikit-learn
torch
google-cloud-bigquery
transformers
tensorflow
```

Text preprocessing loads `emilyalsentzer/Bio_ClinicalBERT` through Hugging Face Transformers. TensorFlow is used to parse the downloaded CXR TFRecords. Training uses CUDA when available and otherwise falls back to CPU. Dependency versions are not pinned in this repository.

## Usage

### Step 1: Data Preprocessing

After setting the configuration in `preprocess.py`, run the preprocessing script to generate structured features, patient-level note embeddings, and aggregated CXR embeddings for both cohorts:

```bash
python preprocess.py
```

The default configuration selects 20,000 subjects per cohort with seed 42 and a 24-hour ICU observation window. The post-discharge note window allows a 24-hour grace period after discharge.

This script will:

* Select reproducible MIMIC-IV cohorts from BigQuery.
* Download required structured hospital and ICU tables to `data/mortality/` and `data/post_discharge/`.
* Build task-specific structured features and outcome labels.
* Generate patient-level embeddings from `../mimiciv_note/discharge.csv` and `../mimiciv_note/radiology.csv`.
* Aggregate downloaded MIMIC-CXR TFRecord embeddings from `../mimiciv_cxr/` using early and hospital-course time windows.
* Validate cohort alignment and write preflight reports.

### Step 2: Train the Multimodal Models and Apply the Rectifier

Run:

```bash
python multimodal_missingness.py
```

This script will:

* Load the post-discharge inputs for readmission and ICU prediction, and early ICU-landmark inputs for mortality.
* Train each task with MMNAR-aware fusion, reconstruction/contrastive pretraining, and auxiliary objectives. Readmission and ICU share the same subject split.
* Use a stratified holdout split: 68% training, 12% validation, and 20% test, with seed 42 by default.
* Save `best_model.pt`, `structured_preprocessing.json`, and `metrics.json` under `outputs/model/<task>/seed_42/holdout/`.
* Save the base test metric summary to `outputs/model/metrics.json`.
* Fit and tune the rectifier on validation predictions, apply it to test predictions, and save base/rectified metrics and correction details under `outputs/rectifier/<task>/metrics.json`, with a combined summary in `outputs/rectifier/metrics.json`.
