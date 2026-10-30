# Causal Representation Learning from Multimodal Clinical Records under Non-Random Modality Missingness

This repository contains the implementation of our causal representation learning framework for handling multimodal clinical data with Missing-Not-At-Random (MMNAR) patterns. The pipeline integrates structured EHR features, MIMIC-CXR image embeddings, and MIMIC-IV clinical notes, then applies a rectifier to reduce residual modality-specific bias.

All code in this repository is developed and tested with publicly available PhysioNet datasets:

* MIMIC-IV v3.1
* MIMIC-IV-Note: Deidentified free-text clinical notes v2.2
* Generalized Image Embeddings for the MIMIC Chest X-Ray dataset v1.0

Access to these datasets requires the appropriate PhysioNet credentialing and data-use agreements.

## Overview

The framework consists of three key components:

1. **MMNAR-Aware Modality Fusion**: integrates representations from structured EHR data, chest X-ray embeddings, discharge notes, and radiology reports through missingness embeddings, modality gating, and a shared representation.
2. **Representation Balancing Module**: learns to predict the latent representations of masked modalities from the remaining observed modalities.
3. **Outcome Prediction and Rectifier Module**: jointly predicts readmission, ICU admission, and mortality with task-specific heads, then applies a validation-fitted pattern rectifier and calibration.

The pipeline prepares a single post-discharge cohort for 30-day readmission, ICU admission within 90 days after discharge, and 90-day mortality after discharge. All three tasks use the same patient split and modality observation windows.

## Data Setup

Before running the pipeline, prepare the local data directories below. By default, all source data are stored under `data/` within this repository. Paths can be configured through the `MIMIC_SNAPSHOT`, `MIMIC_SUPPLEMENT`, `MIMIC_NOTE_DIR`, and `MIMIC_CXR_DIR` environment variables.

### Structured MIMIC-IV Tables

`data.py` downloads the required MIMIC-IV v3.1 structured tables from the BigQuery datasets under `physionet-data`. Configure Google Cloud application-default credentials with access to PhysioNet BigQuery before running the script.

Pass your billing project ID with `--project YOUR_GCP_PROJECT_ID` when running the `export`, `select`, and `fetch` stages. The `export` stage writes the main structured tables to `data/mimic/`, and `fetch` writes supplementary tables to `data/mimic/supplement/`. Override these locations with `MIMIC_SNAPSHOT` and `MIMIC_SUPPLEMENT`; the supplement directory defaults to `supplement/` under the configured snapshot directory.

### MIMIC-CXR Image Embeddings

Download **Generalized Image Embeddings for the MIMIC Chest X-Ray dataset v1.0** from:

[PhysioNet: MIMIC-CXR Image Embeddings](https://physionet.org/content/image-embeddings-mimic-cxr/1.0/)

After extraction, move or copy the `p*` folders from the downloaded `files/` directory into `data/mimic-cxr/`, or set `MIMIC_CXR_DIR` to their location. The expected layout is:

```text
data/mimic-cxr/
  p10/
    p10000032/
      s.../
        *.tfrecord
  ...
  p19/
  mimic-cxr-2.0.0-metadata.csv.gz
```

The CXR preprocessing code expects paths of the form `data/mimic-cxr/pXX/p<subject_id>/s<study_id>/<dicom_id>.tfrecord`. Place `mimic-cxr-2.0.0-metadata.csv.gz` under the same directory. This file provides acquisition times, view positions, and subject/study identifiers; preprocessing selects frontal AP/PA images acquired between admission and discharge. The CXR expert aggregates embeddings from the latest eligible study for each patient.

### MIMIC-IV-Note

Download **MIMIC-IV-Note v2.2** from:

[PhysioNet: MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/)

Extract the note CSV files and place them directly in `data/mimic-note/`, or set `MIMIC_NOTE_DIR` to their location. The pipeline uses:

```text
data/mimic-note/
  discharge.csv
  radiology.csv
```

Discharge summaries are included from admission through 24 hours after discharge. Radiology reports are included from 6 hours before admission through 24 hours after discharge.

## Expected Directory Layout

After preprocessing and training, the expected directory layout is:

```text
CRL-MMNAR/
  common.py
  data.py
  experts.py
  model.py
  requirements.txt
  data/
    mimic/
      mimiciv_3_1_hosp/
      mimiciv_3_1_icu/
      mimiciv_3_1_derived/
      supplement/
    mimic-cxr/
    mimic-note/
    pool/
  outputs/
    experts/
      ft/
      radiology_oof/
      structured_oof/
      cxr/
    final/
      final/
```

`data.py` and `experts.py` generate the following main inputs:

| Directory | Main feature, split, and prediction files |
| --- | --- |
| `data/pool/` | `selected_subject_ids.csv`, `cohort.parquet`, `cohort_split.npz`, `cohort_flags.npz`, `features_concepts.parquet`, `features_supplement.parquet`, `cxr_images.npz` |
| `outputs/experts/ft/`, `outputs/experts/cxr/` | `predictions.npz`, `run.json` |
| `outputs/experts/radiology_oof/`, `outputs/experts/structured_oof/` | `oof_readmission.npz`, `oof_icu.npz`, `oof_mortality.npz` |

`common.py` defines shared paths, cohort and split loading, and note windows. Keep the cohort, split, observation flags, and expert artifacts together; training checks patient alignment and expert training partitions, and final evaluation verifies the frozen artifact hashes.

## Prerequisites

Use a Python 3.11 environment and install the dependencies listed in `requirements.txt`. Core dependencies are:

```text
numpy
pandas
pyarrow
scipy
scikit-learn
lightgbm
joblib
torch
google-cloud-bigquery
google-cloud-bigquery-storage
db-dtypes
transformers
tensorflow
```

Install them with:

```bash
pip install -r requirements.txt
```

Text modeling loads `thomas-sounack/BioClinical-ModernBERT-base` through Hugging Face Transformers. TensorFlow is used to parse the downloaded CXR TFRecords. The discharge-summary fine-tuning stage uses CUDA and NCCL through `torchrun`; the example below uses two GPUs. Dependency versions are specified in `requirements.txt`.

## Usage

### Step 1: Data Preprocessing

After preparing the source data and Google Cloud credentials, run the preprocessing stages in order:

```bash
python data.py export --project YOUR_GCP_PROJECT_ID
python data.py select --project YOUR_GCP_PROJECT_ID
python data.py labels
python data.py cxr
python data.py flags
python data.py features
python data.py fetch --project YOUR_GCP_PROJECT_ID
python data.py supplement
```

The default configuration selects 20,000 subjects with seed 42 from adults who survived their first hospital admission. The cohort is split into 68% development training, 12% validation, and 20% test. Validation is further divided equally into selection and calibration partitions. Eligible patients outside the fixed cohort form the additional expert-training pool (`pool_extra`).

These stages will:

* Download required structured hospital, ICU, and derived tables from BigQuery.
* Select a reproducible MIMIC-IV cohort and build post-discharge outcome labels and shared patient splits.
* Read eligible MIMIC-CXR TFRecord embeddings and record image acquisition times.
* Build observation flags for structured data, imaging, discharge summaries, and radiology reports.
* Generate structured clinical features and supplementary features under `data/pool/`.

### Step 2: Train the Multimodal Models and Apply the Rectifier

First, prepare the discharge-summary token cache and train the modality experts:

```bash
python experts.py tokenize
torchrun --standalone --nproc_per_node=2 experts.py finetune --accum 4
OMP_NUM_THREADS=16 python experts.py radiology-oof
OMP_NUM_THREADS=16 python experts.py structured-oof --task readmission
OMP_NUM_THREADS=16 python experts.py structured-oof --task icu
OMP_NUM_THREADS=16 python experts.py structured-oof --task mortality
python experts.py cxr
```

Then train the shared multimodal model and fit the rectifier:

```bash
OMP_NUM_THREADS=16 python model.py fit
```

These stages will:

* Fine-tune the discharge-summary encoder and train the CXR expert on `pool_extra`, then generate predictions for the fixed cohort.
* Generate out-of-fold structured and radiology expert predictions for the development training partition, using `pool_extra` as additional training data.
* Train a shared multimodal representation with missingness embeddings, gating, multitask heads, and cross-modal latent prediction of masked modalities.
* Fit and select the pattern rectifier and calibration using validation data.
* Save `models.joblib`, `meta_predictions.npz`, `test_predictions.npz`, `report.json`, and `frozen_bundle.json` under `outputs/final/final/` with the default run name `final`.

The fitting command prints the SHA-256 release key for `outputs/final/final/frozen_bundle.json`. Use that key to run the final test evaluation:

```bash
python model.py evaluate --release YOUR_FROZEN_BUNDLE_SHA256
```

Evaluation verifies the frozen code, data, split, and model artifacts, then writes task-level metrics, bootstrap confidence intervals, and modality-pattern results to `outputs/final/final/test_report.json`. The release is recorded in `outputs/final/TEST_RELEASED.json`, and the code permits a single test release. To use a different model run name, pass `--run-name` to both `fit` and `evaluate`.
