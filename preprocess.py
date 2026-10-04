#!/usr/bin/env python3
"""Prepare both clinical cohorts and their structured, text and image inputs."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import preprocess_mortality as clinical
import preprocess_post_discharge as discharge


def main() -> None:
    root = Path(__file__).resolve().parent
    mortality = clinical.PipelineConfig(
        data_dir=str(root / "data/mortality"),
        note_dir=str(root.parent / "mimiciv_note"),
        cxr_dir=str(root.parent / "mimiciv_cxr"),
        cohort_size=20000, random_seed=42, observation_hours=24,
        protocol="icu_landmark", gcp_project="YOUR_GCP_PROJECT_ID",
        bq_page_size=50000, csv_chunksize=250000,
    )
    post = SimpleNamespace(
        data_dir=str(root / "data/post_discharge"),
        note_dir=str(root.parent / "mimiciv_note"),
        cxr_dir=str(root.parent / "mimiciv_cxr"),
        cohort_size=20000, random_seed=42, note_grace_hours=24,
        gcp_project="YOUR_GCP_PROJECT_ID", bq_page_size=50000, csv_chunksize=250000,
        text_model="emilyalsentzer/Bio_ClinicalBERT", text_batch_size=8,
        text_chunk_stride=384, note_chunksize=5000,
        text_cache_dir=str(root.parent / "mimiciv_data/text_embedding_cache"),
    )
    print("Preparing raw clinical tables", flush=True)
    clinical.download_stage(mortality)
    discharge.download_stage(post)
    print("Preparing structured features", flush=True)
    clinical.structured_stage(mortality)
    discharge.structured_stage(post)
    # The image selector needs an aligned patient-feature manifest.
    data = root / "data/mortality"
    temporary = data / "patient_features.csv.tmp"
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to("patient_features_early.csv")
    temporary.replace(data / "patient_features.csv")
    clinical.atomic_write_json(
        json.loads((data / "patient_features_early.meta.json").read_text()),
        data / "patient_features.meta.json",
    )
    print("Preparing text embeddings", flush=True)
    clinical.text_stage(mortality, post.text_model, 8, 384, 5000)
    discharge.text_stage(post)
    print("Preparing chest radiograph embeddings", flush=True)
    subprocess.run([sys.executable, str(root / "preprocess_cxr.py")], check=True)
    print("Checking prepared data", flush=True)
    clinical.validate_stage(mortality)
    discharge.validate_stage(post)


if __name__ == "__main__":
    main()
