#!/usr/bin/env python3
"""Prepare common discharge inputs for all three prediction tasks."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import preprocess_post_discharge as discharge


def main() -> None:
    root = Path(__file__).resolve().parent
    post = SimpleNamespace(
        data_dir=str(root / "data/post_discharge"),
        note_dir=str(root.parent / "mimiciv_note"),
        cxr_dir=str(root.parent / "mimiciv_cxr"),
        cohort_size=20000, random_seed=42, note_grace_hours=24,
        gcp_project="mimic-project-01", bq_page_size=50000, csv_chunksize=250000,
        text_model="emilyalsentzer/Bio_ClinicalBERT", text_batch_size=8,
        text_chunk_stride=384, note_chunksize=5000,
        text_cache_dir=str(root.parent / "mimiciv_data/text_embedding_cache"),
    )
    print("Preparing raw clinical tables", flush=True)
    discharge.download_stage(post)
    print("Preparing structured features", flush=True)
    discharge.structured_stage(post)
    print("Preparing text embeddings", flush=True)
    discharge.text_stage(post)
    print("Preparing chest radiograph embeddings", flush=True)
    subprocess.run([sys.executable, str(root / "preprocess_cxr.py")], check=True)
    print("Checking prepared data", flush=True)
    discharge.validate_stage(post)


if __name__ == "__main__":
    main()
