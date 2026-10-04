#!/usr/bin/env python
"""Select available chest radiographs and pool embeddings per patient."""

import csv
import hashlib
import json
import os
import re
from typing import List, Optional, Tuple

# TensorFlow is used only as a TFRecord parser. SLURM controls device
# visibility; do not override its assignment from inside the process.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf


METADATA_FILENAMES = (
    "mimic-cxr-2.0.0-metadata.csv.gz",
    "mimic-cxr-2.0.0-metadata.csv",
)
SPLIT_FILENAMES = (
    "mimic-cxr-2.0.0-split.csv.gz",
    "mimic-cxr-2.0.0-split.csv",
)


def require_nonempty_file(path: str, description: str) -> None:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"Missing {description}: {path}")


def find_asset(
    embedding_dir: str,
    candidate_names: Tuple[str, ...],
    explicit_path: Optional[str],
    description: str,
) -> str:
    if explicit_path is not None:
        require_nonempty_file(explicit_path, description)
        return explicit_path

    for root, _, files in os.walk(embedding_dir):
        for name in candidate_names:
            if name in files:
                return os.path.join(root, name)

    raise FileNotFoundError(
        f"Missing {description}. Place one of {candidate_names} under "
        f"{embedding_dir}, or pass its path explicitly."
    )


def load_prediction_subject_ids(data_dir: str) -> List[int]:
    cohort_path = os.path.join(data_dir, "analysis_cohort.csv")
    features_path = os.path.join(data_dir, "patient_features.csv")
    selected_path = os.path.join(data_dir, "selected_subject_ids.csv")
    meta_path = os.path.join(data_dir, "patient_features.meta.json")
    for path, description in (
        (cohort_path, "ICU landmark cohort manifest"),
        (features_path, "structured patient features"),
        (selected_path, "selected subject list"),
        (meta_path, "structured-stage metadata"),
    ):
        require_nonempty_file(path, description)
    selected_meta_path = f"{selected_path[:-4]}.meta.json"
    require_nonempty_file(selected_meta_path, "selected-subject metadata")
    with open(meta_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with open(selected_meta_path, "r", encoding="utf-8") as handle:
        selected_metadata = json.load(handle)
    supported_pipelines = {"icu_landmark", "post_discharge"}
    if (
        metadata.get("pipeline") not in supported_pipelines
        or selected_metadata.get("pipeline") != metadata.get("pipeline")
        or metadata.get("cohort_fingerprint") != selected_metadata.get("cohort_fingerprint")
    ):
        raise ValueError(
            "Structured-stage metadata is stale or incompatible. Run "
            "preprocess.sh before model training."
        )

    manifests = {}
    for name, path in (("cohort", cohort_path), ("features", features_path), ("selected", selected_path)):
        values = pd.read_csv(path, usecols=["subject_id"])["subject_id"].dropna().astype(int)
        if values.duplicated().any():
            raise ValueError(f"{path} contains duplicate subject IDs")
        manifests[name] = set(values)
    if not (manifests["cohort"] == manifests["features"] == manifests["selected"]):
        raise ValueError(
            "selected_subject_ids.csv, analysis_cohort.csv, and structured features "
            "do not describe the same cohort; rerun the structured stage."
        )
    subject_ids = sorted(manifests["cohort"])
    if not subject_ids:
        raise ValueError(f"No subject IDs found in {cohort_path}")
    return subject_ids


def load_cohort(
    data_dir: str,
    subject_ids: List[int],
) -> pd.DataFrame:
    cohort_path = os.path.join(data_dir, "analysis_cohort.csv")
    require_nonempty_file(cohort_path, "ICU landmark cohort manifest")

    cohort = pd.read_csv(
        cohort_path,
        usecols=["subject_id", "admittime", "dischtime", "intime", "prediction_cutoff"],
        parse_dates=["admittime", "dischtime", "intime", "prediction_cutoff"],
    )
    cohort["subject_id"] = pd.to_numeric(
        cohort["subject_id"], errors="coerce"
    ).astype("Int64")
    cohort = cohort.dropna(subset=["subject_id"]).copy()
    cohort["subject_id"] = cohort["subject_id"].astype(int)

    subject_set = set(subject_ids)
    cohort = cohort[cohort["subject_id"].isin(subject_set)].copy()
    if cohort["subject_id"].duplicated().any():
        raise ValueError("analysis_cohort.csv must contain one row per subject")

    missing_subjects = subject_set - set(cohort["subject_id"])
    if missing_subjects:
        raise ValueError(
            f"analysis_cohort.csv is missing {len(missing_subjects)} prediction subjects"
        )
    if cohort[["admittime", "dischtime", "intime", "prediction_cutoff"]].isna().any().any():
        raise ValueError("Cohort contains missing ICU observation-window timestamps")

    return cohort.sort_values("subject_id").reset_index(drop=True)


def parse_study_datetime(study_date, study_time) -> pd.Timestamp:
    if pd.isna(study_date) or pd.isna(study_time):
        return pd.NaT
    date_digits = re.sub(r"\D", "", str(study_date).split(".")[0])
    time_digits = re.sub(r"\D", "", str(study_time).split(".")[0]).zfill(6)[:6]
    if len(date_digits) != 8 or len(time_digits) != 6:
        return pd.NaT
    return pd.to_datetime(
        date_digits + time_digits,
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )


def load_cxr_records(
    embedding_dir: str,
    metadata_path: Optional[str],
    split_path: Optional[str],
    frontal_only: bool = False,
) -> pd.DataFrame:
    metadata_path = find_asset(
        embedding_dir,
        METADATA_FILENAMES,
        metadata_path,
        "MIMIC-CXR acquisition metadata",
    )
    print(f"Using CXR metadata: {metadata_path}")
    metadata = pd.read_csv(metadata_path, low_memory=False)
    metadata.columns = [str(column).strip() for column in metadata.columns]

    metadata_required = {"dicom_id"}
    if frontal_only:
        metadata_required.add("ViewPosition")
    missing_metadata = sorted(metadata_required - set(metadata.columns))
    if missing_metadata:
        raise ValueError(
            f"CXR metadata is missing required columns: {missing_metadata}"
        )

    identifier_columns = {"dicom_id", "subject_id", "study_id"}
    if not identifier_columns.issubset(metadata.columns):
        split_path = find_asset(
            embedding_dir,
            SPLIT_FILENAMES,
            split_path,
            "MIMIC-CXR subject/study split mapping",
        )
        print(f"Using CXR subject/study mapping: {split_path}")
        split = pd.read_csv(split_path, low_memory=False)
        split.columns = [str(column).strip() for column in split.columns]
        missing_split = sorted(identifier_columns - set(split.columns))
        if missing_split:
            raise ValueError(f"CXR split file is missing required columns: {missing_split}")

        metadata = metadata.merge(
            split[["dicom_id", "subject_id", "study_id"]].drop_duplicates(),
            on="dicom_id",
            how="left",
            validate="one_to_one",
        )

    if frontal_only:
        view = metadata["ViewPosition"].astype("string").str.strip().str.upper()
        # MIMIC-CXR documents AP and PA as the canonical frontal projections.
        metadata = metadata[view.isin({"AP", "PA"})].copy()
        print(f"Retained {len(metadata):,} AP/PA frontal CXR records")

    timeline_columns = ["dicom_id", "subject_id", "study_id"]
    if {"StudyDate", "StudyTime"}.issubset(metadata.columns):
        timeline_columns.extend(["StudyDate", "StudyTime"])
    timeline = metadata[timeline_columns].copy()
    timeline["subject_id"] = pd.to_numeric(
        timeline["subject_id"], errors="coerce"
    ).astype("Int64")
    timeline["study_id"] = pd.to_numeric(
        timeline["study_id"], errors="coerce"
    ).astype("Int64")
    timeline["dicom_id"] = timeline["dicom_id"].astype(str)
    if {"StudyDate", "StudyTime"}.issubset(timeline.columns):
        timeline["study_datetime"] = [
            parse_study_datetime(date, time)
            for date, time in zip(timeline["StudyDate"], timeline["StudyTime"])
        ]
    else:
        timeline["study_datetime"] = pd.NaT
    timeline = timeline.dropna(subset=["subject_id", "study_id"])
    print(f"Loaded {len(timeline):,} CXR records")
    return timeline[["dicom_id", "subject_id", "study_id", "study_datetime"]]


def subject_folder(embedding_dir: str, subject_id: int) -> str:
    prefix = subject_id // 1_000_000
    return os.path.join(embedding_dir, f"p{prefix}", f"p{subject_id}")


def read_embedding_from_tfrecord(path: str) -> Optional[np.ndarray]:
    try:
        dataset = tf.data.TFRecordDataset([path])
        feature_spec = {"embedding": tf.io.VarLenFeature(tf.float32)}
        for raw_record in dataset.take(1):
            example = tf.io.parse_single_example(raw_record, feature_spec)
            return tf.sparse.to_dense(example["embedding"]).numpy()
    except Exception as exc:
        print(f"Warning: could not read {path}: {exc}")
    return None


def patient_records(
    cohort: pd.DataFrame,
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    timeline = timeline.merge(
        cohort[["subject_id", "admittime", "dischtime", "intime", "prediction_cutoff"]],
        on="subject_id",
        how="inner",
        validate="many_to_one",
    )
    return timeline


def extract_subject_embeddings(
    cohort: pd.DataFrame,
    timeline: pd.DataFrame,
    embedding_dir: str,
) -> Tuple[List[Tuple[int, List[List[float]]]], int]:
    columns = ["dicom_id", "subject_id", "study_id", "study_datetime"]
    timeline = patient_records(cohort, timeline[columns])
    eligible_by_subject = {
        int(subject_id): set(
            zip(group["study_id"].astype(int), group["dicom_id"].astype(str))
        )
        for subject_id, group in timeline.groupby("subject_id")
    }
    print(
        f"Found {len(timeline):,} images for "
        f"{len(eligible_by_subject):,} cohort subjects"
    )

    rows: List[Tuple[int, List[List[float]]]] = []
    max_files = 0
    expected_dimension: Optional[int] = None
    subject_ids = cohort["subject_id"].astype(int).tolist()
    print(f"Scanning TFRecords for {len(subject_ids):,} cohort subjects")

    for index, subject_id in enumerate(subject_ids):
        if index % 100 == 0:
            print(f"Progress: {index}/{len(subject_ids)} subjects processed")

        eligible_images = eligible_by_subject.get(subject_id, set())
        if not eligible_images:
            continue

        folder = subject_folder(embedding_dir, subject_id)
        if not os.path.isdir(folder):
            continue

        embeddings: List[List[float]] = []
        for study_name in os.listdir(folder):
            study_dir = os.path.join(folder, study_name)
            if not os.path.isdir(study_dir):
                continue
            study_text = (
                study_name[1:] if study_name.lower().startswith("s") else study_name
            )
            try:
                study_id = int(study_text)
            except ValueError:
                continue

            for filename in os.listdir(study_dir):
                if not filename.endswith(".tfrecord"):
                    continue
                dicom_id = filename[: -len(".tfrecord")]
                if (study_id, dicom_id) not in eligible_images:
                    continue

                path = os.path.join(study_dir, filename)
                vector = read_embedding_from_tfrecord(path)
                if vector is None:
                    continue
                vector = np.asarray(vector, dtype=np.float32).reshape(-1)
                if expected_dimension is None:
                    expected_dimension = int(vector.size)
                    print(f"Detected TFRecord embedding dimension: {expected_dimension}")
                elif vector.size != expected_dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch in {path}: expected "
                        f"{expected_dimension}, found {vector.size}"
                    )
                embeddings.append(vector.tolist())

        if embeddings:
            rows.append((subject_id, embeddings))
            max_files = max(max_files, len(embeddings))

    print(f"Extracted embeddings for {len(rows):,} subjects")
    print(f"Maximum eligible CXR files per subject: {max_files}")
    return rows, max_files


def atomic_replace(temp_path: str, final_path: str) -> None:
    os.replace(temp_path, final_path)


def save_intermediate_embeddings(
    rows: List[Tuple[int, List[List[float]]]],
    max_files: int,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_path = f"{output_path}.tmp"
    column_names = ["subject_id"] + [f"file_{index}" for index in range(max_files)]

    with open(temp_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(column_names)
        for subject_id, embeddings in rows:
            cells = [
                "[" + ",".join(f"{value:.6f}" for value in vector) + "]"
                for vector in embeddings
            ]
            writer.writerow([subject_id] + cells + [""] * (max_files - len(cells)))

    atomic_replace(temp_path, output_path)
    print(f"Intermediate embeddings saved to: {output_path}")


def aggregate_embeddings(input_csv: str, output_csv: str) -> int:
    """Mean-pool file-level embeddings without importing CUDA-enabled PyTorch."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    temp_path = f"{output_csv}.tmp"
    subject_count = 0
    expected_dimension: Optional[int] = None

    with open(input_csv, "r", newline="", encoding="utf-8") as input_file, open(
        temp_path, "w", newline="", encoding="utf-8"
    ) as output_file:
        reader = csv.reader(input_file)
        next(reader, None)
        writer = csv.writer(output_file)
        writer.writerow(["subject_id", "agg_embedding"])

        for row in reader:
            if not row:
                continue
            parsed = [json.loads(cell) for cell in row[1:] if cell.strip()]
            if not parsed:
                continue

            embeddings = np.asarray(parsed, dtype=np.float32)
            if embeddings.ndim != 2:
                raise ValueError(
                    f"Malformed embeddings for subject {row[0]}: "
                    f"expected a 2-D array, found shape {embeddings.shape}"
                )
            if expected_dimension is None:
                expected_dimension = int(embeddings.shape[1])
                print(f"Aggregating embeddings at dimension {expected_dimension}")
            elif embeddings.shape[1] != expected_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch for subject {row[0]}: expected "
                    f"{expected_dimension}, found {embeddings.shape[1]}"
                )

            vector = embeddings.mean(axis=0, dtype=np.float32).tolist()
            writer.writerow([row[0], json.dumps(vector)])
            subject_count += 1
            if subject_count % 100 == 0:
                print(f"Aggregation progress: {subject_count} subjects")

    if subject_count == 0:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError("No valid embeddings found in the intermediate CSV")

    atomic_replace(temp_path, output_csv)
    print(f"Aggregated embeddings saved to: {output_csv}")
    return subject_count


def prepare_inputs(args):
    print("=== CXR-only Embedding Selection and Aggregation ===")
    subject_ids = load_prediction_subject_ids(args.data_dir)
    cohort = load_cohort(args.data_dir, subject_ids)
    print(f"Validated cohort: {len(cohort):,} subjects")

    records = load_cxr_records(
        args.embedding_dir,
        args.metadata_path,
        args.split_path,
        args.frontal_only,
    )
    return cohort, records


def write_embeddings(args, cohort, records) -> None:
    rows, max_files = extract_subject_embeddings(
        cohort,
        records,
        args.embedding_dir,
    )
    if not rows:
        raise RuntimeError(
            "No eligible CXR embeddings were found. Check image metadata, "
            "cohort dates, and TFRecord directory layout."
        )

    save_intermediate_embeddings(rows, max_files, args.intermediate_csv)
    aggregated_subjects = aggregate_embeddings(args.intermediate_csv, args.output_csv)
    structured_meta_path = os.path.join(args.data_dir, "patient_features.meta.json")
    with open(structured_meta_path, "r", encoding="utf-8") as handle:
        structured_metadata = json.load(handle)
    cohort_payload = cohort[
        ["subject_id", "admittime", "dischtime", "intime", "prediction_cutoff"]
    ].to_csv(index=False).encode("utf-8")
    output_meta = os.path.splitext(args.output_csv)[0] + ".meta.json"
    temp_meta = f"{output_meta}.tmp"
    payload = {
        "pipeline": structured_metadata.get("pipeline"),
        "cohort_fingerprint": structured_metadata.get("cohort_fingerprint"),
        "frontal_only": bool(args.frontal_only),
        "cohort_hash": hashlib.sha256(cohort_payload).hexdigest(),
        "cohort_subjects": len(cohort),
        "embedded_subjects": aggregated_subjects,
    }
    if structured_metadata.get("pipeline") == "icu_landmark":
        payload["window"] = "landmark" if "cxr_embeddings_early" in os.path.basename(args.output_csv) else "hospital"
    with open(temp_meta, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temp_meta, output_meta)
    print("=== CXR-only pipeline completed successfully ===")
    print(f"Subjects with aggregated CXR embeddings: {aggregated_subjects:,}")


def build_embeddings(args) -> None:
    cohort, records = prepare_inputs(args)
    write_embeddings(args, cohort, records)


def build_mortality_embeddings(args) -> None:
    from preprocess_mortality import cxr_views
    from types import SimpleNamespace
    cohort, records = prepare_inputs(args)
    if (cohort.prediction_cutoff < cohort.intime).any():
        raise ValueError("Cohort contains a prediction cutoff before ICU intime")
    for stem, selected in cxr_views(cohort, records).items():
        output = SimpleNamespace(**vars(args))
        output.intermediate_csv = os.path.join(args.data_dir, f"{stem}_by_file.csv")
        output.output_csv = os.path.join(args.data_dir, f"{stem}_aggregated.csv")
        write_embeddings(output, cohort, selected)



def main() -> None:
    from pathlib import Path
    from types import SimpleNamespace
    root = Path(__file__).resolve().parent
    for cohort in ("mortality", "post_discharge"):
        data = root / "data" / cohort
        args = SimpleNamespace(
            data_dir=str(data), embedding_dir=str(root.parent / "mimiciv_cxr"),
            intermediate_csv=str(data / "cxr_embeddings_by_file.csv"),
            output_csv=str(data / "cxr_embeddings_aggregated.csv"),
            metadata_path=None, split_path=None, frontal_only=True,
        )
        if cohort == "mortality":
            build_mortality_embeddings(args)
        else:
            build_embeddings(args)


if __name__ == "__main__":
    main()
