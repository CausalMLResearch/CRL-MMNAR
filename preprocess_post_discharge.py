#!/usr/bin/env python3
"""Hospital-discharge preprocessing for readmission and future ICU outcomes."""

from __future__ import annotations

from types import SimpleNamespace
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import preprocess_common as core


PIPELINE = "post_discharge"
STRUCTURED_PIPELINE = "post_discharge_dual_view"
TEXT_PIPELINE = "post_discharge_dual_view"
HOSP_TABLES = (
    "admissions", "patients", "diagnoses_icd", "procedures_icd", "drgcodes",
    "labevents", "microbiologyevents", "prescriptions", "emar", "services",
    "transfers",
)
ICU_TABLES = ("icustays", "chartevents", "inputevents", "outputevents")


def cohort_payload(args: SimpleNamespace) -> Dict[str, object]:
    return {
        "pipeline": PIPELINE,
        "cohort_size": int(args.cohort_size),
        "random_seed": int(args.random_seed),
        "prediction_landmark": "index_discharge",
        "complete_note_grace_hours": int(args.note_grace_hours),
        "selection": (
            "adult first hospital admission per subject; survived index admission; "
            "no post-discharge washout or outcome-dependent exclusion"
        ),
    }


def cohort_fingerprint(args: SimpleNamespace) -> str:
    return "fe1afb2eab57ba75"


def metadata(args: SimpleNamespace) -> Dict[str, object]:
    return {**cohort_payload(args), "cohort_fingerprint": cohort_fingerprint(args)}


def table_path(args: SimpleNamespace, module: str, table: str) -> Path:
    return Path(args.data_dir) / f"{module}_{table}.csv"


def clinical_config(args: SimpleNamespace) -> core.PipelineConfig:
    return core.PipelineConfig(
        data_dir=args.data_dir, note_dir=args.note_dir, cxr_dir=args.cxr_dir,
        cohort_size=args.cohort_size, random_seed=args.random_seed,
        # Urine output is normalized by hospital duration below.
        observation_hours=1, protocol="post_discharge",
        gcp_project=args.gcp_project, bq_page_size=args.bq_page_size,
        csv_chunksize=args.csv_chunksize,
    )


def download_stage(args: SimpleNamespace) -> None:
    if args.note_grace_hours < 0:
        raise ValueError("note-grace-hours must be non-negative")
    from google.cloud import bigquery

    client = bigquery.Client(project=args.gcp_project) if args.gcp_project else bigquery.Client()
    output = Path(args.data_dir)
    output.mkdir(parents=True, exist_ok=True)
    ids_path = output / "selected_subject_ids.csv"
    ids_meta_path = output / "selected_subject_ids.meta.json"
    expected = metadata(args)
    cached = False
    if ids_path.is_file() and ids_meta_path.is_file():
        try:
            cached = json.loads(ids_meta_path.read_text(encoding="utf-8")) == expected
        except json.JSONDecodeError:
            cached = False

    if not cached:
        sql = """
        WITH ranked AS (
          SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime,
                 a.hospital_expire_flag, p.anchor_age, p.anchor_year,
                 p.anchor_year_group,
                 ROW_NUMBER() OVER (
                   PARTITION BY a.subject_id ORDER BY a.admittime, a.hadm_id
                 ) AS rn
          FROM `physionet-data.mimiciv_3_1_hosp.admissions` a
          JOIN `physionet-data.mimiciv_3_1_hosp.patients` p USING(subject_id)
          WHERE a.admittime IS NOT NULL AND a.dischtime IS NOT NULL
        ), first_admission AS (
          SELECT * FROM ranked WHERE rn = 1
        )
        SELECT f.subject_id
        FROM first_admission f
        WHERE f.anchor_age + EXTRACT(YEAR FROM f.admittime) - f.anchor_year >= 18
          AND f.anchor_year_group IN ('2008 - 2010', '2011 - 2013', '2014 - 2016', '2017 - 2019')
          AND COALESCE(f.hospital_expire_flag, 0) = 0
        ORDER BY FARM_FINGERPRINT(CONCAT(CAST(f.subject_id AS STRING), '-', CAST(@seed AS STRING)))
        LIMIT @cohort_size
        """
        job = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("seed", "INT64", args.random_seed),
            bigquery.ScalarQueryParameter("cohort_size", "INT64", args.cohort_size),
        ])
        ids = client.query(sql, job_config=job).to_dataframe()[["subject_id"]]
        if len(ids) != args.cohort_size or ids.subject_id.duplicated().any():
            raise RuntimeError(f"Expected {args.cohort_size} unique post-discharge subjects, found {len(ids)}")
        tmp = ids_path.with_suffix(".csv.tmp")
        ids.to_csv(tmp, index=False)
        os.replace(tmp, ids_path)
        core.atomic_write_json(expected, ids_meta_path)
    else:
        ids = pd.read_csv(ids_path, usecols=["subject_id"])

    subject_ids = ids.subject_id.astype(int).tolist()
    job = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("subject_ids", "INT64", subject_ids),
    ])
    for module, tables in (("hosp", HOSP_TABLES), ("icu", ICU_TABLES)):
        for table in tables:
            path = table_path(args, module, table)
            meta_path = Path(str(path) + ".cohort.json")
            valid = False
            if path.is_file() and meta_path.is_file():
                try:
                    valid = json.loads(meta_path.read_text(encoding="utf-8")) == expected
                except json.JSONDecodeError:
                    valid = False
            if valid:
                print(f"Using cached {path}")
                continue
            sql = f"SELECT * FROM `physionet-data.mimiciv_3_1_{module}.{table}` WHERE subject_id IN UNNEST(@subject_ids)"
            count = core.stream_query(client, sql, path, job, args.bq_page_size)
            core.atomic_write_json(expected, meta_path)
            print(f"Downloaded {table}: {count:,} rows")

    for module, tables in (("hosp", core.HOSP_DICTIONARIES), ("icu", core.ICU_DICTIONARIES)):
        for table in tables:
            path = table_path(args, module, table)
            if path.is_file() and path.stat().st_size > 0:
                continue
            count = core.stream_query(
                client, f"SELECT * FROM `physionet-data.mimiciv_3_1_{module}.{table}`",
                path, None, args.bq_page_size,
            )
            print(f"Downloaded dictionary {table}: {count:,} rows")


def verify_raw_metadata(args: SimpleNamespace) -> None:
    expected = metadata(args)
    paths = [Path(args.data_dir) / "selected_subject_ids.meta.json"] + [
        Path(str(table_path(args, module, table)) + ".cohort.json")
        for module, tables in (("hosp", HOSP_TABLES), ("icu", ICU_TABLES))
        for table in tables
    ]
    for path in paths:
        actual = json.loads(core.require_file(path).read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"Stale post-discharge raw artifact: {path}")


def build_cohort(args: SimpleNamespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    verify_raw_metadata(args)
    ids = pd.read_csv(core.require_file(Path(args.data_dir) / "selected_subject_ids.csv"), usecols=["subject_id"])
    selected = set(ids.subject_id.astype(int))
    admissions = pd.read_csv(
        core.require_file(table_path(args, "hosp", "admissions")),
        parse_dates=["admittime", "dischtime", "deathtime"], low_memory=False,
    )
    patients = pd.read_csv(core.require_file(table_path(args, "hosp", "patients")), low_memory=False)
    icu = pd.read_csv(
        core.require_file(table_path(args, "icu", "icustays")),
        parse_dates=["intime", "outtime"], low_memory=False,
    )
    admissions = admissions[admissions.subject_id.isin(selected)].copy()
    patients = patients[patients.subject_id.isin(selected)].copy()
    icu = icu[icu.subject_id.isin(selected)].copy()
    index = admissions.sort_values(["subject_id", "admittime", "hadm_id"]).drop_duplicates("subject_id", keep="first")
    if set(index.subject_id.astype(int)) != selected:
        raise RuntimeError("Downloaded admissions do not cover the selected post-discharge cohort")
    if index.hospital_expire_flag.fillna(0).astype(int).any():
        raise RuntimeError("Post-discharge cohort contains an index-admission death")

    first_icu = (
        icu.merge(index[["subject_id", "hadm_id"]], on=["subject_id", "hadm_id"], how="inner")
        .sort_values(["subject_id", "intime", "stay_id"])
        .drop_duplicates("subject_id", keep="first")
        [["subject_id", "stay_id", "intime", "outtime", "first_careunit"]]
    )
    cohort = index.merge(first_icu, on="subject_id", how="left", validate="one_to_one")
    cohort["has_index_icu"] = cohort.stay_id.notna().astype(np.int8)
    cohort["stay_id"] = cohort.stay_id.fillna(-1).astype(np.int64)
    cohort["intime"] = cohort.intime.fillna(cohort.admittime)
    cohort["outtime"] = cohort.outtime.fillna(cohort.dischtime)
    cohort["first_careunit"] = cohort.first_careunit.fillna("NO_INDEX_ICU")
    patient_age = patients.set_index("subject_id")
    cohort["age_at_admission"] = (
        cohort.subject_id.map(patient_age.anchor_age)
        + cohort.admittime.dt.year - cohort.subject_id.map(patient_age.anchor_year)
    )
    cohort["prediction_cutoff"] = cohort.dischtime
    cohort["clinical_cutoff"] = cohort.dischtime

    future = admissions.merge(
        cohort[["subject_id", "hadm_id", "dischtime"]].rename(
            columns={"hadm_id": "index_hadm_id", "dischtime": "index_dischtime"}
        ),
        on="subject_id", validate="many_to_one",
    )
    delta = (future.admittime - future.index_dischtime).dt.total_seconds() / 86400.0
    future = future[(future.hadm_id != future.index_hadm_id) & delta.gt(0)].copy()
    future["days_after_discharge"] = delta[delta.gt(0)]
    readmit_ids = set(future.loc[future.days_after_discharge.le(30), "subject_id"].astype(int))
    readmit_90_ids = set(future.loc[future.days_after_discharge.le(90), "subject_id"].astype(int))
    # Match ICU stays to a genuine post-discharge admission, as in the official
    # task definition. This excludes anomalous/overlapping index-HADM ICU rows.
    future_icu = icu.merge(
        future[["subject_id", "hadm_id", "index_dischtime"]].drop_duplicates(),
        on=["subject_id", "hadm_id"], how="inner", validate="many_to_one",
    )
    icu_delta = (future_icu.intime - future_icu.index_dischtime).dt.total_seconds() / 86400.0
    icu_ids = set(future_icu.loc[icu_delta.gt(0) & icu_delta.le(90), "subject_id"].astype(int))
    cohort["readmission_30d"] = cohort.subject_id.astype(int).isin(readmit_ids).astype(np.int8)
    # Auxiliary outcome used only to express the clinical relationship
    # P(ICU within 90d) = P(readmission within 90d) * P(ICU | readmission).
    # It is never used as an input feature.
    cohort["readmission_90d"] = cohort.subject_id.astype(int).isin(readmit_90_ids).astype(np.int8)
    cohort["icu_need_after_discharge_90d"] = cohort.subject_id.astype(int).isin(icu_ids).astype(np.int8)
    cohort["in_hospital_mortality"] = np.int8(0)
    cohort["post_discharge_observed"] = np.int8(1)
    if (cohort.icu_need_after_discharge_90d > cohort.readmission_90d).any():
        raise RuntimeError("An ICU-after-discharge outcome has no matching 90-day readmission")
    if len(cohort) != args.cohort_size or cohort.subject_id.duplicated().any():
        raise RuntimeError("Post-discharge cohort lost its one-row-per-subject invariant")
    return cohort.sort_values("subject_id").reset_index(drop=True), admissions, patients, icu


def hashed_index_counts(
    args: SimpleNamespace, cohort: pd.DataFrame, table: str, value_columns: Sequence[str],
    prefix: str, buckets: int = 64,
) -> pd.DataFrame:
    keys = cohort[["subject_id", "hadm_id"]]
    states: Dict[int, np.ndarray] = {}
    totals: Dict[int, int] = {}
    unique_values: Dict[int, set[str]] = {}
    usecols = ["subject_id", "hadm_id", *value_columns]
    for chunk in pd.read_csv(
        core.require_file(table_path(args, "hosp", table)), usecols=usecols,
        chunksize=args.csv_chunksize, low_memory=False,
    ):
        chunk = chunk.merge(keys, on=["subject_id", "hadm_id"], how="inner", validate="many_to_one")
        if chunk.empty:
            continue
        token = chunk[list(value_columns)].astype("string").fillna("UNKNOWN").agg(": ".join, axis=1)
        chunk = chunk.assign(_token=token)
        chunk["_bucket"] = chunk._token.map(lambda value: core.stable_bucket(str(value), buckets))
        for subject_id, group in chunk.groupby("subject_id"):
            sid = int(subject_id)
            totals[sid] = totals.get(sid, 0) + len(group)
            unique_values.setdefault(sid, set()).update(group._token.astype(str))
            counts = np.bincount(group._bucket.to_numpy(dtype=int), minlength=buckets)
            states[sid] = states.get(sid, np.zeros(buckets, dtype=np.int64)) + counts
    columns: Dict[str, pd.Series] = {
        f"idx_{prefix}_count": pd.Series(totals, dtype=float),
        f"idx_{prefix}_unique": pd.Series(
            {sid: len(values) for sid, values in unique_values.items()}, dtype=float,
        ),
    }
    for bucket in range(buckets):
        columns[f"idx_{prefix}_hash_{bucket:02d}"] = pd.Series(
            {sid: int(values[bucket]) for sid, values in states.items()}, dtype=float,
        )
    result = pd.DataFrame(columns, index=cohort.subject_id.astype(int))
    result.index.name = "subject_id"
    return result


def index_administrative_features(args: SimpleNamespace, cohort: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=cohort.subject_id.astype(int)); result.index.name = "subject_id"
    result = result.join(hashed_index_counts(args, cohort, "diagnoses_icd", ("icd_version", "icd_code"), "dx", 128))
    result = result.join(hashed_index_counts(args, cohort, "procedures_icd", ("icd_version", "icd_code"), "proc", 96))
    result = result.join(hashed_index_counts(args, cohort, "drgcodes", ("drg_type", "drg_code"), "drg", 64))
    result = result.join(hashed_index_counts(args, cohort, "prescriptions", ("drug_type", "drug"), "rx", 128))
    result = result.join(hashed_index_counts(args, cohort, "services", ("curr_service",), "service", 32))
    result = result.join(hashed_index_counts(args, cohort, "transfers", ("eventtype", "careunit"), "transfer", 64))

    keys = cohort[["subject_id", "hadm_id"]]
    drg = pd.read_csv(
        core.require_file(table_path(args, "hosp", "drgcodes")),
        usecols=["subject_id", "hadm_id", "drg_severity", "drg_mortality"], low_memory=False,
    ).merge(keys, on=["subject_id", "hadm_id"], how="inner", validate="many_to_one")
    if not drg.empty:
        severity = drg.groupby("subject_id").agg(
            idx_drg_severity_max=("drg_severity", "max"),
            idx_drg_mortality_max=("drg_mortality", "max"),
        )
        result = result.join(severity)
    return result


def build_structured_view(
    args: SimpleNamespace, cohort: pd.DataFrame, admissions: pd.DataFrame,
    patients: pd.DataFrame, icu: pd.DataFrame, view: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if view != "complete":
        raise ValueError(view)
    cfg = clinical_config(args)
    window = cohort.copy()
    window["window_start"] = window.admittime
    window["clinical_cutoff"] = window.dischtime
    window["prediction_cutoff"] = window.dischtime
    # Keep index-admission events through discharge despite delayed entry.
    window["enforce_store_cutoff"] = False
    window["include_all_stays"] = True
    indexed = cohort.set_index("subject_id")
    base = indexed[["age_at_admission", "has_index_icu"]].copy()
    base["idx_hospital_duration_hours"] = (
        (indexed.dischtime - indexed.admittime).dt.total_seconds().clip(lower=0) / 3600.0
    )
    bounds = cohort[["subject_id", "hadm_id", "admittime", "dischtime"]]
    stays = icu.merge(bounds, on=["subject_id", "hadm_id"], how="inner", validate="many_to_one")
    interval_start = pd.concat([stays.intime, stays.admittime], axis=1).max(axis=1)
    interval_end = pd.concat([stays.outtime, stays.dischtime], axis=1).min(axis=1)
    stays["_duration_hours"] = (interval_end - interval_start).dt.total_seconds().clip(lower=0) / 3600.0
    base["idx_icu_duration_hours"] = stays.groupby("subject_id")._duration_hours.sum()
    base["idx_icu_stay_count"] = stays.groupby("subject_id").stay_id.nunique()

    categorical = [
        "first_careunit", "admission_type", "admission_location", "insurance",
        "language", "marital_status", "race", "discharge_location",
    ]
    demographics = cohort[["subject_id", *categorical]].merge(
        patients[["subject_id", "gender", "anchor_year_group"]], on="subject_id", validate="one_to_one",
    ).set_index("subject_id")
    categorical += ["gender", "anchor_year_group"]
    demographics = pd.get_dummies(
        demographics.astype("string").fillna("UNKNOWN"), columns=categorical,
        prefix=[f"idx_{name}" for name in categorical], dtype=np.int8,
    )
    labels = indexed[[
        "readmission_30d", "readmission_90d", "icu_need_after_discharge_90d",
        "in_hospital_mortality", "post_discharge_observed",
    ]]
    print(f"Aggregating post-discharge {view} labs")
    labs = core.aggregate_mapped_numeric(table_path(args, "hosp", "labevents"), window, core.LAB_SPECS, args.csv_chunksize, "hosp")
    print(f"Aggregating post-discharge {view} vitals")
    vitals = core.aggregate_mapped_numeric(table_path(args, "icu", "chartevents"), window, core.VITAL_SPECS, args.csv_chunksize, "icu")
    support = core.aggregate_binary_and_counts(cfg, window)
    medications = core.aggregate_medications(cfg, window)
    microbiology = core.aggregate_microbiology(cfg, window)
    history = core.aggregate_history(cfg, cohort, admissions)
    interactions = core.derive_clinical_interactions(window, labs, vitals, support, 1)
    duration = ((window.set_index("subject_id").dischtime - window.set_index("subject_id").admittime).dt.total_seconds() / 3600.0).clip(lower=1)
    if "idx_urine_output_sum" in support:
        interactions["idx_urine_output_per_hour"] = pd.to_numeric(
            support.idx_urine_output_sum, errors="coerce",
        ).reindex(duration.index) / duration
    frames = [labels, demographics, labs, vitals, support, medications, microbiology, history, interactions]
    groups = {
        "labs": len(labs.columns), "vitals": len(vitals.columns), "support": len(support.columns),
        "medications": len(medications.columns), "microbiology": len(microbiology.columns),
        "history": len(history.columns), "clinical_interactions": len(interactions.columns),
    }
    if view == "complete":
        administrative = index_administrative_features(args, cohort)
        frames.append(administrative)
        groups["index_administrative"] = len(administrative.columns)
    features = base.join(frames, how="left")
    count_like = [
        column for column in features
        if column.endswith("_count") or "_hash_" in column or column.endswith("_used")
        or column.endswith("_present") or column.endswith("_observed")
    ]
    features[count_like] = features[count_like].fillna(0)
    if features.index.duplicated().any() or len(features) != len(cohort):
        raise RuntimeError(f"{view} view lost post-discharge subjects")
    forbidden = [column for column in features if any(token in column.lower() for token in ("deathtime", "hospital_expire_flag"))]
    if forbidden:
        raise RuntimeError(f"Post-outcome feature columns detected: {forbidden}")
    return features, groups


def structured_stage(args: SimpleNamespace) -> None:
    cohort, admissions, patients, icu = build_cohort(args)
    output = Path(args.data_dir); output.mkdir(parents=True, exist_ok=True)
    cohort_path = output / "analysis_cohort.csv"
    tmp = cohort_path.with_suffix(".csv.tmp")
    cohort.to_csv(tmp, index=False); os.replace(tmp, cohort_path)
    for view in ("complete",):
        features, groups = build_structured_view(args, cohort, admissions, patients, icu, view)
        stem = f"patient_features_discharge_{view}"
        path = output / f"{stem}.csv"; tmp = path.with_suffix(".csv.tmp")
        features.reset_index().to_csv(tmp, index=False); os.replace(tmp, path)
        core.atomic_write_json({
            **metadata(args), "structured_pipeline": STRUCTURED_PIPELINE, "view": view,
            "rows": len(features), "columns": len(features.columns), "feature_groups": groups,
            "clinical_cutoff": "index_discharge",
            "recorded_availability_cutoff": "not_applied_for_index_admission_events",
            "prediction_landmark": "index_discharge",
            "label_prevalence": {
                "readmission_30d": float(features.readmission_30d.mean()),
                "readmission_90d": float(features.readmission_90d.mean()),
                "icu_need_after_discharge_90d": float(features.icu_need_after_discharge_90d.mean()),
            },
        }, output / f"{stem}.meta.json")
        print(f"Wrote {path}: {features.shape}")
    # Patient-feature manifest used by the image selector.
    complete = output / "patient_features_discharge_complete.csv"
    compatibility = output / "patient_features.csv"
    compatibility_tmp = compatibility.with_suffix(".csv.tmp")
    pd.read_csv(complete).to_csv(compatibility_tmp, index=False)
    os.replace(compatibility_tmp, compatibility)
    core.atomic_write_json({
        **metadata(args), "structured_pipeline": STRUCTURED_PIPELINE, "view": "complete",
    }, output / "patient_features.meta.json")


def filter_notes(
    notes: pd.DataFrame,
    cohort: pd.DataFrame,
    kind: str,
    note_grace_hours: int,
) -> pd.DataFrame:
    core.require_columns(notes, {"subject_id", "hadm_id", "charttime", "storetime", "text"}, kind)
    joined = notes.merge(
        cohort[["subject_id", "hadm_id", "admittime", "dischtime", "prediction_cutoff"]],
        on=["subject_id", "hadm_id"], how="inner", validate="many_to_one",
    )
    chart = pd.to_datetime(joined.charttime, errors="coerce")
    source = kind.split("_")[0]
    cutoff = joined.dischtime + pd.to_timedelta(note_grace_hours, unit="h")
    start = joined.admittime if source == "discharge" else joined.admittime - pd.Timedelta(hours=6)
    valid = chart.notna() & chart.between(start, cutoff, inclusive="both")
    return joined.loc[valid, notes.columns].copy()


def text_stage(args: SimpleNamespace) -> None:
    cohort = pd.read_csv(
        core.require_file(Path(args.data_dir) / "analysis_cohort.csv"),
        parse_dates=["admittime", "dischtime", "prediction_cutoff"],
    )
    encoder = core.ClinicalTextEncoder(args.text_model, args.text_batch_size, args.text_chunk_stride)
    cache_dir = Path(args.text_cache_dir) / re.sub(r"[^A-Za-z0-9_.-]+", "_", args.text_model)
    cache_dir.mkdir(parents=True, exist_ok=True)
    kinds = ("discharge_complete", "radiology_complete")
    sums: Dict[str, Dict[int, np.ndarray]] = {kind: {} for kind in kinds}
    counts: Dict[str, Dict[int, int]] = {kind: {} for kind in kinds}

    def accumulate(kind: str, notes: pd.DataFrame) -> None:
        notes = notes.reset_index(drop=True)
        for start in range(0, len(notes), 64):
            block = notes.iloc[start:start + 64]
            rows = list(block.itertuples(index=False))
            vectors: List[Optional[np.ndarray]] = [None] * len(rows)
            uncached_positions: List[int] = []; uncached_texts: List[object] = []; cache_paths: List[Path] = []
            for position, row in enumerate(rows):
                note_id = str(getattr(row, "note_id", f"{row.subject_id}_{row.hadm_id}_{start + position}"))
                key = hashlib.sha256((args.text_model + "\0" + note_id + "\0" + str(row.text)).encode()).hexdigest()
                cache = cache_dir / f"{key}.npy"
                if cache.exists():
                    value = np.asarray(np.load(cache), dtype=np.float32).reshape(-1)
                    if value.size != encoder.hidden_size or not np.isfinite(value).all():
                        raise ValueError(f"Invalid cached embedding: {cache}")
                    vectors[position] = value
                else:
                    uncached_positions.append(position); uncached_texts.append(row.text); cache_paths.append(cache)
            if uncached_texts:
                for position, value, cache in zip(uncached_positions, encoder.encode_documents(uncached_texts), cache_paths):
                    if value is None:
                        continue
                    tmp = cache.with_suffix(".tmp.npy"); np.save(tmp, value); os.replace(tmp, cache)
                    vectors[position] = value
            for row, value in zip(rows, vectors):
                if value is None:
                    continue
                sid = int(row.subject_id)
                sums[kind][sid] = sums[kind].get(sid, np.zeros(encoder.hidden_size, dtype=np.float32)) + value
                counts[kind][sid] = counts[kind].get(sid, 0) + 1

    for source in ("discharge", "radiology"):
        path = core.require_file(Path(args.note_dir) / f"{source}.csv")
        for chunk_index, notes in enumerate(pd.read_csv(path, chunksize=args.note_chunksize, low_memory=False), start=1):
            for view in ("complete",):
                kind = f"{source}_{view}"
                accumulate(kind, filter_notes(notes, cohort, kind, args.note_grace_hours))
            if chunk_index % 20 == 0:
                print(f"Processed {chunk_index * args.note_chunksize:,} {source} note rows")
    outputs: Dict[str, Dict[int, np.ndarray]] = {kind: {} for kind in kinds}
    for kind in kinds:
        for sid, total in sums[kind].items():
            value = total / counts[kind][sid]
            norm = np.linalg.norm(value)
            outputs[kind][sid] = value / norm if norm > 0 else value
    rows = []
    for sid in cohort.subject_id.astype(int):
        row: Dict[str, object] = {"subject_id": sid}
        for kind in kinds:
            column = f"{kind}_embedding"
            row[column] = json.dumps(outputs[kind][sid].tolist()) if sid in outputs[kind] else np.nan
        rows.append(row)
    path = Path(args.data_dir) / "patient_text_embeddings_post.csv"; tmp = path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False); os.replace(tmp, path)
    core.atomic_write_json({
        **metadata(args), "text_pipeline": TEXT_PIPELINE, "text_model": args.text_model,
        "embedding_dimension": encoder.hidden_size,
        "complete_cutoff": (
            f"index_hadm_charttime_through_discharge_plus_{args.note_grace_hours}h;"
            "storetime_not_restricted"
        ),
        "prediction_landmark": "index_discharge",
        **{f"{kind}_subjects": len(outputs[kind]) for kind in kinds},
    }, Path(args.data_dir) / "patient_text_embeddings_post.meta.json")
    print(json.dumps({kind: len(outputs[kind]) for kind in kinds}, indent=2))


def validate_stage(args: SimpleNamespace) -> None:
    root = Path(args.data_dir)
    cohort = pd.read_csv(core.require_file(root / "analysis_cohort.csv"))
    features = pd.read_csv(core.require_file(root / "patient_features_discharge_complete.csv"))
    text = pd.read_csv(core.require_file(root / "patient_text_embeddings_post.csv"))
    cxr = pd.read_csv(core.require_file(root / "cxr_embeddings_aggregated.csv"))
    ids = set(cohort.subject_id.astype(int))
    if len(ids) != 20000 or cohort.subject_id.duplicated().any():
        raise ValueError("Post-discharge cohort must contain 20000 unique patients")
    for name, frame in (("features", features), ("text", text)):
        if frame.subject_id.duplicated().any() or set(frame.subject_id.astype(int)) != ids:
            raise ValueError(f"{name} does not align to the post-discharge cohort")
    if cxr.subject_id.duplicated().any() or not set(cxr.subject_id.astype(int)).issubset(ids):
        raise ValueError("CXR patient identifiers do not match the cohort")
    if not features.post_discharge_observed.eq(1).all():
        raise ValueError("Post-discharge cohort must contain surviving discharges")
    for label in ("readmission_30d", "readmission_90d", "icu_need_after_discharge_90d"):
        if features[label].isna().any() or not set(features[label].unique()).issubset({0, 1}):
            raise ValueError(f"Invalid {label}")
    forbidden = [c for c in features if "deathtime" in c.lower() or "hospital_expire_flag" in c.lower()]
    if forbidden or features.drop(columns="subject_id").select_dtypes(exclude=[np.number, "bool"]).columns.tolist():
        raise ValueError("Structured features contain forbidden or non-numeric columns")
    for stem in ("patient_features_discharge_complete", "patient_text_embeddings_post", "cxr_embeddings_aggregated"):
        payload = json.loads(core.require_file(root / f"{stem}.meta.json").read_text())
        if payload.get("cohort_fingerprint") != cohort_fingerprint(args):
            raise ValueError(f"{stem} belongs to another cohort")
    report = {
        "subjects": len(ids), "cohort_fingerprint": cohort_fingerprint(args),
        "readmission_prevalence": float(features.readmission_30d.mean()),
        "icu_prevalence": float(features.icu_need_after_discharge_90d.mean()),
        "cxr": int(cxr.agg_embedding.notna().sum()),
        "discharge_text": int(text.discharge_complete_embedding.notna().sum()),
        "radiology_text": int(text.radiology_complete_embedding.notna().sum()),
    }
    core.atomic_write_json(report, root / "preflight_report.json")
    print(json.dumps(report, indent=2))
