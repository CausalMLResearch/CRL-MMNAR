"""Prepare the ICU-landmark mortality cohort and its clinical inputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from preprocess_common import (
    HOSP_TABLES,
    ICU_TABLES,
    HOSP_DICTIONARIES,
    ICU_DICTIONARIES,
    LAB_SPECS,
    clean_fio2,
    VITAL_SPECS,
    OXYGEN_DEVICE_ITEM,
    VENTILATOR_TEXT_ITEMS,
    URINE_ITEMS,
    VASOPRESSOR_ITEMS,
    MEDICATION_PATTERNS,
    RRT_ITEMS,
    PipelineConfig,
    atomic_write_json,
    require_columns,
    require_file,
    table_path,
    bigquery_client,
    stream_query,
    event_window,
    interval_window,
    update_numeric_state,
    numeric_state_frame,
    aggregate_mapped_numeric,
    aggregate_binary_and_counts,
    aggregate_medications,
    aggregate_microbiology,
    stable_bucket,
    aggregate_history,
    derive_clinical_interactions,
    ClinicalTextEncoder,
)


def require_current_structured_products(cfg: PipelineConfig) -> None:
    path = Path(cfg.data_dir) / "patient_features_early.meta.json"
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Missing current structured-stage metadata: {path}. Run "
            "preprocess.sh first."
        )
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed structured-stage metadata: {path}") from exc
    expected = cohort_metadata(cfg)
    mismatched = [key for key, value in expected.items() if metadata.get(key) != value]
    if metadata.get("structured_pipeline") != "icu_dual_view" or metadata.get("view") != "early_24h":
        mismatched.append("structured_pipeline/view")
    discharge_meta = Path(cfg.data_dir) / "patient_features_discharge.meta.json"
    if not discharge_meta.is_file():
        mismatched.append("discharge_view")
    else:
        try:
            discharge_metadata = json.loads(discharge_meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed structured-stage metadata: {discharge_meta}") from exc
        discharge_mismatch = [
            key for key, value in expected.items() if discharge_metadata.get(key) != value
        ]
        if (
            discharge_metadata.get("structured_pipeline") != "icu_dual_view"
            or discharge_metadata.get("view") != "discharge"
        ):
            discharge_mismatch.append("structured_pipeline/view")
        if discharge_mismatch:
            mismatched.extend(f"discharge:{key}" for key in discharge_mismatch)
    if mismatched:
        raise ValueError(
            f"Structured products are stale for keys {mismatched}; rerun "
            "preprocess.sh."
        )



def cohort_metadata(cfg: PipelineConfig) -> Dict[str, object]:
    return {
        "pipeline": "icu_landmark",
        "cohort_fingerprint": cfg.fingerprint,
        "cohort_size": cfg.cohort_size,
        "random_seed": cfg.random_seed,
        "observation_hours": cfg.observation_hours,
        "selection": (
            "adult first ICU stay per subject from the 2008-2019 MIMIC-IV-Note-compatible era; "
            "alive and still in ICU/hospital at the landmark; future admissions retained "
            "exclusively for post-discharge labels"
        ),
    }



def download_stage(cfg: PipelineConfig) -> None:
    """Download the fixed patient cohort and its clinical tables."""
    from google.cloud import bigquery

    client = bigquery_client(cfg)
    out = Path(cfg.data_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids_path = out / "selected_subject_ids.csv"
    meta_path = out / "selected_subject_ids.meta.json"
    expected = cohort_metadata(cfg)
    cached_ok = False
    if ids_path.exists() and meta_path.exists():
        try:
            cached_ok = json.loads(meta_path.read_text()) == expected
        except (OSError, json.JSONDecodeError):
            cached_ok = False

    if not cached_ok:
        # Do not exclude subjects with future admissions: doing so would make a
        # readmission endpoint structurally impossible.  Selection depends only
        # on information available by the mortality landmark.
        sql = """
        WITH ranked AS (
          SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime,
                 a.admittime, a.dischtime, a.deathtime,
                 p.anchor_age, p.anchor_year, p.anchor_year_group,
                 ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime, i.stay_id) AS rn
          FROM `physionet-data.mimiciv_3_1_icu.icustays` i
          JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a USING(subject_id, hadm_id)
          JOIN `physionet-data.mimiciv_3_1_hosp.patients` p USING(subject_id)
        )
        SELECT subject_id FROM ranked
        WHERE rn=1
          AND anchor_age + EXTRACT(YEAR FROM admittime) - anchor_year >= 18
          AND anchor_year_group IN ('2008 - 2010', '2011 - 2013', '2014 - 2016', '2017 - 2019')
          AND intime IS NOT NULL
          AND outtime >= DATETIME_ADD(intime, INTERVAL @observation_hours HOUR)
          AND dischtime > DATETIME_ADD(intime, INTERVAL @observation_hours HOUR)
          AND (deathtime IS NULL OR deathtime > DATETIME_ADD(intime, INTERVAL @observation_hours HOUR))
        ORDER BY FARM_FINGERPRINT(CONCAT(CAST(subject_id AS STRING), '-', CAST(@seed AS STRING)))
        LIMIT @cohort_size
        """
        params = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("observation_hours", "INT64", cfg.observation_hours),
            bigquery.ScalarQueryParameter("seed", "INT64", cfg.random_seed),
            bigquery.ScalarQueryParameter("cohort_size", "INT64", cfg.cohort_size),
        ])
        ids = client.query(sql, job_config=params).to_dataframe()[["subject_id"]]
        if len(ids) != cfg.cohort_size or ids.subject_id.duplicated().any():
            raise RuntimeError(f"Expected {cfg.cohort_size} unique subjects, received {len(ids)}")
        tmp = ids_path.with_suffix(".csv.tmp")
        ids.to_csv(tmp, index=False)
        os.replace(tmp, ids_path)
        atomic_write_json(expected, meta_path)
    else:
        ids = pd.read_csv(ids_path, usecols=["subject_id"])

    subject_ids = ids.subject_id.astype(int).tolist()
    subject_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("subject_ids", "INT64", subject_ids)
    ])
    for module, tables in (("hosp", HOSP_TABLES), ("icu", ICU_TABLES)):
        for name in tables:
            path = table_path(cfg, module, name)
            cache_meta = Path(str(path) + ".cohort.json")
            valid_cache = False
            if path.exists() and cache_meta.exists():
                try:
                    valid_cache = json.loads(cache_meta.read_text()) == expected
                except (OSError, json.JSONDecodeError):
                    pass
            if valid_cache:
                print(f"Using cached {path}")
                continue
            sql = f"SELECT * FROM `physionet-data.mimiciv_3_1_{module}.{name}` WHERE subject_id IN UNNEST(@subject_ids)"
            count = stream_query(client, sql, path, subject_config, cfg.bq_page_size)
            atomic_write_json(expected, cache_meta)
            print(f"Downloaded {name}: {count:,} rows")

    # Dictionary tables are not subject scoped.
    for module, names in (("hosp", HOSP_DICTIONARIES), ("icu", ICU_DICTIONARIES)):
        for name in names:
            path = table_path(cfg, module, name)
            if path.exists() and path.stat().st_size:
                continue
            count = stream_query(
                client, f"SELECT * FROM `physionet-data.mimiciv_3_1_{module}.{name}`",
                path, None, cfg.bq_page_size,
            )
            print(f"Downloaded dictionary {name}: {count:,} rows")



def load_core_tables(cfg: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expected = cohort_metadata(cfg)
    metadata_paths = [Path(cfg.data_dir) / "selected_subject_ids.meta.json"] + [
        Path(str(table_path(cfg, module, table)) + ".cohort.json")
        for module, tables in (("hosp", HOSP_TABLES), ("icu", ICU_TABLES))
        for table in tables
    ]
    for metadata_path in metadata_paths:
        try:
            actual = json.loads(require_file(metadata_path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed raw-cache metadata: {metadata_path}") from exc
        if actual != expected:
            raise ValueError(
                f"Raw cache is stale for the legal cohort: {metadata_path}. "
                "Submit preprocess.sh before structured preprocessing."
            )
    admissions = pd.read_csv(require_file(table_path(cfg, "hosp", "admissions")), parse_dates=["admittime", "dischtime", "deathtime"])
    patients = pd.read_csv(require_file(table_path(cfg, "hosp", "patients")))
    icu = pd.read_csv(require_file(table_path(cfg, "icu", "icustays")), parse_dates=["intime", "outtime"])
    ids = pd.read_csv(require_file(Path(cfg.data_dir) / "selected_subject_ids.csv"), usecols=["subject_id"])
    require_columns(admissions, {
        "subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
        "hospital_expire_flag", "admission_type", "admission_location",
        "insurance", "language", "marital_status", "race", "discharge_location",
    }, "admissions")
    require_columns(patients, {"subject_id", "anchor_age", "anchor_year", "anchor_year_group", "gender"}, "patients")
    require_columns(icu, {"subject_id", "hadm_id", "stay_id", "intime", "outtime", "first_careunit"}, "icustays")
    if ids.subject_id.duplicated().any():
        raise ValueError("selected_subject_ids.csv contains duplicate subjects")
    selected = set(ids.subject_id.astype(int))
    return (
        admissions[admissions.subject_id.isin(selected)].copy(),
        patients[patients.subject_id.isin(selected)].copy(),
        icu[icu.subject_id.isin(selected)].copy(), ids,
    )



def build_cohort(cfg: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    admissions, patients, icu, selected = load_core_tables(cfg)
    merged = icu.merge(admissions, on=["subject_id", "hadm_id"], validate="many_to_one")
    merged = merged.merge(patients[["subject_id", "anchor_age", "anchor_year", "anchor_year_group", "gender"]], on="subject_id", validate="many_to_one")
    # Rank before applying landmark eligibility.  Filtering first would silently
    # substitute a later ICU stay when the true first stay is short or otherwise
    # ineligible, which changes the estimand and the cohort.
    cohort = merged.sort_values(["subject_id", "intime", "stay_id"], na_position="last").drop_duplicates("subject_id", keep="first")
    cohort["age_at_admission"] = cohort.anchor_age + cohort.admittime.dt.year - cohort.anchor_year
    cohort["prediction_cutoff"] = cohort.intime + pd.to_timedelta(cfg.observation_hours, unit="h")
    cohort = cohort[
        (cohort.age_at_admission >= 18)
        & cohort.anchor_year_group.isin({"2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"})
        & cohort.intime.notna() & cohort.outtime.notna()
        & (cohort.outtime >= cohort.prediction_cutoff)
        & cohort.dischtime.notna() & (cohort.dischtime > cohort.prediction_cutoff)
        & (cohort.deathtime.isna() | (cohort.deathtime > cohort.prediction_cutoff))
    ].copy()
    if cohort.subject_id.duplicated().any() or cohort.stay_id.duplicated().any():
        raise ValueError("Cohort must contain unique subject_id and stay_id")
    if len(cohort) != len(selected):
        missing = len(selected) - len(cohort)
        raise RuntimeError(
            f"{missing} selected subjects no longer satisfy the landmark cohort. "
            "Rebuild the cached cohort with preprocess.sh."
        )
    columns = [
        "subject_id", "hadm_id", "stay_id", "admittime", "dischtime", "intime",
        "outtime", "prediction_cutoff", "age_at_admission", "hospital_expire_flag",
        "first_careunit", "admission_type", "admission_location", "insurance",
        "language", "marital_status", "race", "discharge_location",
    ]
    cohort = cohort[columns].sort_values("subject_id").reset_index(drop=True)
    return cohort, admissions, patients



def derive_labels(cohort: pd.DataFrame, admissions: pd.DataFrame, icu: pd.DataFrame) -> pd.DataFrame:
    index_times = cohort[["subject_id", "hadm_id", "dischtime", "hospital_expire_flag"]].rename(columns={"hadm_id": "index_hadm_id", "dischtime": "index_dischtime"})
    future = admissions.merge(index_times[["subject_id", "index_hadm_id", "index_dischtime"]], on="subject_id", validate="many_to_one")
    delta = (future.admittime - future.index_dischtime).dt.total_seconds() / 86400.0
    future = future[(future.hadm_id != future.index_hadm_id) & delta.gt(0)].copy()
    future["days_after_discharge"] = (future.admittime - future.index_dischtime).dt.total_seconds() / 86400.0

    readmit_ids = set(future.loc[future.days_after_discharge.le(30), "subject_id"].astype(int))
    future_icu = icu.merge(index_times[["subject_id", "index_dischtime"]], on="subject_id", how="inner", validate="many_to_one")
    icu_delta = (future_icu.intime - future_icu.index_dischtime).dt.total_seconds() / 86400.0
    icu_ids = set(future_icu.loc[icu_delta.gt(0) & icu_delta.le(90), "subject_id"].astype(int))

    labels = cohort[["subject_id"]].copy().set_index("subject_id")
    labels["readmission_30d"] = labels.index.to_series().isin(readmit_ids).astype(np.int8)
    labels["icu_need_after_discharge_90d"] = labels.index.to_series().isin(icu_ids).astype(np.int8)
    labels["in_hospital_mortality"] = cohort.set_index("subject_id").hospital_expire_flag.fillna(0).astype(np.int8)
    labels["post_discharge_observed"] = (1 - labels.in_hospital_mortality).astype(np.int8)
    # Deceased index-admission patients are outside the post-discharge risk set.
    labels.loc[labels.post_discharge_observed.eq(0), ["readmission_30d", "icu_need_after_discharge_90d"]] = 0
    return labels



def structured_view(
    cfg: PipelineConfig,
    cohort: pd.DataFrame,
    admissions: pd.DataFrame,
    icu: pd.DataFrame,
    patients: pd.DataFrame,
    labels: pd.DataFrame,
    view: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Build one leakage-audited structured view for the unchanged cohort."""
    if view not in {"early_24h", "discharge"}:
        raise ValueError(view)
    view_cohort = cohort.copy()
    if view == "early_24h":
        view_cohort["window_start"] = view_cohort.intime - pd.Timedelta(hours=6)
    else:
        view_cohort["window_start"] = view_cohort.admittime
        view_cohort["prediction_cutoff"] = view_cohort.dischtime
        # At the discharge landmark, every ICU stay in the index admission is
        # observable.  The early view intentionally remains tied to the index
        # stay used to define the +24 h mortality landmark.
        view_cohort["include_all_stays"] = True

    base = cohort.set_index("subject_id")[["age_at_admission"]].copy()
    base["idx_preicu_hours"] = ((cohort.set_index("subject_id").intime - cohort.set_index("subject_id").admittime).dt.total_seconds() / 3600.0).clip(lower=0)
    if view == "discharge":
        base["idx_hospital_duration_hours"] = (
            (cohort.set_index("subject_id").dischtime - cohort.set_index("subject_id").admittime)
            .dt.total_seconds().clip(lower=0) / 3600.0
        )
        hospital_bounds = cohort[["subject_id", "hadm_id", "admittime", "dischtime"]]
        index_icu = icu.merge(
            hospital_bounds, on=["subject_id", "hadm_id"], how="inner", validate="many_to_one",
        )
        index_icu = index_icu[
            index_icu.intime.notna() & index_icu.intime.le(index_icu.dischtime)
            & (index_icu.outtime.isna() | index_icu.outtime.ge(index_icu.admittime))
        ].copy()
        interval_start = pd.concat([index_icu.intime, index_icu.admittime], axis=1).max(axis=1)
        interval_end = pd.concat([index_icu.outtime, index_icu.dischtime], axis=1).min(axis=1)
        index_icu["_duration_hours"] = (
            (interval_end - interval_start).dt.total_seconds().clip(lower=0) / 3600.0
        )
        base["idx_icu_duration_hours"] = index_icu.groupby("subject_id")._duration_hours.sum()
        base["idx_icu_stay_count"] = index_icu.groupby("subject_id").stay_id.nunique()
    categorical = [
        "first_careunit", "admission_type", "admission_location", "insurance",
        "language", "marital_status", "race",
    ]
    if view == "discharge":
        categorical.append("discharge_location")
    demographics = cohort[["subject_id", *categorical]].merge(
        patients[["subject_id", "gender", "anchor_year_group"]],
        on="subject_id", validate="one_to_one",
    ).set_index("subject_id")
    categorical += ["gender", "anchor_year_group"]
    demographics = pd.get_dummies(
        demographics.astype("string").fillna("UNKNOWN"),
        columns=categorical, prefix=[f"idx_{name}" for name in categorical],
        dtype=np.int8,
    )

    print(f"Aggregating {view} laboratory features...")
    labs = aggregate_mapped_numeric(table_path(cfg, "hosp", "labevents"), view_cohort, LAB_SPECS, cfg.csv_chunksize, "hosp")
    print(f"Aggregating {view} vital/GCS/ventilator features...")
    vitals = aggregate_mapped_numeric(table_path(cfg, "icu", "chartevents"), view_cohort, VITAL_SPECS, cfg.csv_chunksize, "icu")
    support = aggregate_binary_and_counts(cfg, view_cohort)
    medications = aggregate_medications(cfg, view_cohort)
    microbiology = aggregate_microbiology(cfg, view_cohort)
    history = aggregate_history(cfg, cohort, admissions)
    interactions = derive_clinical_interactions(
        view_cohort, labs, vitals, support, cfg.observation_hours,
    )
    if view == "discharge":
        duration_hours = (
            (view_cohort.set_index("subject_id").prediction_cutoff - view_cohort.set_index("subject_id").window_start)
            .dt.total_seconds().div(3600.0).clip(lower=1.0)
        )
        urine = (
            pd.to_numeric(support["idx_urine_output_sum"], errors="coerce").reindex(duration_hours.index)
            if "idx_urine_output_sum" in support
            else pd.Series(np.nan, index=duration_hours.index, dtype=float)
        )
        interactions["idx_urine_output_per_hour"] = urine / duration_hours

    features = base.join([
        labels, demographics, labs, vitals, support, medications,
        microbiology, history, interactions,
    ], how="left")
    count_like = [c for c in features if c.endswith("_count") or "_hash_" in c or c.endswith("_used") or c.endswith("_present") or c.endswith("_observed")]
    features[count_like] = features[count_like].fillna(0)
    # Continuous clinical values remain NaN.  They are imputed from training rows
    # only in multimodal_missingness.py; replacing them here would erase MMNAR.
    if features.index.duplicated().any() or len(features) != len(cohort):
        raise RuntimeError("Structured output lost the one-row-per-subject invariant")
    forbidden_tokens = ["deathtime", "hospital_expire_flag", "los"]
    if view == "early_24h":
        forbidden_tokens.extend(["discharge_location", "hospital_duration", "icu_duration"])
    forbidden = [c for c in features if any(token in c.lower() for token in forbidden_tokens)]
    if forbidden:
        raise RuntimeError(f"Post-outcome feature columns detected: {forbidden}")
    groups = {
        "labs": len(labs.columns), "vitals": len(vitals.columns),
        "support": len(support.columns), "medications": len(medications.columns),
        "microbiology": len(microbiology.columns), "history": len(history.columns),
        "clinical_interactions": len(interactions.columns),
    }
    return features, groups



def structured_stage(cfg: PipelineConfig) -> None:
    cohort, admissions, patients = build_cohort(cfg)
    icu = pd.read_csv(require_file(table_path(cfg, "icu", "icustays")), parse_dates=["intime", "outtime"])
    labels = derive_labels(cohort, admissions, icu)

    out = Path(cfg.data_dir)
    cohort_path = out / "analysis_cohort.csv"
    cohort_tmp = cohort_path.with_suffix(".csv.tmp")
    cohort.to_csv(cohort_tmp, index=False)
    os.replace(cohort_tmp, cohort_path)
    for view, stem in (("early_24h", "patient_features_early"), ("discharge", "patient_features_discharge")):
        print(f"\nBuilding structured  view: {view}")
        features, groups = structured_view(cfg, cohort, admissions, icu, patients, labels, view)
        final_path = out / f"{stem}.csv"
        tmp = final_path.with_suffix(".csv.tmp")
        features.reset_index().to_csv(tmp, index=False)
        os.replace(tmp, final_path)
        atomic_write_json({
            **cohort_metadata(cfg),
            "structured_pipeline": "icu_dual_view",
            "view": view,
            "rows": len(features), "columns": len(features.columns),
            "cutoff_rule": (
                "ICU intime + observation_hours; event and store time must be at or before cutoff"
                if view == "early_24h" else
                "index discharge; event and store time must be at or before discharge"
            ),
            "label_prevalence": {
                name: float(features.loc[features.post_discharge_observed.eq(1), name].mean())
                if name != "in_hospital_mortality" else float(features[name].mean())
                for name in ("readmission_30d", "icu_need_after_discharge_90d", "in_hospital_mortality")
            },
            "feature_groups": groups,
        }, out / f"{stem}.meta.json")
        print(f"Wrote {final_path}: {features.shape}")



def filter_notes(notes: pd.DataFrame, cohort: pd.DataFrame, kind: str) -> pd.DataFrame:
    require_columns(notes, {"subject_id", "hadm_id", "charttime", "storetime", "text"}, f"{kind} notes")
    joined = notes.merge(cohort[["subject_id", "hadm_id", "admittime", "dischtime", "intime", "prediction_cutoff"]], on=["subject_id", "hadm_id"], how="inner", validate="many_to_one")
    chart = pd.to_datetime(joined.charttime, errors="coerce")
    store = pd.to_datetime(joined.storetime, errors="coerce")
    if kind == "radiology_early":
        valid = chart.between(joined.intime - pd.Timedelta(hours=6), joined.prediction_cutoff, inclusive="both") & store.notna() & store.le(joined.prediction_cutoff)
    elif kind in {"radiology", "discharge"}:
        # Post-discharge heads use an index-discharge prediction landmark.
        # Both clinical and recorded-availability times must therefore be no
        # later than discharge; no documentation grace period is allowed.
        documentation_cutoff = joined.dischtime
        document_start = joined.admittime
        if kind == "radiology":
            document_start = pd.concat(
                [joined.admittime, joined.intime - pd.Timedelta(hours=6)], axis=1,
            ).min(axis=1)
        valid = (
            chart.ge(document_start) & chart.le(documentation_cutoff)
            & store.notna() & store.le(documentation_cutoff)
        )
    else:
        raise ValueError(kind)
    return joined.loc[valid, notes.columns].copy()



def text_stage(cfg: PipelineConfig, model_name: str, batch_size: int, stride: int, note_chunksize: int) -> None:
    require_current_structured_products(cfg)
    if note_chunksize <= 0:
        raise ValueError("note_chunksize must be positive")
    cohort = pd.read_csv(require_file(Path(cfg.data_dir) / "analysis_cohort.csv"), parse_dates=["admittime", "dischtime", "intime", "prediction_cutoff"])
    encoder = ClinicalTextEncoder(model_name, batch_size, stride)
    cache_dir = Path(cfg.data_dir) / "text_embedding_cache" / re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sums: Dict[str, Dict[int, np.ndarray]] = {kind: {} for kind in ("discharge", "radiology", "radiology_early")}
    counts: Dict[str, Dict[int, int]] = {kind: {} for kind in sums}

    def accumulate(kind: str, notes: pd.DataFrame) -> None:
        notes = notes.reset_index(drop=True)
        for start in range(0, len(notes), 64):
            block = notes.iloc[start:start + 64]
            rows = list(block.itertuples(index=False))
            vectors: List[Optional[np.ndarray]] = [None] * len(rows)
            uncached_positions: List[int] = []
            uncached_texts: List[object] = []
            cache_paths: List[Path] = []
            for position, row in enumerate(rows):
                note_id = str(getattr(row, "note_id", f"{row.subject_id}_{row.hadm_id}_{start + position}"))
                key = hashlib.sha256((model_name + "\0" + note_id + "\0" + str(row.text)).encode("utf-8")).hexdigest()
                cache = cache_dir / f"{key}.npy"
                if cache.exists():
                    cached = np.asarray(np.load(cache), dtype=np.float32).reshape(-1)
                    if cached.size != encoder.hidden_size or not np.isfinite(cached).all():
                        raise ValueError(f"Invalid cached text embedding: {cache}")
                    vectors[position] = cached
                else:
                    uncached_positions.append(position)
                    uncached_texts.append(row.text)
                    cache_paths.append(cache)
            if uncached_texts:
                encoded = encoder.encode_documents(uncached_texts)
                for position, vector, cache in zip(uncached_positions, encoded, cache_paths):
                    if vector is None:
                        continue
                    tmp = cache.with_suffix(".tmp.npy")
                    np.save(tmp, vector)
                    os.replace(tmp, cache)
                    vectors[position] = vector
            for row, vector in zip(rows, vectors):
                if vector is None:
                    continue
                subject_id = int(row.subject_id)
                value = np.asarray(vector, dtype=np.float32)
                if subject_id in sums[kind]:
                    sums[kind][subject_id] += value
                else:
                    sums[kind][subject_id] = value.copy()
                counts[kind][subject_id] = counts[kind].get(subject_id, 0) + 1

    for source in ("discharge", "radiology"):
        path = require_file(Path(cfg.note_dir) / f"{source}.csv")
        for chunk_index, notes in enumerate(pd.read_csv(path, chunksize=note_chunksize, low_memory=False), start=1):
            kinds = ("discharge",) if source == "discharge" else ("radiology", "radiology_early")
            for kind in kinds:
                accumulate(kind, filter_notes(notes, cohort, kind))
            if chunk_index % 20 == 0:
                print(f"Processed {chunk_index * note_chunksize:,} {source} note rows")

    outputs: Dict[str, Dict[int, np.ndarray]] = {kind: {} for kind in sums}
    for kind in sums:
        for sid, total in sums[kind].items():
            value = total / counts[kind][sid]
            norm = np.linalg.norm(value)
            outputs[kind][sid] = value / norm if norm > 0 else value

    rows = []
    for sid in cohort.subject_id.astype(int):
        rows.append({
            "subject_id": sid,
            "discharge_embedding": json.dumps(outputs["discharge"][sid].tolist()) if sid in outputs["discharge"] else np.nan,
            "radiology_embedding": json.dumps(outputs["radiology"][sid].tolist()) if sid in outputs["radiology"] else np.nan,
            "radiology_early_embedding": json.dumps(outputs["radiology_early"][sid].tolist()) if sid in outputs["radiology_early"] else np.nan,
        })
    path = Path(cfg.data_dir) / "patient_text_embeddings.csv"
    tmp = path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)
    atomic_write_json({
        **cohort_metadata(cfg), "text_model": model_name,
        "text_pipeline": "icu_dual_view",
        "post_discharge_cutoff": "charttime_and_storetime_at_or_before_index_discharge",
        "mortality_cutoff": "charttime_and_storetime_at_or_before_icu_plus_observation_hours",
        "embedding_dimension": encoder.hidden_size,
        "discharge_subjects": len(outputs["discharge"]),
        "radiology_subjects": len(outputs["radiology"]),
        "radiology_early_subjects": len(outputs["radiology_early"]),
    }, Path(cfg.data_dir) / "patient_text_embeddings.meta.json")
    print(
        f"Wrote {path}; discharge={len(outputs['discharge'])}, "
        f"radiology={len(outputs['radiology'])}, "
        f"radiology_early={len(outputs['radiology_early'])}"
    )



def validate_stage(cfg: PipelineConfig) -> None:
    require_current_structured_products(cfg)
    expected = cohort_metadata(cfg)
    artifact_metadata = {
        "text": Path(cfg.data_dir) / "patient_text_embeddings.meta.json",
        "cxr": Path(cfg.data_dir) / "cxr_embeddings_aggregated.meta.json",
        "cxr_early": Path(cfg.data_dir) / "cxr_embeddings_early_aggregated.meta.json",
    }
    loaded_metadata = {}
    for name, path in artifact_metadata.items():
        try:
            metadata = json.loads(require_file(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed {name} metadata: {path}") from exc
        if metadata.get("cohort_fingerprint") != expected["cohort_fingerprint"]:
            raise ValueError(f"{name} embeddings were produced for a different cohort")
        loaded_metadata[name] = metadata
    if (
        loaded_metadata["text"].get("text_pipeline") != "icu_dual_view"
        or loaded_metadata["text"].get("post_discharge_cutoff")
        != "charttime_and_storetime_at_or_before_index_discharge"
    ):
        raise ValueError("Text embeddings do not use the index-discharge cutoff; rerun the text stage")
    if loaded_metadata["cxr"].get("window") != "hospital" or loaded_metadata["cxr_early"].get("window") != "landmark":
        raise ValueError("CXR artifacts have incorrect prediction windows")
    if not loaded_metadata["cxr"].get("frontal_only") or not loaded_metadata["cxr_early"].get("frontal_only"):
        raise ValueError("Final-model CXR artifacts must contain frontal AP/PA views only")
    early = pd.read_csv(require_file(Path(cfg.data_dir) / "patient_features_early.csv"))
    discharge = pd.read_csv(require_file(Path(cfg.data_dir) / "patient_features_discharge.csv"))
    text = pd.read_csv(require_file(Path(cfg.data_dir) / "patient_text_embeddings.csv"), usecols=["subject_id", "discharge_embedding", "radiology_embedding", "radiology_early_embedding"])
    cxr = pd.read_csv(require_file(Path(cfg.data_dir) / "cxr_embeddings_aggregated.csv"), usecols=["subject_id", "agg_embedding"])
    cxr_early = pd.read_csv(require_file(Path(cfg.data_dir) / "cxr_embeddings_early_aggregated.csv"), usecols=["subject_id", "agg_embedding"])
    cohort = pd.read_csv(require_file(Path(cfg.data_dir) / "analysis_cohort.csv"))
    for name, frame in (("early", early), ("discharge", discharge), ("text", text), ("cxr", cxr), ("cxr_early", cxr_early), ("cohort", cohort)):
        require_columns(frame, {"subject_id"}, name)
        if frame.subject_id.duplicated().any(): raise ValueError(f"{name} contains duplicate subject_id")
    ids = set(early.subject_id.astype(int))
    if ids != set(cohort.subject_id.astype(int)): raise ValueError("Feature and cohort subject sets differ")
    if ids != set(discharge.subject_id.astype(int)): raise ValueError("Early and discharge subject sets differ")
    if len(ids) != cfg.cohort_size:
        raise ValueError(f"Expected {cfg.cohort_size} cohort subjects, found {len(ids)}")
    if set(text.subject_id.astype(int)) != ids:
        raise ValueError("Text embedding file must contain exactly the analysis-cohort subjects")
    if not set(cxr.subject_id.astype(int)).issubset(ids):
        raise ValueError("CXR embedding file contains subjects outside the analysis cohort")
    if not set(cxr_early.subject_id.astype(int)).issubset(set(cxr.subject_id.astype(int))):
        raise ValueError("Landmark CXR subjects must be a subset of hospital-course CXR subjects")
    if (text.radiology_early_embedding.notna() & text.radiology_embedding.isna()).any():
        raise ValueError("Landmark radiology availability must be a subset of hospital-course availability")
    label_columns = [
        "readmission_30d", "icu_need_after_discharge_90d",
        "in_hospital_mortality", "post_discharge_observed",
    ]
    required_labels = set(label_columns)
    aligned_early = early.sort_values("subject_id").reset_index(drop=True)
    aligned_discharge = discharge.sort_values("subject_id").reset_index(drop=True)
    if not aligned_early[["subject_id", *label_columns]].equals(aligned_discharge[["subject_id", *label_columns]]):
        raise ValueError("Early and discharge labels differ")
    for view_name, features in (("early", early), ("discharge", discharge)):
        require_columns(features, required_labels, view_name)
        for label in required_labels:
            if features[label].isna().any(): raise ValueError(f"{view_name}:{label} contains missing values")
            values = set(features[label].astype(int).unique())
            if not values.issubset({0, 1}): raise ValueError(f"{view_name}:{label} is not binary: {values}")
        if not features.post_discharge_observed.equals(1 - features.in_hospital_mortality):
            raise ValueError(f"{view_name}: post_discharge_observed must equal 1 - mortality")
        deceased = features.in_hospital_mortality.eq(1)
        if features.loc[deceased, ["readmission_30d", "icu_need_after_discharge_90d"]].to_numpy().any():
            raise ValueError(f"{view_name}: post-discharge labels are nonzero outside the risk set")
        numeric = features.drop(columns=["subject_id"])
        non_numeric = numeric.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
        if non_numeric: raise ValueError(f"{view_name} contains non-numeric columns: {non_numeric[:10]}")
        if np.isinf(numeric.select_dtypes(include=[np.number]).to_numpy(dtype=float)).any():
            raise ValueError(f"{view_name} contains infinite values")
    forbidden_early = [c for c in early if any(x in c.lower() for x in (
        "deathtime", "hospital_expire_flag", "discharge_location", "hospital_duration", "icu_duration",
    ))]
    if forbidden_early: raise ValueError(f"Early view contains post-landmark features: {forbidden_early}")
    report = {
        "subjects": len(early),
        "cxr": int(cxr.agg_embedding.notna().sum()),
        "cxr_early": int(cxr_early.agg_embedding.notna().sum()),
        "discharge": int(text.discharge_embedding.notna().sum()),
        "radiology": int(text.radiology_embedding.notna().sum()),
        "radiology_early": int(text.radiology_early_embedding.notna().sum()),
        "early_feature_columns": len(early.columns) - 5,
        "discharge_feature_columns": len(discharge.columns) - 5,
    }
    atomic_write_json(report, Path(cfg.data_dir) / "preflight_report.json")
    print(json.dumps(report, indent=2))



def cxr_views(cohort: pd.DataFrame, records: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    joined = records.merge(
        cohort[["subject_id", "admittime", "dischtime", "intime", "prediction_cutoff"]],
        on="subject_id", how="inner", validate="many_to_one",
    )
    hospital_start = pd.concat(
        [joined["admittime"], joined["intime"] - pd.Timedelta(hours=6)], axis=1,
    ).min(axis=1)
    hospital = joined[
        joined.study_datetime.ge(hospital_start) & joined.study_datetime.le(joined.dischtime)
    ].copy()
    landmark = joined[
        joined.study_datetime.ge(joined.intime - pd.Timedelta(hours=6))
        & joined.study_datetime.le(joined.prediction_cutoff)
    ].copy()
    return {"cxr_embeddings": hospital, "cxr_embeddings_early": landmark}
