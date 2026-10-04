"""Shared clinical feature aggregation, text encoding and file utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


HOSP_TABLES = (
    "admissions", "patients", "diagnoses_icd", "procedures_icd", "labevents",
    "microbiologyevents", "emar",
)



ICU_TABLES = ("icustays", "chartevents", "inputevents", "outputevents")



HOSP_DICTIONARIES = ("d_labitems",)



ICU_DICTIONARIES = ("d_items",)



LAB_SPECS: Dict[int, Tuple[str, Optional[float], Optional[float]]] = {
    50862: ("albumin", 0.0, 10.0), 50868: ("anion_gap", -50.0, 100.0),
    50882: ("bicarbonate", 0.0, 100.0), 50885: ("bilirubin_total", 0.0, 150.0),
    50912: ("creatinine", 0.0, 150.0), 50902: ("chloride", 0.0, 200.0),
    50931: ("glucose_lab", 0.0, 10000.0), 50971: ("potassium", 0.0, 30.0),
    50983: ("sodium", 0.0, 200.0), 51006: ("bun", 0.0, 300.0),
    51221: ("hematocrit", 0.0, 100.0), 51222: ("hemoglobin", 0.0, 30.0),
    51265: ("platelet", 0.0, 5000.0), 51301: ("wbc", 0.0, 1000.0),
    50861: ("alt", 0.0, 100000.0), 50863: ("alp", 0.0, 100000.0),
    50878: ("ast", 0.0, 100000.0), 50813: ("lactate", 0.0, 100.0),
    50820: ("ph", 6.0, 8.0), 50818: ("pco2", 0.0, 300.0),
    50821: ("po2", 0.0, 800.0), 50816: ("fio2_lab", 0.2, 100.0),
    51237: ("inr", 0.0, 50.0), 51274: ("pt", 0.0, 150.0),
    51275: ("ptt", 0.0, 300.0), 50960: ("magnesium", 0.0, 20.0),
    50893: ("calcium", 0.0, 30.0), 50970: ("phosphate", 0.0, 30.0),
}



def clean_fio2(value: float) -> float:
    """Apply the MIT-LCP ventilator_setting FiO2 unit correction."""
    if 0.20 <= value <= 1.0:
        return value * 100.0
    if 20.0 <= value <= 100.0:
        return value
    return float("nan")



VITAL_SPECS: Dict[int, Tuple[str, Optional[float], Optional[float], Optional[Callable[[float], float]]]] = {
    220045: ("heart_rate", 0.0, 300.0, None),
    220179: ("sbp", 0.0, 400.0, None), 220050: ("sbp", 0.0, 400.0, None),
    225309: ("sbp", 0.0, 400.0, None),
    220180: ("dbp", 0.0, 300.0, None), 220051: ("dbp", 0.0, 300.0, None),
    225310: ("dbp", 0.0, 300.0, None),
    220181: ("mbp", 0.0, 300.0, None), 220052: ("mbp", 0.0, 300.0, None),
    225312: ("mbp", 0.0, 300.0, None),
    220210: ("resp_rate", 0.0, 70.0, None), 224690: ("resp_rate", 0.0, 70.0, None),
    220277: ("spo2", 0.0, 100.0, None),
    225664: ("glucose_chart", 0.0, 10000.0, None),
    220621: ("glucose_chart", 0.0, 10000.0, None),
    226537: ("glucose_chart", 0.0, 10000.0, None),
    223762: ("temperature_c", 10.0, 50.0, None),
    223761: ("temperature_c", 70.0, 120.0, lambda x: (x - 32.0) / 1.8),
    223835: ("fio2", None, None, clean_fio2),
    220339: ("peep", 0.0, 100.0, None), 224700: ("peep", 0.0, 100.0, None),
    224688: ("resp_rate_set", 0.0, 70.0, None),
    224689: ("resp_rate_spontaneous", 0.0, 70.0, None),
    224687: ("minute_volume", 0.0, 100.0, None),
    224684: ("tidal_volume_set", 0.0, 5000.0, None),
    224685: ("tidal_volume_observed", 0.0, 5000.0, None),
    224686: ("tidal_volume_spontaneous", 0.0, 5000.0, None),
    224696: ("plateau_pressure", 0.0, 100.0, None),
    224691: ("ventilator_flow_rate", 0.0, 200.0, None),
    223834: ("o2_flow", 0.0, 100.0, None), 227582: ("o2_flow", 0.0, 100.0, None),
    227287: ("o2_flow_additional", 0.0, 100.0, None),
    223900: ("gcs_verbal", 0.0, 5.0, None),
    223901: ("gcs_motor", 1.0, 6.0, None), 220739: ("gcs_eye", 1.0, 4.0, None),
}



OXYGEN_DEVICE_ITEM = 226732



VENTILATOR_TEXT_ITEMS = {223848, 223849, 229314}



URINE_ITEMS = {226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558, 227488, 227489}



VASOPRESSOR_ITEMS = {221906: "norepinephrine", 221289: "epinephrine", 222315: "vasopressin", 221662: "dopamine", 221653: "dobutamine", 221749: "phenylephrine"}



MEDICATION_PATTERNS = {
    "antibiotic": r"\b(?:vancomycin|cef[a-z]*|[a-z]*cillin|meropenem|ertapenem|azithromycin|doxycycline|ciprofloxacin|levofloxacin|metronidazole|clindamycin)\b",
    "anticoagulant": r"\b(?:heparin|enoxaparin|warfarin|apixaban|rivaroxaban|argatroban)\b",
    "diuretic": r"\b(?:furosemide|lasix|bumetanide|torsemide|chlorothiazide|metolazone)\b",
    "insulin": r"\binsulin\b",
    "sedative": r"\b(?:propofol|dexmedetomidine|midazolam|lorazepam|ketamine)\b",
    "opioid": r"\b(?:fentanyl|morphine|hydromorphone|oxycodone|methadone)\b",
}



RRT_ITEMS = {
    226499, 224154, 225810, 225959, 227639, 225183, 227438, 224191, 225806,
    225807, 228004, 228005, 228006, 224144, 224145, 224153, 226457, 227290,
}



@dataclass(frozen=True)
class PipelineConfig:
    data_dir: str
    note_dir: str
    cxr_dir: str
    cohort_size: int
    random_seed: int
    observation_hours: int
    protocol: str
    gcp_project: Optional[str]
    bq_page_size: int
    csv_chunksize: int

    @property
    def fingerprint(self) -> str:
        return "3aa2cda1d3682d66"



def atomic_write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)



def require_columns(df: pd.DataFrame, required: Iterable[str], source: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")



def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required non-empty file is missing: {path}")
    return path



def table_path(cfg: PipelineConfig, module: str, table: str) -> Path:
    return Path(cfg.data_dir) / f"{module}_{table}.csv"



def bigquery_client(cfg: PipelineConfig):
    from google.cloud import bigquery
    return bigquery.Client(project=cfg.gcp_project) if cfg.gcp_project else bigquery.Client()



def stream_query(client, sql: str, path: Path, job_config, page_size: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    result = client.query(sql, job_config=job_config).result(page_size=page_size)
    fields = [field.name for field in result.schema]
    count = 0
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for page in result.pages:
            for row in page:
                writer.writerow([row[name] for name in fields])
                count += 1
    os.replace(tmp, path)
    return count



def event_window(chunk: pd.DataFrame, cohort: pd.DataFrame, time_col: str, store_col: Optional[str], allow_pre_hours: int = 6) -> pd.DataFrame:
    all_hospital_stays = "include_all_stays" in cohort.columns and bool(cohort.include_all_stays.all())
    enforce_store_cutoff = (
        "enforce_store_cutoff" not in cohort.columns
        or bool(cohort.enforce_store_cutoff.all())
    )
    keys = ["subject_id", "hadm_id"] + (["stay_id"] if "stay_id" in chunk.columns and not all_hospital_stays else [])
    lookup_cols = keys + ["intime", "prediction_cutoff"]
    if "window_start" in cohort.columns:
        lookup_cols.append("window_start")
    if "clinical_cutoff" in cohort.columns:
        lookup_cols.append("clinical_cutoff")
    lookup = cohort[lookup_cols].drop_duplicates(keys)
    merged = chunk.merge(lookup, on=keys, how="inner", validate="many_to_one")
    event_time = pd.to_datetime(merged[time_col], errors="coerce")
    start = (
        pd.to_datetime(merged.window_start, errors="coerce")
        if "window_start" in merged.columns
        else merged.intime - pd.to_timedelta(allow_pre_hours, unit="h")
    )
    clinical_cutoff = (
        pd.to_datetime(merged.clinical_cutoff, errors="coerce")
        if "clinical_cutoff" in merged.columns else merged.prediction_cutoff
    )
    valid = event_time.notna() & event_time.between(start, clinical_cutoff, inclusive="both")
    if enforce_store_cutoff and store_col and store_col in merged.columns:
        store = pd.to_datetime(merged[store_col], errors="coerce")
        valid &= store.notna() & store.le(merged.prediction_cutoff)
    merged = merged.loc[valid].copy()
    merged["_event_time"] = event_time.loc[valid]
    merged["_hours_from_icu"] = (merged._event_time - merged.intime).dt.total_seconds() / 3600.0
    merged["_hours_to_cutoff"] = (clinical_cutoff.loc[valid] - merged._event_time).dt.total_seconds() / 3600.0
    return merged



def interval_window(
    chunk: pd.DataFrame,
    cohort: pd.DataFrame,
    start_col: str,
    end_col: str,
    store_col: Optional[str],
    allow_pre_hours: int = 6,
) -> pd.DataFrame:
    """Keep interval events that overlap the observable landmark window."""
    all_hospital_stays = "include_all_stays" in cohort.columns and bool(cohort.include_all_stays.all())
    enforce_store_cutoff = (
        "enforce_store_cutoff" not in cohort.columns
        or bool(cohort.enforce_store_cutoff.all())
    )
    keys = ["subject_id", "hadm_id"] + (["stay_id"] if "stay_id" in chunk.columns and not all_hospital_stays else [])
    lookup_cols = keys + ["intime", "prediction_cutoff"]
    if "window_start" in cohort.columns:
        lookup_cols.append("window_start")
    if "clinical_cutoff" in cohort.columns:
        lookup_cols.append("clinical_cutoff")
    lookup = cohort[lookup_cols].drop_duplicates(keys)
    merged = chunk.merge(lookup, on=keys, how="inner", validate="many_to_one")
    start = pd.to_datetime(merged[start_col], errors="coerce")
    end = pd.to_datetime(merged[end_col], errors="coerce")
    window_start = (
        pd.to_datetime(merged.window_start, errors="coerce")
        if "window_start" in merged.columns
        else merged.intime - pd.to_timedelta(allow_pre_hours, unit="h")
    )
    clinical_cutoff = (
        pd.to_datetime(merged.clinical_cutoff, errors="coerce")
        if "clinical_cutoff" in merged.columns else merged.prediction_cutoff
    )
    valid = start.notna() & end.notna() & start.le(clinical_cutoff) & end.ge(window_start)
    if enforce_store_cutoff and store_col and store_col in merged.columns:
        store = pd.to_datetime(merged[store_col], errors="coerce")
        valid &= store.notna() & store.le(merged.prediction_cutoff)
    return merged.loc[valid].copy()



def update_numeric_state(state: MutableMapping[Tuple[int, str], List[object]], rows: pd.DataFrame) -> None:
    fields = ["subject_id", "feature", "numeric_value", "_event_time"]
    for subject_id, feature, numeric_value, event_time in rows[fields].itertuples(index=False, name=None):
        key = (int(subject_id), str(feature))
        value, when = float(numeric_value), event_time
        current = state.get(key)
        if current is None:
            # count, sum, sumsq, min, max, first_time, first_value,
            # last_time, last_value
            state[key] = [1, value, value * value, value, value, when, value, when, value]
        else:
            current[0] += 1; current[1] += value; current[2] += value * value
            current[3] = min(float(current[3]), value); current[4] = max(float(current[4]), value)
            if when < current[5]:
                current[5], current[6] = when, value
            if when >= current[7]:
                current[7], current[8] = when, value



def numeric_state_frame(state: Mapping[Tuple[int, str], Sequence[object]], cohort: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cutoff_column = "clinical_cutoff" if "clinical_cutoff" in cohort.columns else "prediction_cutoff"
    cutoff = cohort.set_index("subject_id")[cutoff_column].to_dict()
    records: List[Dict[str, object]] = []
    for (subject_id, feature), values in state.items():
        count, total, total_sq, minimum, maximum, first_time, first_value, last_time, last_value = values
        mean = float(total) / int(count)
        variance = max(float(total_sq) / int(count) - mean * mean, 0.0)
        records.append({
            "subject_id": subject_id, "feature": feature, "count": count,
            "mean": mean, "std": math.sqrt(variance), "min": minimum, "max": maximum,
            "last": last_value,
            "change": float(last_value) - float(first_value),
            "slope_per_hour": (
                (float(last_value) - float(first_value))
                / max((last_time - first_time).total_seconds() / 3600.0, 1.0 / 60.0)
                if int(count) > 1 else 0.0
            ),
            "hours_since_last": max((cutoff[subject_id] - last_time).total_seconds() / 3600.0, 0.0),
        })
    if not records:
        return pd.DataFrame(index=cohort.subject_id)
    long = pd.DataFrame.from_records(records)
    wide = long.pivot(index="subject_id", columns="feature")
    wide.columns = [f"{prefix}_{feature}_{stat}" for stat, feature in wide.columns]
    # Explicit observation indicators are central MMNAR features.
    for feature in long.feature.unique():
        wide[f"{prefix}_{feature}_observed"] = wide.get(f"{prefix}_{feature}_count", 0).notna().astype(np.int8)
    return wide.sort_index(axis=1)



def aggregate_mapped_numeric(path: Path, cohort: pd.DataFrame, specs: Mapping[int, Sequence[object]], chunksize: int, module: str) -> pd.DataFrame:
    state: Dict[Tuple[int, str], List[object]] = {}
    usecols = ["subject_id", "hadm_id", "itemid", "charttime", "storetime", "valuenum"] + (["stay_id"] if module == "icu" else [])
    for chunk in pd.read_csv(require_file(path), usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk = chunk[chunk.itemid.isin(specs)]
        if chunk.empty:
            continue
        chunk = event_window(chunk, cohort, "charttime", "storetime")
        if chunk.empty:
            continue
        chunk["numeric_value"] = pd.to_numeric(chunk.valuenum, errors="coerce")
        chunk["feature"] = chunk.itemid.map(lambda x: specs[int(x)][0])
        lower = chunk.itemid.map(lambda x: specs[int(x)][1])
        upper = chunk.itemid.map(lambda x: specs[int(x)][2])
        valid = chunk.numeric_value.notna()
        valid &= lower.isna() | chunk.numeric_value.gt(lower.astype(float))
        valid &= upper.isna() | chunk.numeric_value.lt(upper.astype(float))
        chunk = chunk[valid].copy()
        if module == "icu" and not chunk.empty:
            for itemid, spec in specs.items():
                transform = spec[3]
                if transform is not None:
                    mask = chunk.itemid.eq(itemid)
                    chunk.loc[mask, "numeric_value"] = chunk.loc[mask, "numeric_value"].map(transform)
            chunk = chunk[np.isfinite(chunk.numeric_value)].copy()

        # Preserve both whole-landmark summaries and coarse temporal dynamics.
        # The bins are fixed a priori and never use outcome information.
        update_numeric_state(state, chunk)
        bins = (
            ("pre6_0h", chunk._hours_from_icu.lt(0.0)),
            ("h00_06", chunk._hours_from_icu.ge(0.0) & chunk._hours_from_icu.lt(6.0)),
            ("h06_12", chunk._hours_from_icu.ge(6.0) & chunk._hours_from_icu.lt(12.0)),
            ("h12_24", chunk._hours_from_icu.ge(12.0) & chunk._hours_from_icu.le(24.0)),
            ("recent_00_06", chunk._hours_to_cutoff.ge(0.0) & chunk._hours_to_cutoff.lt(6.0)),
            ("recent_06_12", chunk._hours_to_cutoff.ge(6.0) & chunk._hours_to_cutoff.lt(12.0)),
            ("recent_12_24", chunk._hours_to_cutoff.ge(12.0) & chunk._hours_to_cutoff.le(24.0)),
        )
        for suffix, mask in bins:
            if mask.any():
                binned = chunk.loc[mask].copy()
                binned["feature"] = binned.feature.astype(str) + "_" + suffix
                update_numeric_state(state, binned)
    return numeric_state_frame(state, cohort, "idx")



def aggregate_binary_and_counts(cfg: PipelineConfig, cohort: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=cohort.subject_id.astype(int))
    result.index.name = "subject_id"
    # Urine output using the official GU irrigant sign convention.
    urine_sum: Dict[int, float] = {}
    urine_count: Dict[int, int] = {}
    path = table_path(cfg, "icu", "outputevents")
    for chunk in pd.read_csv(require_file(path), usecols=["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "storetime", "value"], chunksize=cfg.csv_chunksize, low_memory=False):
        chunk = chunk[chunk.itemid.isin(URINE_ITEMS)]
        if chunk.empty: continue
        chunk = event_window(chunk, cohort, "charttime", "storetime")
        chunk["value"] = pd.to_numeric(chunk.value, errors="coerce")
        chunk = chunk[chunk.value.notna()]
        chunk.loc[chunk.itemid.eq(227488) & chunk.value.gt(0), "value"] *= -1
        grouped = chunk.groupby("subject_id").value.agg(["sum", "count"])
        for sid, row in grouped.iterrows():
            urine_sum[int(sid)] = urine_sum.get(int(sid), 0.0) + float(row["sum"])
            urine_count[int(sid)] = urine_count.get(int(sid), 0) + int(row["count"])
    result["idx_urine_output_sum"] = pd.Series(urine_sum)
    result["idx_urine_output_count"] = pd.Series(urine_count)
    result["idx_urine_output_observed"] = result.idx_urine_output_count.notna().astype(np.int8)

    # Vasopressor administration/order signals from inputevents.
    vaso_counts: Dict[Tuple[int, str], int] = {}
    path = table_path(cfg, "icu", "inputevents")
    for chunk in pd.read_csv(require_file(path), usecols=["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "endtime", "storetime", "statusdescription"], chunksize=cfg.csv_chunksize, low_memory=False):
        chunk = chunk[chunk.itemid.isin(VASOPRESSOR_ITEMS)]
        if chunk.empty: continue
        chunk = interval_window(chunk, cohort, "starttime", "endtime", "storetime")
        if "statusdescription" in chunk:
            chunk = chunk[~chunk.statusdescription.astype(str).str.contains("rewritten|cancel", case=False, regex=True, na=False)]
        for (sid, itemid), count in chunk.groupby(["subject_id", "itemid"]).size().items():
            key = (int(sid), VASOPRESSOR_ITEMS[int(itemid)])
            vaso_counts[key] = vaso_counts.get(key, 0) + int(count)
    for drug in sorted(set(VASOPRESSOR_ITEMS.values())):
        series = pd.Series({sid: count for (sid, name), count in vaso_counts.items() if name == drug})
        result[f"idx_{drug}_event_count"] = series
        result[f"idx_{drug}_used"] = series.gt(0).astype(np.int8)

    # Text-valued respiratory support and RRT signals from chartevents.
    support: Dict[int, Dict[str, int]] = {}
    relevant = set(VENTILATOR_TEXT_ITEMS) | {OXYGEN_DEVICE_ITEM} | RRT_ITEMS
    path = table_path(cfg, "icu", "chartevents")
    for chunk in pd.read_csv(require_file(path), usecols=["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "storetime", "value"], chunksize=cfg.csv_chunksize, low_memory=False):
        chunk = chunk[chunk.itemid.isin(relevant)]
        if chunk.empty: continue
        chunk = event_window(chunk, cohort, "charttime", "storetime")
        for sid, group in chunk.groupby("subject_id"):
            rec = support.setdefault(int(sid), {"vent": 0, "oxygen": 0, "rrt": 0})
            text = " ".join(group.value.dropna().astype(str)).lower()
            rec["vent"] += int(group.itemid.isin(VENTILATOR_TEXT_ITEMS).sum() + bool(re.search(r"endotracheal|invasive|cmv|simv|prvc|ventilator", text)))
            rec["oxygen"] += int(group.itemid.eq(OXYGEN_DEVICE_ITEM).sum())
            rec["rrt"] += int(group.itemid.isin(RRT_ITEMS).sum())
    for key in ("vent", "oxygen", "rrt"):
        counts = pd.Series({sid: rec[key] for sid, rec in support.items()})
        result[f"idx_{key}_event_count"] = counts
        result[f"idx_{key}_present"] = counts.gt(0).astype(np.int8)
    return result



def aggregate_medications(cfg: PipelineConfig, cohort: pd.DataFrame) -> pd.DataFrame:
    """Count medications documented as given by the +24 h landmark.

    EMAR is preferred to prescriptions because its chart/store timestamps let us
    enforce recorded availability.  Medication strings are used only for broad,
    fixed clinical classes; no outcome-derived vocabulary is fitted.
    """
    total: Dict[int, int] = {}
    unique: Dict[int, set] = {}
    category_counts: Dict[str, Dict[int, int]] = {name: {} for name in MEDICATION_PATTERNS}
    path = table_path(cfg, "hosp", "emar")
    columns = ["subject_id", "hadm_id", "charttime", "storetime", "medication", "event_txt"]
    for chunk in pd.read_csv(require_file(path), usecols=columns, chunksize=cfg.csv_chunksize, low_memory=False):
        chunk = event_window(chunk, cohort, "charttime", "storetime")
        if chunk.empty:
            continue
        status = chunk.event_txt.astype("string")
        chunk = chunk[~status.str.contains(r"not given|held|refused|cancel", case=False, regex=True, na=False)].copy()
        if chunk.empty:
            continue
        medication = chunk.medication.astype("string").str.strip().str.lower()
        chunk = chunk[medication.notna() & medication.ne("")].copy()
        chunk["_medication"] = medication.loc[chunk.index]
        for sid, group in chunk.groupby("subject_id"):
            subject_id = int(sid)
            total[subject_id] = total.get(subject_id, 0) + len(group)
            unique.setdefault(subject_id, set()).update(group["_medication"].tolist())
        for name, pattern in MEDICATION_PATTERNS.items():
            matched = chunk["_medication"].str.contains(pattern, case=False, regex=True, na=False)
            for sid, count in chunk.loc[matched].groupby("subject_id").size().items():
                category_counts[name][int(sid)] = category_counts[name].get(int(sid), 0) + int(count)

    result = pd.DataFrame(index=cohort.subject_id.astype(int)); result.index.name = "subject_id"
    result["idx_medication_event_count"] = pd.Series(total)
    result["idx_unique_medication_count"] = pd.Series({sid: len(names) for sid, names in unique.items()})
    for name, counts in category_counts.items():
        result[f"idx_{name}_event_count"] = pd.Series(counts)
        result[f"idx_{name}_used"] = result[f"idx_{name}_event_count"].gt(0).astype(np.int8)
    return result



def aggregate_microbiology(cfg: PipelineConfig, cohort: pd.DataFrame) -> pd.DataFrame:
    """Summarize microbiology results that were stored by the landmark."""
    test_count: Dict[int, int] = {}
    positive_count: Dict[int, int] = {}
    unique_tests: Dict[int, set] = {}
    columns = ["subject_id", "hadm_id", "charttime", "storetime", "test_name", "org_name"]
    path = table_path(cfg, "hosp", "microbiologyevents")
    for chunk in pd.read_csv(require_file(path), usecols=columns, chunksize=cfg.csv_chunksize, low_memory=False):
        # Some cultures have only a stored result time.  Falling back to that
        # time is conservative because it cannot make a result available early.
        chunk["charttime"] = pd.to_datetime(chunk.charttime, errors="coerce").fillna(
            pd.to_datetime(chunk.storetime, errors="coerce")
        )
        chunk = event_window(chunk, cohort, "charttime", "storetime")
        if chunk.empty:
            continue
        for sid, group in chunk.groupby("subject_id"):
            subject_id = int(sid)
            test_count[subject_id] = test_count.get(subject_id, 0) + len(group)
            positive_count[subject_id] = positive_count.get(subject_id, 0) + int(group.org_name.notna().sum())
            unique_tests.setdefault(subject_id, set()).update(group.test_name.dropna().astype(str))
    result = pd.DataFrame(index=cohort.subject_id.astype(int)); result.index.name = "subject_id"
    result["idx_microbiology_test_count"] = pd.Series(test_count)
    result["idx_microbiology_positive_count"] = pd.Series(positive_count)
    result["idx_microbiology_unique_test_count"] = pd.Series({sid: len(values) for sid, values in unique_tests.items()})
    result["idx_microbiology_positive"] = result.idx_microbiology_positive_count.gt(0).astype(np.int8)
    return result



def stable_bucket(value: str, buckets: int = 64) -> int:
    return int(hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest(), 16) % buckets



def aggregate_history(cfg: PipelineConfig, cohort: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    index_times = cohort[["subject_id", "admittime", "intime"]].rename(
        columns={"admittime": "index_admittime", "intime": "index_intime"},
    )
    prior = admissions.merge(index_times, on="subject_id", validate="many_to_one")
    prior = prior[prior.dischtime.notna() & prior.dischtime.lt(prior.index_admittime)].copy()
    result = pd.DataFrame(index=cohort.subject_id.astype(int)); result.index.name = "subject_id"
    result["prior_admission_count"] = prior.groupby("subject_id").hadm_id.nunique()
    last_discharge = prior.groupby("subject_id").dischtime.max()
    index_admittime = cohort.set_index("subject_id").admittime
    result["days_since_prior_discharge"] = (index_admittime - last_discharge).dt.total_seconds() / 86400.0
    prior["_days_before_index"] = (prior.index_admittime - prior.dischtime).dt.total_seconds() / 86400.0
    prior["_duration_days"] = (prior.dischtime - prior.admittime).dt.total_seconds().clip(lower=0) / 86400.0
    result["prior_hospital_days_sum"] = prior.groupby("subject_id")._duration_days.sum()
    result["prior_hospital_days_max"] = prior.groupby("subject_id")._duration_days.max()
    for days in (30, 90, 365):
        recent = prior[prior._days_before_index.le(days)]
        result[f"prior_admission_count_{days}d"] = recent.groupby("subject_id").hadm_id.nunique()
    prior_keys = prior[["subject_id", "hadm_id"]].drop_duplicates()
    for table, prefix in (("diagnoses_icd", "dx"), ("procedures_icd", "proc")):
        data = pd.read_csv(require_file(table_path(cfg, "hosp", table)), usecols=["subject_id", "hadm_id", "icd_code", "icd_version"], low_memory=False)
        data = data.merge(prior_keys, on=["subject_id", "hadm_id"], how="inner", validate="many_to_one")
        result[f"prior_{prefix}_count"] = data.groupby("subject_id").size()
        result[f"prior_{prefix}_unique"] = data.groupby("subject_id").icd_code.nunique()
        if not data.empty:
            bucket = data.apply(lambda row: stable_bucket(f"{row.icd_version}:{row.icd_code}"), axis=1)
            data = data.assign(_bucket=bucket)
            pivot = data.groupby(["subject_id", "_bucket"]).size().unstack(fill_value=0)
            pivot = pivot.reindex(columns=range(64), fill_value=0)
            pivot.columns = [f"prior_{prefix}_hash_{i:02d}" for i in range(64)]
            result = result.join(pivot, how="left")
    return result



def derive_clinical_interactions(
    cohort: pd.DataFrame,
    labs: pd.DataFrame,
    vitals: pd.DataFrame,
    support: pd.DataFrame,
    observation_hours: int,
) -> pd.DataFrame:
    """Add fixed, clinically standard interactions without fitting labels."""
    index = cohort.subject_id.astype(int)
    result = pd.DataFrame(index=index); result.index.name = "subject_id"

    def value(frame: pd.DataFrame, column: str) -> pd.Series:
        if column in frame:
            return pd.to_numeric(frame[column], errors="coerce").reindex(index)
        return pd.Series(np.nan, index=index, dtype=float)

    def ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        output = numerator / denominator.where(denominator.abs().gt(1e-6))
        return output.where(np.isfinite(output))

    result["idx_shock_index_last"] = ratio(
        value(vitals, "idx_heart_rate_last"), value(vitals, "idx_sbp_last")
    )
    result["idx_shock_index_worst_proxy"] = ratio(
        value(vitals, "idx_heart_rate_max"), value(vitals, "idx_sbp_min")
    )
    result["idx_gcs_total_last"] = sum(
        value(vitals, f"idx_gcs_{component}_last")
        for component in ("eye", "verbal", "motor")
    )
    result["idx_gcs_total_worst_proxy"] = sum(
        value(vitals, f"idx_gcs_{component}_min")
        for component in ("eye", "verbal", "motor")
    )
    result["idx_pao2_fio2_worst_proxy"] = 100.0 * ratio(
        value(labs, "idx_po2_min"), value(vitals, "idx_fio2_max")
    )
    result["idx_spo2_fio2_worst_proxy"] = 100.0 * ratio(
        value(vitals, "idx_spo2_min"), value(vitals, "idx_fio2_max")
    )
    result["idx_bun_creatinine_ratio_last"] = ratio(
        value(labs, "idx_bun_last"), value(labs, "idx_creatinine_last")
    )
    result["idx_map_deficit_worst_proxy"] = (65.0 - value(vitals, "idx_mbp_min")).clip(lower=0.0)
    result["idx_albumin_corrected_anion_gap"] = (
        value(labs, "idx_anion_gap_max")
        + 2.5 * (4.0 - value(labs, "idx_albumin_min"))
    )
    result["idx_urine_output_per_hour"] = (
        value(support, "idx_urine_output_sum") / float(observation_hours)
    )
    return result



class ClinicalTextEncoder:
    def __init__(self, model_name: str, batch_size: int, stride: int):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if batch_size <= 0 or stride <= 0:
            raise ValueError("text batch size and chunk stride must be positive")
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.batch_size = batch_size
        self.stride = stride
        self.max_tokens = min(int(getattr(self.tokenizer, "model_max_length", 512)), 512)
        self.hidden_size = int(self.model.config.hidden_size)

    @staticmethod
    def clean(text: object) -> str:
        if not isinstance(text, str): return ""
        text = re.sub(r"\[\*\*.*?\*\*\]", " [PHI] ", text, flags=re.DOTALL)
        text = re.sub(r"[_=*\-]{4,}", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def chunks(self, text: str) -> List[List[int]]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        capacity = self.max_tokens - self.tokenizer.num_special_tokens_to_add(pair=False)
        if not ids: return []
        step = min(max(self.stride, 1), capacity)
        return [ids[start:start + capacity] for start in range(0, len(ids), step)]

    def encode_documents(self, texts: Sequence[object]) -> List[Optional[np.ndarray]]:
        pieces: List[List[int]] = []
        owners: List[int] = []
        for owner, text in enumerate(texts):
            document_pieces = self.chunks(self.clean(text))
            pieces.extend(document_pieces)
            owners.extend([owner] * len(document_pieces))
        document_vectors: List[List[np.ndarray]] = [[] for _ in texts]
        for start in range(0, len(pieces), self.batch_size):
            batch_pieces = pieces[start:start + self.batch_size]
            prepared = [self.tokenizer.prepare_for_model(piece, add_special_tokens=True, truncation=True, max_length=self.max_tokens) for piece in batch_pieces]
            batch = self.tokenizer.pad(prepared, padding=True, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with self.torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                pooled = self.torch.nn.functional.normalize(pooled, dim=1)
            for owner, vector in zip(owners[start:start + self.batch_size], pooled.cpu().numpy()):
                document_vectors[owner].append(vector)
        results: List[Optional[np.ndarray]] = []
        for vectors in document_vectors:
            if not vectors:
                results.append(None)
                continue
            vector = np.mean(np.stack(vectors), axis=0)
            norm = np.linalg.norm(vector)
            results.append((vector / norm if norm > 0 else vector).astype(np.float32))
        return results

    def encode_document(self, text: object) -> Optional[np.ndarray]:
        return self.encode_documents([text])[0]

