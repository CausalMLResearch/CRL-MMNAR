import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CXR_DIR, LABEL, POOL, SNAPSHOT, SPLIT_SEED, SUPPLEMENT, TASKS, log, notes_by_patient

COHORT_SIZE, COHORT_SEED = 20000, 42
MORTALITY_LABEL = "mortality_after_discharge_90d"
LABELS = ["readmission_30d", "readmission_90d", "icu_need_after_discharge_90d", MORTALITY_LABEL]
YEAR_GROUPS = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"]
EXPORT = {
    "mimiciv_3_1_hosp.admissions": ("subject_id,hadm_id,admittime,dischtime,deathtime,admission_type,admit_provider_id,"
                                    "admission_location,discharge_location,insurance,language,marital_status,race,edregtime,"
                                    "edouttime,hospital_expire_flag", ""),
    "mimiciv_3_1_hosp.patients": ("subject_id,gender,anchor_age,anchor_year,anchor_year_group,dod", ""),
    "mimiciv_3_1_hosp.diagnoses_icd": ("subject_id,hadm_id,seq_num,icd_code,icd_version", ""),
    "mimiciv_3_1_hosp.transfers": ("subject_id,hadm_id,transfer_id,eventtype,careunit,intime,outtime", ""),
    "mimiciv_3_1_hosp.labevents": ("labevent_id,subject_id,hadm_id,specimen_id,itemid,charttime,storetime,value,valuenum,"
                                   "valueuom,ref_range_lower,ref_range_upper,flag,priority",
                                   "itemid IN (50801, 50802, 50803, 50804, 50805, 50806, 50808, 50809, 50810, 50811, 50813, "
                                   "50814, 50815, 50816, 50817, 50818, 50819, 50820, 50821, 50822, 50823, 50824, 50825, 50861, "
                                   "50862, 50863, 50868, 50878, 50882, 50885, 50893, 50902, 50912, 50931, 50971, 50983, 51006, "
                                   "51221, 51222, 51237, 51265, 51274, 51275, 51301, 52033)"),
    "mimiciv_3_1_icu.icustays": ("subject_id,hadm_id,stay_id,first_careunit,last_careunit,intime,outtime,los", ""),
    "mimiciv_3_1_icu.chartevents": ("subject_id,hadm_id,stay_id,charttime,storetime,itemid,value,valuenum,valueuom,warning",
                                    "itemid IN (220045, 220050, 220051, 220052, 220179, 220180, 220181, 220210, 220277, 220339, "
                                    "220621, 220739, 223761, 223762, 223834, 223835, 223848, 223849, 223900, 223901, 224639, "
                                    "224684, 224685, 224686, 224687, 224688, 224689, 224690, 224691, 224696, 224700, 225309, "
                                    "225310, 225312, 225664, 226512, 226537, 226707, 226730, 226732, 227287, 227582, 229314) "
                                    "AND stay_id IS NOT NULL"),
    "mimiciv_3_1_derived.charlson": ("subject_id,hadm_id,age_score,myocardial_infarct,congestive_heart_failure,"
                                     "peripheral_vascular_disease,cerebrovascular_disease,dementia,chronic_pulmonary_disease,"
                                     "rheumatic_disease,peptic_ulcer_disease,mild_liver_disease,diabetes_without_cc,"
                                     "diabetes_with_cc,paraplegia,renal_disease,malignant_cancer,severe_liver_disease,"
                                     "metastatic_solid_tumor,aids,charlson_comorbidity_index", ""),
    "mimiciv_3_1_derived.sofa": ("stay_id,hr,starttime,endtime,pao2fio2ratio_novent,pao2fio2ratio_vent,rate_epinephrine,"
                                 "rate_norepinephrine,rate_dopamine,rate_dobutamine,meanbp_min,gcs_min,uo_24hr,bilirubin_max,"
                                 "creatinine_max,platelet_min,respiration,coagulation,liver,cardiovascular,cns,renal,"
                                 "respiration_24hours,coagulation_24hours,liver_24hours,cardiovascular_24hours,cns_24hours,"
                                 "renal_24hours,sofa_24hours", ""),
    "mimiciv_3_1_derived.ventilation": ("stay_id,starttime,endtime,ventilation_status", ""),
    "mimiciv_3_1_derived.vasoactive_agent": ("stay_id,starttime,endtime,dopamine,epinephrine,norepinephrine,phenylephrine,"
                                             "vasopressin,dobutamine,milrinone", ""),
    "mimiciv_3_1_derived.rrt": ("stay_id,charttime,dialysis_present,dialysis_active,dialysis_type", ""),
}
SUPPLEMENT_TABLES = {
    "prescriptions": ("subject_id, hadm_id, starttime, stoptime, drug, drug_type, route", "hadm"),
    "procedures_icd": ("subject_id, hadm_id, seq_num, chartdate, icd_code, icd_version", "hadm"),
    "drgcodes": ("*", "hadm"),
    "services": ("subject_id, hadm_id, transfertime, prev_service, curr_service", "hadm"),
    "microbiologyevents": ("subject_id, hadm_id, chartdate, charttime, storetime, spec_type_desc, org_name, interpretation", "hadm"),
    "emar": ("subject_id, hadm_id, charttime, storetime, medication, event_txt", "hadm"),
    "omr": ("subject_id, chartdate, result_name, result_value", "subject"),
}
SELECT_SQL = """
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
CLASSES = {
    "anticoagulant": r"heparin|enoxaparin|warfarin|apixaban|rivaroxaban|dabigatran|argatroban",
    "insulin": r"insulin", "opioid": r"fentanyl|morphine|hydromorphone|oxycodone|methadone|tramadol",
    "diuretic": r"furosemide|bumetanide|torsemide|metolazone|spironolactone|hydrochlorothiazide",
    "antibiotic": r"vancomycin|cef|cillin|meropenem|ertapenem|azithromycin|doxycycline|floxacin|metronidazole|clindamycin|linezolid|daptomycin",
    "steroid": r"prednisone|methylprednisolone|hydrocortisone|dexamethasone",
    "antipsychotic": r"haloperidol|quetiapine|olanzapine|risperidone",
    "anticonvulsant": r"levetiracetam|phenytoin|valproate|lacosamide",
    "immunosuppressant": r"tacrolimus|mycophenolate|cyclosporine|sirolimus|azathioprine",
    "chemotherapy": r"cisplatin|carboplatin|oxaliplatin|cyclophosphamide|doxorubicin|paclitaxel|methotrexate|cytarabine|gemcitabine|fluorouracil",
    "antiarrhythmic": r"amiodarone|sotalol|dofetilide", "vasopressor": r"norepinephrine|vasopressin|phenylephrine|epinephrine|dopamine",
    "beta_blocker": r"metoprolol|carvedilol|atenolol|labetalol", "statin": r"statin",
    "benzodiazepine": r"lorazepam|diazepam|alprazolam|clonazepam|midazolam",
}
LAB_SPECS = {
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


def clean_fio2(value):
    if 0.20 <= value <= 1.0:
        return value * 100.0
    if 20.0 <= value <= 100.0:
        return value
    return float("nan")


VITAL_SPECS = {
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


def table(name, columns, filt=None):
    import pyarrow.dataset as ds
    return ds.dataset(SNAPSHOT / name, format="parquet").to_table(columns=columns, filter=filt).to_pandas()


def write(frame, name):
    path = POOL / name
    frame.to_parquet(path, index=False)
    print("wrote", path, frame.shape, flush=True)


def derive_mortality_labels(cohort, patients):
    for frame, columns in ((cohort, {"subject_id", "dischtime"}), (patients, {"subject_id", "dod"})):
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"Mortality labeling is missing columns: {sorted(missing)}")
    if cohort.subject_id.duplicated().any() or patients.subject_id.duplicated().any():
        raise ValueError("Mortality labeling requires unique patient identifiers")
    if not set(cohort.subject_id).issubset(set(patients.subject_id)):
        raise ValueError("Patients table does not cover the mortality cohort")
    if "hospital_expire_flag" in cohort and cohort.hospital_expire_flag.fillna(0).ne(0).any():
        raise ValueError("Post-discharge mortality requires a surviving index discharge")
    discharge = pd.to_datetime(cohort.dischtime, errors="raise").dt.normalize()
    if discharge.isna().any():
        raise ValueError("Mortality labeling requires a discharge date")
    death_dates = pd.to_datetime(patients.dod, errors="raise")
    deaths = cohort.subject_id.map(pd.Series(death_dates.to_numpy(), index=patients.subject_id)).dt.normalize()
    days = (deaths - discharge).dt.days
    if days.lt(0).any():
        raise ValueError("A recorded death date precedes a surviving discharge")
    return days.between(0, 90, inclusive="both").astype("int8").rename(MORTALITY_LABEL)


def relationship_strata(labels, indices, minimum):
    readmission = labels["readmission_30d"][indices].astype(int)
    readmission_90d = labels["readmission_90d"][indices].astype(int)
    icu = labels["icu_need_after_discharge_90d"][indices].astype(int)
    candidates = [
        np.column_stack([readmission, readmission_90d, icu]),
        np.column_stack([readmission, icu]),
        np.column_stack([readmission_90d, icu]),
        readmission.reshape(-1, 1),
    ]
    for components in candidates:
        _, encoded = np.unique(components, axis=0, return_inverse=True)
        if np.bincount(encoded).min() >= minimum:
            return encoded
    raise ValueError("No statistically viable relationship-aware stratification exists")


def make_splits(labels, seed):
    from sklearn.model_selection import train_test_split
    all_indices = np.arange(len(labels["readmission_30d"]))
    strata = relationship_strata(labels, all_indices, minimum=5)
    train_val, test = train_test_split(all_indices, test_size=0.20, random_state=seed, stratify=strata)
    inner_strata = relationship_strata(labels, train_val, minimum=2)
    train, validation = train_test_split(train_val, test_size=0.15, random_state=seed, stratify=inner_strata)
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise RuntimeError("Subject split overlap detected")
    for split_name, split in (("train", train), ("validation", validation), ("test", test)):
        for task in TASKS:
            if set(np.unique(labels[LABEL[task]][split]).tolist()) != {0.0, 1.0}:
                raise ValueError(f"{split_name} holdout has fewer than two {task} classes")
    return np.asarray(train), np.asarray(validation), np.asarray(test)


def task_strata(labels, rows, minimum=2, max_classes=None):
    targets = np.column_stack([labels[LABEL[t]][rows].astype(int) for t in TASKS])
    for columns in (targets, targets[:, [0, 1]], targets[:, [2]], targets[:, [0]]):
        _, encoded = np.unique(columns, axis=0, return_inverse=True)
        if np.bincount(encoded).min() >= minimum and (max_classes is None or len(np.unique(encoded)) <= max_classes):
            return encoded
    raise ValueError("Not enough patients for stratification")


def split_fit_stop(labels, rows, fraction, seed):
    from sklearn.model_selection import train_test_split
    stop_n = int(np.ceil(len(rows) * fraction))
    fitting, stopping = train_test_split(rows, test_size=fraction, random_state=seed,
                                         stratify=task_strata(labels, rows, max_classes=min(stop_n, len(rows) - stop_n)))
    return np.sort(fitting), np.sort(stopping)


def assert_disjoint(*groups):
    for i, group in enumerate(groups):
        group = np.asarray(group)
        if len(group) != len(np.unique(group)):
            raise ValueError("Duplicate patient rows")
        for other in groups[i + 1:]:
            if np.intersect1d(group, other).size:
                raise ValueError("Patient boundary violation")


def export(args):
    import concurrent.futures as cf
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from google.cloud import bigquery_storage_v1
    from google.cloud.bigquery_storage_v1 import types
    client = bigquery_storage_v1.BigQueryReadClient()

    def normalise(batch):
        columns, names = [], []
        for name, column in zip(batch.schema.names, batch.columns):
            if pa.types.is_decimal(column.type):
                column = pc.cast(column, pa.float64())
            columns.append(column)
            names.append(name.lower())
        return pa.RecordBatch.from_arrays(columns, names=names)

    def read(session, stream, path):
        writer = None
        for page in client.read_rows(stream).rows(session).pages:
            batch = normalise(page.to_arrow())
            if writer is None:
                writer = pq.ParquetWriter(path, batch.schema, compression="zstd")
            writer.write_batch(batch)
        if writer is not None:
            writer.close()

    for name, (fields, restriction) in EXPORT.items():
        dataset, name_ = name.split(".")
        out = SNAPSHOT / dataset / name_
        out.mkdir(parents=True, exist_ok=True)
        options = types.ReadSession.TableReadOptions(selected_fields=fields.split(","), row_restriction=restriction)
        session = client.create_read_session(
            parent=f"projects/{args.project}",
            read_session=types.ReadSession(table=f"projects/physionet-data/datasets/{dataset}/tables/{name_}",
                                           data_format=types.DataFormat.ARROW, read_options=options),
            max_stream_count=4)
        with cf.ThreadPoolExecutor(max_workers=max(1, len(session.streams))) as pool:
            list(pool.map(lambda s: read(session, s[1].name, out / f"part-{s[0]:03d}.parquet"), enumerate(session.streams)))
        log("exported", name)


def select(args):
    from google.cloud import bigquery
    client = bigquery.Client(project=args.project)
    job = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("seed", "INT64", COHORT_SEED),
        bigquery.ScalarQueryParameter("cohort_size", "INT64", COHORT_SIZE),
    ])
    ids = client.query(SELECT_SQL, job_config=job).to_dataframe()[["subject_id"]]
    if len(ids) != COHORT_SIZE or ids.subject_id.duplicated().any():
        raise RuntimeError(f"Expected {COHORT_SIZE} unique post-discharge subjects, found {len(ids)}")
    POOL.mkdir(parents=True, exist_ok=True)
    ids.to_csv(POOL / "selected_subject_ids.csv", index=False)
    log("selected", len(ids))


def labels(args):
    adm = table("mimiciv_3_1_hosp/admissions", None)
    pat = table("mimiciv_3_1_hosp/patients", None)
    icu = table("mimiciv_3_1_icu/icustays", ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "first_careunit"])
    adm = adm[adm.admittime.notna() & adm.dischtime.notna()]
    index = adm.sort_values(["subject_id", "admittime", "hadm_id"]).drop_duplicates("subject_id").merge(
        pat[["subject_id", "anchor_age", "anchor_year", "anchor_year_group", "gender", "dod"]], on="subject_id")
    age = index.anchor_age + index.admittime.dt.year - index.anchor_year
    keep = (age >= 18) & index.anchor_year_group.isin(YEAR_GROUPS) & index.hospital_expire_flag.fillna(0).eq(0)
    cohort = index[keep].copy()
    cohort["age_at_admission"] = age[keep]
    future = adm.merge(cohort[["subject_id", "hadm_id", "dischtime"]].rename(
        columns={"hadm_id": "index_hadm_id", "dischtime": "index_dischtime"}), on="subject_id", validate="many_to_one")
    delta = (future.admittime - future.index_dischtime).dt.total_seconds() / 86400.0
    future = future[(future.hadm_id != future.index_hadm_id) & delta.gt(0)].copy()
    future["days"] = delta[delta.gt(0)]
    future_icu = icu.merge(future[["subject_id", "hadm_id", "index_dischtime"]].drop_duplicates(),
                           on=["subject_id", "hadm_id"], how="inner", validate="many_to_one")
    icu_delta = (future_icu.intime - future_icu.index_dischtime).dt.total_seconds() / 86400.0
    sid = cohort.subject_id.astype(int)
    cohort["readmission_30d"] = sid.isin(set(future.loc[future.days.le(30), "subject_id"])).astype(np.int8)
    cohort["readmission_90d"] = sid.isin(set(future.loc[future.days.le(90), "subject_id"])).astype(np.int8)
    cohort["icu_need_after_discharge_90d"] = sid.isin(set(future_icu.loc[icu_delta.gt(0) & icu_delta.le(90), "subject_id"])).astype(np.int8)
    days = (pd.to_datetime(cohort.dod).dt.normalize() - cohort.dischtime.dt.normalize()).dt.days
    inconsistent = days.lt(0)
    cohort = cohort[~inconsistent].copy()
    cohort[MORTALITY_LABEL] = derive_mortality_labels(cohort[["subject_id", "dischtime"]], pat[pat.subject_id.isin(cohort.subject_id)]).to_numpy()

    selected = pd.read_csv(POOL / "selected_subject_ids.csv", usecols=["subject_id"]).subject_id.astype(np.int64)
    if not selected.isin(cohort.subject_id).all():
        raise SystemExit("Selected cohort is not contained in the eligible population")
    pinned = cohort.set_index("subject_id").loc[np.sort(selected.to_numpy())]
    ids = pinned.index.to_numpy().astype(np.int64)
    lab = {c: pinned[c].to_numpy().astype(np.float32) for c in LABELS}
    train, validation, test = (np.sort(r) for r in make_splits(lab, SPLIT_SEED))
    assert_disjoint(train, validation, test)
    if not np.array_equal(np.sort(np.concatenate([train, validation, test])), np.arange(len(ids))):
        raise ValueError("Patient split is not a partition of the cohort")
    selection, calibration = split_fit_stop(lab, validation, 0.5, SPLIT_SEED)
    np.savez_compressed(POOL / "cohort_split.npz", subject_id=ids, train=train, validation=validation, test=test,
                        selection=selection, calibration=calibration)
    part = pd.Series("pool_extra", index=cohort.subject_id.to_numpy())
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        part.loc[ids[rows]] = "cohort_" + name
    cohort["partition"] = part.reindex(cohort.subject_id.to_numpy()).to_numpy()
    log(cohort.partition.value_counts().to_dict())
    columns = ["subject_id", "hadm_id", "admittime", "dischtime", "edregtime", "edouttime", "admission_type",
               "admission_location", "discharge_location", "insurance", "language", "marital_status", "race",
               "gender", "anchor_year_group", "age_at_admission", *LABELS, "partition"]
    write(cohort[columns].sort_values("subject_id").reset_index(drop=True), "cohort.parquet")


def cxr_images(args):
    import tensorflow as tf

    def parse_study_datetime(study_date, study_time):
        if pd.isna(study_date) or pd.isna(study_time):
            return pd.NaT
        date_digits = re.sub(r"\D", "", str(study_date).split(".")[0])
        time_digits = re.sub(r"\D", "", str(study_time).split(".")[0]).zfill(6)[:6]
        if len(date_digits) != 8 or len(time_digits) != 6:
            return pd.NaT
        return pd.to_datetime(date_digits + time_digits, format="%Y%m%d%H%M%S", errors="coerce")

    cohort = pd.read_parquet(POOL / "cohort.parquet", columns=["subject_id", "admittime", "dischtime"])
    meta = pd.read_csv(CXR_DIR / "mimic-cxr-2.0.0-metadata.csv.gz", low_memory=False)
    meta["view"] = meta.ViewPosition.astype("string").str.strip().str.upper()
    meta = meta[meta.view.isin({"AP", "PA"})].copy()
    meta["t"] = [parse_study_datetime(d, s) for d, s in zip(meta.StudyDate, meta.StudyTime)]
    meta = meta.merge(cohort, on="subject_id")
    meta = meta[meta.t.notna() & meta.t.ge(meta.admittime) & meta.t.le(meta.dischtime)]
    paths = [str(CXR_DIR / f"p{s // 1_000_000}" / f"p{s}" / f"s{st}" / f"{d}.tfrecord")
             for s, st, d in zip(meta.subject_id, meta.study_id, meta.dicom_id)]
    keep = np.array([Path(p).exists() for p in paths])
    meta, paths = meta[keep].reset_index(drop=True), [p for p, k in zip(paths, keep) if k]
    spec = {"embedding": tf.io.VarLenFeature(tf.float32)}
    ds = tf.data.TFRecordDataset(paths).map(lambda r: tf.sparse.to_dense(tf.io.parse_single_example(r, spec)["embedding"]))
    emb = np.stack([e.numpy() for e in ds]).astype(np.float32)
    if len(emb) != len(paths):
        raise ValueError("one embedding per image expected")
    np.savez(POOL / "cxr_images.npz", subject_id=meta.subject_id.to_numpy(), study_id=meta.study_id.to_numpy(), X=emb,
             hours_to_discharge=((meta.dischtime - meta.t).dt.total_seconds() / 3600).to_numpy(np.float32))
    log("images", emb.shape, "patients", meta.subject_id.nunique())


def flags(args):
    with np.load(POOL / "cohort_split.npz") as z:
        ids = z["subject_id"]
    pool = pd.read_parquet(POOL / "cohort.parquet")
    window = pool[["subject_id", "hadm_id", "admittime", "dischtime"]]
    wanted = set(ids.tolist())
    has = {}
    for name, source in (("text", "discharge"), ("rad", "radiology")):
        docs = notes_by_patient(source, window, wanted).reindex(ids).fillna("").to_numpy()
        has[name] = docs != ""
    with np.load(POOL / "cxr_images.npz") as z:
        has_img = np.isin(ids, z["subject_id"])
    observed = np.column_stack([np.ones(len(ids)), has_img, has["text"], has["rad"]]).astype(np.float32)
    np.savez_compressed(POOL / "cohort_flags.npz", subject_id=ids, flags=observed)
    log("flags: CXR", int(has_img.sum()), "discharge", int(has["text"].sum()), "radiology", int(has["rad"].sum()))


def windowed(name, columns, cohort, time_col, items=None, pre_hours=0):
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    filt = pc.field("subject_id").isin(cohort.subject_id.to_numpy())
    if items is not None:
        filt = filt & pc.field("itemid").isin(sorted(items))
    frames = []
    for batch in ds.dataset(SNAPSHOT / name, format="parquet").to_batches(columns=columns, filter=filt, batch_size=4_000_000):
        df = batch.to_pandas().merge(cohort[["subject_id", "admittime", "dischtime"]], on="subject_id")
        start = df.admittime - pd.Timedelta(hours=pre_hours)
        df = df[(df[time_col] >= start) & (df[time_col] <= df.dischtime)]
        frames.append(df.drop(columns=["admittime"]))
    return pd.concat(frames, ignore_index=True)


def summarize(df, key, value, time_col, prefix):
    df = df.sort_values(["subject_id", key, time_col])
    g = df.groupby(["subject_id", key])[value]
    stats = pd.DataFrame({"n": g.size(), "mean": g.mean(), "min": g.min(), "max": g.max(), "first": g.first(), "last": g.last()})
    stats["delta"] = stats["last"] - stats["first"]
    recent = df[df[time_col] >= df.dischtime - pd.Timedelta(hours=24)].groupby(["subject_id", key])[value]
    stats["last24_mean"] = recent.mean()
    wide = stats.unstack(key)
    wide.columns = [f"{prefix}_{name}_{stat}" for stat, name in wide.columns]
    return wide


def one_hot(series, prefix, min_count=50):
    values = series.fillna("UNKNOWN").astype(str)
    common = values.value_counts()
    values = values.where(values.isin(common[common >= min_count].index), "OTHER")
    return pd.get_dummies(values, prefix=prefix, dtype=np.float32)


def features(args):
    import pyarrow.compute as pc
    cohort = pd.read_parquet(POOL / "cohort.parquet")
    train_mask = cohort.partition.isin(["pool_extra", "cohort_train"])
    feats = [cohort.set_index("subject_id")[["age_at_admission"]].astype(np.float32)]
    base = cohort.set_index("subject_id")
    feats.append(pd.DataFrame({
        "los_hours": (base.dischtime - base.admittime).dt.total_seconds() / 3600,
        "ed_hours": (base.edouttime - base.edregtime).dt.total_seconds() / 3600,
        "has_ed": base.edregtime.notna().astype(np.float32),
        "female": base.gender.eq("F").astype(np.float32)}, index=base.index))
    race = base.race.fillna("UNKNOWN").str.split(r" - |/", regex=True).str[0]
    for col, series in (("admission_type", base.admission_type), ("admission_location", base.admission_location),
                        ("discharge_location", base.discharge_location), ("insurance", base.insurance),
                        ("language", base.language), ("marital", base.marital_status), ("race", race),
                        ("year_group", base.anchor_year_group)):
        feats.append(one_hot(series, col).set_index(base.index))
    print("admission features", flush=True)

    keys = cohort[["subject_id", "hadm_id"]]
    hadm = pc.field("hadm_id").isin(cohort.hadm_id.to_numpy())
    dx = table("mimiciv_3_1_hosp/diagnoses_icd", ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"], hadm).merge(keys)
    dx["cat"] = "v" + dx.icd_version.astype(str) + "_" + dx.icd_code.str.strip().str[:3]
    train_ids = set(cohort.loc[train_mask, "subject_id"])
    counts = dx[dx.subject_id.isin(train_ids)].drop_duplicates(["subject_id", "cat"]).cat.value_counts()
    vocab = counts[counts >= 200].index
    present = dx[dx.cat.isin(vocab)].drop_duplicates(["subject_id", "cat"])
    feats.append(pd.crosstab(present.subject_id, "dx_" + present.cat).astype(np.float32))
    primary = dx[dx.seq_num == 1]
    pcounts = primary[primary.subject_id.isin(train_ids)].cat.value_counts()
    pvocab = pcounts[pcounts >= 100].index
    pp = primary[primary.cat.isin(pvocab)]
    feats.append(pd.crosstab(pp.subject_id, "dxprimary_" + pp.cat).astype(np.float32))
    feats.append(dx.groupby("subject_id").size().rename("dx_count").astype(np.float32).to_frame())
    charlson = table("mimiciv_3_1_derived/charlson", None, hadm).merge(keys).drop(columns=["hadm_id"]).set_index("subject_id")
    feats.append(charlson.add_prefix("charlson_").astype(np.float32))
    print("diagnoses", len(vocab), "categories;", len(pvocab), "primary", flush=True)

    tr = table("mimiciv_3_1_hosp/transfers", ["subject_id", "hadm_id", "eventtype", "careunit", "intime", "outtime"], hadm).merge(keys)
    tr = tr.sort_values(["subject_id", "intime"])
    g = tr.groupby("subject_id")
    feats.append(pd.DataFrame({"transfer_count": g.size(), "careunit_distinct": g.careunit.nunique()}).astype(np.float32))
    last_unit = tr[tr.eventtype != "discharge"].groupby("subject_id").careunit.last()
    feats.append(one_hot(last_unit.reindex(base.index), "last_unit", 200).set_index(base.index))

    stays = table("mimiciv_3_1_icu/icustays", ["subject_id", "hadm_id", "stay_id", "intime", "outtime"], hadm).merge(keys)
    stays = stays.merge(base[["dischtime"]], left_on="subject_id", right_index=True)
    stays = stays[stays.intime <= stays.dischtime]
    stays["hours"] = (stays.outtime.where(stays.outtime < stays.dischtime, stays.dischtime) - stays.intime).dt.total_seconds() / 3600
    sg = stays.groupby("subject_id")
    feats.append(pd.DataFrame({"icu_stays": sg.size(), "icu_hours": sg.hours.sum(),
        "hours_icu_out_to_discharge": (sg.dischtime.first() - sg.outtime.max()).dt.total_seconds() / 3600}).astype(np.float32))
    stay_ids = pc.field("stay_id").isin(stays.stay_id.to_numpy())
    cut = stays[["stay_id", "subject_id", "dischtime"]]
    sofa = table("mimiciv_3_1_derived/sofa", ["stay_id", "endtime", "sofa_24hours", "respiration_24hours", "cardiovascular_24hours",
                                              "renal_24hours", "cns_24hours", "liver_24hours", "coagulation_24hours"], stay_ids).merge(cut)
    sofa = sofa[sofa.endtime <= sofa.dischtime].sort_values(["subject_id", "endtime"])
    comp = [c for c in sofa.columns if c.endswith("_24hours")]
    feats.append(sofa.groupby("subject_id")[comp].max().add_prefix("max_").astype(np.float32))
    feats.append(sofa.groupby("subject_id")[comp].last().add_prefix("last_").astype(np.float32))
    for name, cols, label in (("ventilation", ["stay_id", "starttime", "endtime", "ventilation_status"], "vent"),
                              ("vasoactive_agent", ["stay_id", "starttime", "endtime"], "vaso")):
        ev = table("mimiciv_3_1_derived/" + name, cols, stay_ids).merge(cut)
        ev = ev[ev.starttime <= ev.dischtime]
        ev["hours"] = (ev.endtime.where(ev.endtime < ev.dischtime, ev.dischtime) - ev.starttime).dt.total_seconds() / 3600
        if label == "vent":
            wide = ev.pivot_table(index="subject_id", columns="ventilation_status", values="hours", aggfunc="sum")
            wide.columns = ["vent_hours_" + str(c).replace(" ", "_") for c in wide.columns]
            feats.append(wide.astype(np.float32))
        else:
            feats.append(ev.groupby("subject_id").hours.sum().rename("vasoactive_hours").astype(np.float32).to_frame())
    rrt = table("mimiciv_3_1_derived/rrt", ["stay_id", "charttime", "dialysis_active"], stay_ids).merge(cut)
    rrt = rrt[rrt.charttime <= rrt.dischtime]
    feats.append(rrt.groupby("subject_id").dialysis_active.max().rename("rrt_active").astype(np.float32).to_frame())
    print("icu derived", flush=True)

    labs = windowed("mimiciv_3_1_hosp/labevents", ["subject_id", "itemid", "charttime", "valuenum", "flag"], cohort,
                    "charttime", LAB_SPECS, pre_hours=24)
    labs["name"] = labs.itemid.map({k: v[0] for k, v in LAB_SPECS.items()})
    lo = labs.itemid.map({k: v[1] for k, v in LAB_SPECS.items()}); hi = labs.itemid.map({k: v[2] for k, v in LAB_SPECS.items()})
    labs = labs[labs.valuenum.between(lo, hi)]
    feats.append(summarize(labs, "name", "valuenum", "charttime", "lab").astype(np.float32))
    abn = labs.assign(a=labs.flag.eq("abnormal").astype(np.float32)).groupby(["subject_id", "name"]).a.mean().unstack()
    feats.append(abn.add_prefix("lab_abnormal_frac_").astype(np.float32))
    print("labs", len(labs), flush=True)
    del labs

    vit = windowed("mimiciv_3_1_icu/chartevents", ["subject_id", "itemid", "charttime", "valuenum"], cohort, "charttime", VITAL_SPECS)
    vit["name"] = vit.itemid.map({k: v[0] for k, v in VITAL_SPECS.items()})
    for item, (name, lo, hi, convert) in VITAL_SPECS.items():
        rows = vit.itemid == item
        if convert is not None:
            vit.loc[rows, "valuenum"] = vit.loc[rows, "valuenum"].map(convert)
        if lo is not None:
            bad = rows & ~vit.valuenum.between(lo, hi)
            vit.loc[bad, "valuenum"] = np.nan
    vit = vit.dropna(subset=["valuenum"])
    feats.append(summarize(vit, "name", "valuenum", "charttime", "vital").astype(np.float32))
    print("vitals", len(vit), flush=True)
    del vit

    frame = pd.concat([f.reindex(base.index) for f in feats], axis=1)
    count_cols = [c for c in frame.columns if c.startswith(("dx_", "dxprimary_"))] + ["transfer_count", "careunit_distinct"]
    frame[count_cols] = frame[count_cols].fillna(0)
    frame = frame.reset_index().rename(columns={"index": "subject_id"})
    write(frame, "features_concepts.parquet")


def fetch(args):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from google.cloud import bigquery, bigquery_storage
    pool = pd.read_parquet(POOL / "cohort.parquet", columns=["subject_id", "hadm_id"])
    keys = {"hadm": pa.array(pool.hadm_id.to_numpy(), pa.int64()), "subject": pa.array(pool.subject_id.to_numpy(), pa.int64())}
    client = bigquery.Client(project=args.project)
    storage = bigquery_storage.BigQueryReadClient()
    SUPPLEMENT.mkdir(parents=True, exist_ok=True)
    for name, (columns, key) in SUPPLEMENT_TABLES.items():
        job = client.query(f"SELECT {columns} FROM `physionet-data.mimiciv_3_1_hosp.{name}`")
        result = job.result()
        writer, kept = None, 0
        for batch in result.to_arrow_iterable(bqstorage_client=storage):
            frame = pa.Table.from_batches([batch])
            column = "hadm_id" if key == "hadm" else "subject_id"
            mask = pc.is_in(frame[column], value_set=keys[key])
            frame = frame.filter(pc.fill_null(mask, False))
            if writer is None:
                writer = pq.ParquetWriter(SUPPLEMENT / f"{name}.parquet", frame.schema, compression="zstd")
            writer.write_table(frame)
            kept += frame.num_rows
        writer.close()
        log(name, kept)


def vocab_binary(frame, column, prefix, train_ids, min_patients):
    present = frame[["subject_id", column]].dropna().drop_duplicates()
    counts = present[present.subject_id.isin(train_ids)][column].value_counts()
    keep = counts[counts >= min_patients].index
    present = present[present[column].isin(keep)]
    return pd.crosstab(present.subject_id, prefix + present[column].astype(str)).clip(upper=1).astype(np.float32), len(keep)


def supplement(args):
    snap = SUPPLEMENT
    cohort = pd.read_parquet(POOL / "cohort.parquet", columns=["subject_id", "hadm_id", "admittime", "dischtime", "partition"])
    train_ids = set(cohort.loc[cohort.partition.isin(["pool_extra", "cohort_train"]), "subject_id"])
    index = cohort.set_index("subject_id")
    win = cohort[["subject_id", "hadm_id", "dischtime"]]
    feats, info = [], {}

    rx = pd.read_parquet(snap / "prescriptions.parquet").merge(win, on=["subject_id", "hadm_id"])
    rx = rx[rx.starttime.isna() | (rx.starttime <= rx.dischtime)]
    rx["drug_l"] = rx.drug.fillna("").str.lower().str.strip()
    rx["active"] = rx.stoptime.isna() | (rx.stoptime >= rx.dischtime - pd.Timedelta(hours=24))
    route = rx.route.fillna("").str.upper()
    rx["iv"] = route.str.contains("IV").astype(float)
    rx["po"] = route.isin(["PO", "PO/NG", "ORAL"]).astype(float)
    g = rx.groupby("subject_id")
    act = rx[rx.active].groupby("subject_id")
    feats.append(pd.DataFrame({"rx_orders": g.size(), "rx_distinct": g.drug_l.nunique(), "rx_frac_iv": g.iv.mean(),
                               "rx_frac_po": g.po.mean(), "rx_active_discharge": act.size(),
                               "rx_active_distinct": act.drug_l.nunique()}).astype(np.float32))
    for name, pattern in CLASSES.items():
        hit = rx.drug_l.str.contains(pattern, regex=True)
        feats.append(rx[hit].groupby("subject_id").size().rename("rxclass_" + name).astype(np.float32).to_frame())
        feats.append(rx[hit & rx.active].groupby("subject_id").size().rename("rxclass_active_" + name).astype(np.float32).to_frame())
    frame, info["rx_vocab"] = vocab_binary(rx, "drug_l", "rx_", train_ids, 1000)
    feats.append(frame)
    print("prescriptions", len(rx), info["rx_vocab"], flush=True)

    em = pd.read_parquet(snap / "emar.parquet").merge(win, on=["subject_id", "hadm_id"])
    em = em[em.charttime <= em.dischtime]
    txt = em.event_txt.fillna("")
    em["not_given"] = txt.str.contains("Not Given|Held|Not Applied", regex=True).astype(float)
    em["last24"] = (em.charttime >= em.dischtime - pd.Timedelta(hours=24)).astype(float)
    g = em.groupby("subject_id")
    feats.append(pd.DataFrame({"emar_events": g.size(), "emar_distinct": g.medication.nunique(), "emar_frac_not_given": g.not_given.mean(),
                               "emar_last24_events": g.last24.sum()}).astype(np.float32))
    print("emar", len(em), flush=True)

    pr = pd.read_parquet(snap / "procedures_icd.parquet").merge(win, on=["subject_id", "hadm_id"])
    pr = pr[pd.to_datetime(pr.chartdate) <= pr.dischtime.dt.normalize()]
    pr["cat"] = "v" + pr.icd_version.astype(str) + "_" + pr.icd_code.str.strip().str[:3]
    feats.append(pr.groupby("subject_id").size().rename("proc_count").astype(np.float32).to_frame())
    frame, info["proc_vocab"] = vocab_binary(pr, "cat", "proc_", train_ids, 200)
    feats.append(frame)

    drg = pd.read_parquet(snap / "drgcodes.parquet").merge(win[["subject_id", "hadm_id"]], on=["subject_id", "hadm_id"])
    sev = drg.groupby("subject_id")[["drg_severity", "drg_mortality"]].max().add_prefix("apr_").astype(np.float32)
    feats.append(sev)
    drg["code"] = drg.drg_type.astype(str) + "_" + drg.drg_code.astype(str)
    frame, info["drg_vocab"] = vocab_binary(drg, "code", "drg_", train_ids, 100)
    feats.append(frame)

    sv = pd.read_parquet(snap / "services.parquet").merge(win, on=["subject_id", "hadm_id"])
    sv = sv[sv.transfertime <= sv.dischtime].sort_values(["subject_id", "transfertime"])
    g = sv.groupby("subject_id")
    feats.append(pd.DataFrame({"service_changes": g.size(), "service_surgical_ever": g.curr_service.apply(
        lambda s: float(s.str.contains("SURG|ORTHO|NSURG|CSURG|TSURG|VSURG|PSURG", regex=True).any()))}).astype(np.float32))
    last = g.curr_service.last()
    common = last[last.index.isin(train_ids)].value_counts()
    last = last.where(last.isin(common[common >= 200].index), "OTHER")
    feats.append(pd.get_dummies(last, prefix="service_last", dtype=np.float32))

    mb = pd.read_parquet(snap / "microbiologyevents.parquet").merge(win, on=["subject_id", "hadm_id"])
    when = mb.charttime.fillna(pd.to_datetime(mb.chartdate))
    mb = mb[when <= mb.dischtime]
    mb["pos"] = mb.org_name.notna().astype(float)
    mb["blood_pos"] = (mb.spec_type_desc.fillna("").str.contains("BLOOD CULTURE") & mb.org_name.notna()).astype(float)
    mb["resistant"] = mb.interpretation.eq("R").astype(float)
    g = mb.groupby("subject_id")
    feats.append(pd.DataFrame({"micro_records": g.size(), "micro_positive": g.pos.sum(), "micro_blood_positive": g.blood_pos.max(),
                               "micro_resistant": g.resistant.sum()}).astype(np.float32))
    print("procedures/drg/services/micro", info, flush=True)

    om = pd.read_parquet(snap / "omr.parquet").merge(cohort[["subject_id", "admittime"]], on="subject_id")
    om["chartdate"] = pd.to_datetime(om.chartdate)
    om = om[om.chartdate < om.admittime.dt.normalize()]
    g = om.groupby("subject_id")
    feats.append(pd.DataFrame({"omr_records": g.size(), "omr_history_days": (g.admittime.first().dt.normalize() - g.chartdate.min()).dt.days,
                               "omr_last_days_before": (g.admittime.first().dt.normalize() - g.chartdate.max()).dt.days}).astype(np.float32))
    om = om.sort_values(["subject_id", "chartdate"])
    for name, key in (("bmi", r"^BMI"), ("weight", r"^Weight"), ("egfr", r"^eGFR")):
        sel = om[om.result_name.str.contains(key, regex=True, na=False)]
        val = pd.to_numeric(sel.result_value, errors="coerce")
        feats.append(sel.assign(v=val).dropna(subset=["v"]).groupby("subject_id").v.last().rename("omr_last_" + name).astype(np.float32).to_frame())
    bp = om[om.result_name.str.startswith("Blood Pressure", na=False)]
    parts = bp.result_value.str.extract(r"(\d+)\s*/\s*(\d+)").astype(float)
    bp = bp.assign(sbp=parts[0], dbp=parts[1]).dropna(subset=["sbp"])
    feats.append(bp.groupby("subject_id")[["sbp", "dbp"]].last().add_prefix("omr_last_").astype(np.float32))
    print("omr prior-history rows", len(om), flush=True)

    out = pd.concat([f.reindex(index.index) for f in feats], axis=1)
    count_like = [c for c in out.columns if re.match(r"^(rx_|rxclass|proc_|drg_|emar_events|service_|micro_records|micro_positive|omr_records)", c)]
    out[count_like] = out[count_like].fillna(0)
    out = out.reset_index().rename(columns={"index": "subject_id"})
    write(out, "features_supplement.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["export", "select", "labels", "cxr", "flags", "features", "fetch", "supplement"])
    parser.add_argument("--project", default=None)
    args = parser.parse_args()
    POOL.mkdir(parents=True, exist_ok=True)
    {"export": export, "select": select, "labels": labels, "cxr": cxr_images, "flags": flags, "features": features,
     "fetch": fetch, "supplement": supplement}[args.stage](args)
