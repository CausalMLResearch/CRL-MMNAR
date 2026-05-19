#!/usr/bin/env python
# coding: utf-8

"""Prepare structured, note, and CXR embedding inputs for multimodal training.

Expected local data layout:
  mimiciv_data/      Structured MIMIC-IV CSV files generated from BigQuery.
  mimiciv_note/      Extracted MIMIC-IV-Note CSV files, including discharge.csv and radiology.csv.
  mimiciv_cxr/       MIMIC-CXR image embedding p* folders copied from the downloaded files/ directory.
"""


import os
import sys
import csv
import argparse
import json
import pandas as pd
from google.cloud import bigquery

# --- Project and output directory configuration ---
DEFAULT_GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
OUTPUT_DIR = "mimiciv_data"
ANALYSIS_SUBJECT_IDS_FILE = "analysis_subject_ids.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Run MIMIC-IV multimodal data preprocessing.")
    parser.add_argument(
        "--gcp-project",
        type=str,
        default=DEFAULT_GCP_PROJECT,
        help=(
            "Google Cloud project used for BigQuery billing. If omitted, "
            "google-cloud-bigquery uses the project from application-default credentials."
        )
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download BigQuery CSV files even when the final CSV already exists."
    )
    parser.add_argument(
        "--bq-page-size",
        type=int,
        default=50000,
        help="BigQuery result page size for streaming CSV downloads."
    )
    parser.add_argument(
        "--csv-chunksize",
        type=int,
        default=250000,
        help="Rows per chunk when aggregating large local CSV files."
    )
    parser.add_argument(
        "--cohort-size",
        type=int,
        default=20000,
        help="Number of MIMIC-IV subjects to sample after broad eligibility filtering."
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Deterministic seed used for reproducible cohort sampling."
    )
    return parser.parse_args()

ARGS = parse_args()

# Initialize BigQuery client after CLI parsing so --help does not require credentials.
client = bigquery.Client(project=ARGS.gcp_project) if ARGS.gcp_project else bigquery.Client()

def csv_exists(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0

def cohort_config():
    return {
        "cohort_version": "mimic_iv_v3_1_fixed_random_20k_v1",
        "cohort_size": ARGS.cohort_size,
        "random_seed": ARGS.random_seed,
        "criteria": [
            "anchor_age >= 18",
            "anchor_year_group in 2008-2019",
            "at least one hospital admission",
            "fixed deterministic random sample after eligibility filtering"
        ]
    }

def cohort_config_matches(meta_path):
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f) == cohort_config()
    except Exception:
        return False

def stream_query_to_csv(sql, path, job_config=None, page_size=None):
    """
    Stream a BigQuery result set to CSV without materializing the full table in memory.
    A temporary file is atomically moved into place only after the download completes.
    """
    page_size = page_size or ARGS.bq_page_size
    tmp_path = f"{path}.tmp"
    if os.path.exists(tmp_path):
        print(f"Removing incomplete temporary file {tmp_path}")
        os.remove(tmp_path)

    query_job = client.query(sql, job_config=job_config)
    result = query_job.result(page_size=page_size)
    field_names = [field.name for field in result.schema]
    row_count = 0

    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(field_names)
        for page in result.pages:
            for row in page:
                writer.writerow([row[name] for name in field_names])
                row_count += 1
            if row_count and row_count % (page_size * 10) == 0:
                print(f"  streamed {row_count:,} rows to {path}")

    os.replace(tmp_path, path)
    return row_count

def get_eligible_subject_ids():
    """
    Retrieve a fixed-size reproducible random cohort from the broad MIMIC-IV eligible subject pool.
    """
    sql_query = """
    -- Reproduce the paper-scale cohort by fixed random sampling after broad eligibility filtering.
    WITH eligible_patients AS (
      SELECT
        subject_id,
        anchor_year,
        anchor_year_group
      FROM
        `physionet-data.mimiciv_3_1_hosp.patients`
      WHERE
        anchor_age >= 18
        AND subject_id IS NOT NULL
        AND
        anchor_year_group IN ('2008 - 2010', '2011 - 2013', '2014 - 2016', '2017 - 2019')
    ),
    eligible_subjects AS (
      SELECT DISTINCT
        a.subject_id
      FROM
        `physionet-data.mimiciv_3_1_hosp.admissions` AS a
      JOIN
        eligible_patients AS p
      ON
        a.subject_id = p.subject_id
      WHERE
        a.subject_id IS NOT NULL
        AND a.hadm_id IS NOT NULL
    )
    SELECT
      subject_id
    FROM
      eligible_subjects
    ORDER BY
      FARM_FINGERPRINT(CONCAT(CAST(subject_id AS STRING), '-', CAST(@random_seed AS STRING)))
    LIMIT
      @cohort_size
    """

    subject_ids_path = os.path.join(OUTPUT_DIR, "selected_subject_ids.csv")
    subject_ids_meta_path = os.path.join(OUTPUT_DIR, "selected_subject_ids.meta.json")
    if csv_exists(subject_ids_path) and cohort_config_matches(subject_ids_meta_path) and not ARGS.force_download:
        print(f"Skipping subject_id selection, local file already exists {subject_ids_path}")
        return pd.read_csv(subject_ids_path)['subject_id'].astype(int).tolist()
    if csv_exists(subject_ids_path) and not cohort_config_matches(subject_ids_meta_path):
        print("Existing selected_subject_ids.csv does not match the current paper cohort configuration; regenerating it.")

    print("Executing subject_id selection query...")
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("random_seed", "INT64", ARGS.random_seed),
            bigquery.ScalarQueryParameter("cohort_size", "INT64", ARGS.cohort_size)
        ]
    )
    df_result = client.query(sql_query, job_config=job_config).to_dataframe()
    subject_ids = df_result['subject_id'].astype(int).tolist()
    if len(subject_ids) < ARGS.cohort_size:
        print(f"Warning: eligible cohort only contains {len(subject_ids)} subjects, fewer than requested {ARGS.cohort_size}.")

    # Save the selected subject_ids to CSV for reference
    df_result.to_csv(subject_ids_path, index=False)
    with open(subject_ids_meta_path, "w", encoding="utf-8") as f:
        json.dump(cohort_config(), f, indent=2)
    print(f"Selected subject_ids saved to {subject_ids_path}")

    return subject_ids

def query_table(dataset, table, subject_ids=None, limit=None):
    """
    Fetch all records belonging to subject_ids from a single table and write to CSV.
    If subject_ids=None, fetch the entire table.
    If the corresponding CSV already exists locally, skip fetching.
    """
    fname = f"{dataset.split('.')[-1]}_{table}.csv"
    path = os.path.join(OUTPUT_DIR, fname)

    # Skip download if CSV file already exists
    if csv_exists(path) and not ARGS.force_download:
        print(f"Skipping {table}, local file already exists {path}")
        return path

    # Otherwise, download from BigQuery
    sql = f"SELECT * FROM `{dataset}.{table}`"
    if subject_ids is not None:
        sql += "\nWHERE subject_id IN UNNEST(@subject_ids)"
    if limit:
        sql += f"\nLIMIT {limit}"

    job_config = None
    if subject_ids is not None:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("subject_ids", "INT64", subject_ids)
            ]
        )

    row_count = stream_query_to_csv(sql, path, job_config=job_config)
    print(f"Saved {table} → {path} ({row_count} rows)")
    return path

def main():
    # 1) Get eligible subject_ids using SQL criteria
    subject_ids = get_eligible_subject_ids()
    print(f"Selected {len(subject_ids)} subject_ids based on anchor_year_group criteria")

    # 2) Batch fetch structured tables
    hosp_dataset = "physionet-data.mimiciv_3_1_hosp"
    hosp_tables = [
        "admissions",
        "patients", 
        "diagnoses_icd",
        "procedures_icd",
        "drgcodes",
        "labevents",
        "microbiologyevents",
        "prescriptions",
        "emar",
        "services",
        "transfers"
    ]

    print("\nFetching hospital-related tables...")
    for t in hosp_tables:
        query_table(hosp_dataset, t, subject_ids)

    # 3) Batch fetch ICU-related tables
    icu_dataset = "physionet-data.mimiciv_3_1_icu"
    icu_tables = [
        "icustays",
        "chartevents",
        "datetimeevents",
        "inputevents",
        "outputevents"
    ]

    print("\nFetching ICU-related tables...")
    for t in icu_tables:
        query_table(icu_dataset, t, subject_ids)

    print("\nAll structured data has been fetched successfully.")
    print(f"Total number of patients processed: {len(subject_ids)}")

if __name__ == "__main__":
    main()


# ## Dictionary Table Saving


import os
import sys
import pandas as pd
from google.cloud import bigquery

# Reuse the output directory, CLI arguments, and BigQuery client configured above.
OUTPUT_DIR = "mimiciv_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read subject_id from local file
ids_path = os.path.join(OUTPUT_DIR, "selected_subject_ids.csv")
if not os.path.exists(ids_path):
    print(f"ERROR: Unable to find {ids_path}")
    sys.exit(1)
subject_ids = pd.read_csv(ids_path)['subject_id'].dropna().astype(int).tolist()
print(f"Loaded {len(subject_ids)} subject_id")

# Function specifically for extracting dictionary tables
def extract_dict_table(table_name):
    """
    If mimiciv_3_1_hosp_{table_name}.csv already exists locally, skip download;
    Otherwise, read from BigQuery and save.
    """
    fname = f"mimiciv_3_1_hosp_{table_name}.csv"
    path = os.path.join(OUTPUT_DIR, fname)

    # Skip download if local file already exists
    if csv_exists(path) and not ARGS.force_download:
        print(f"Skipping {table_name}, local file already exists {path}")
        return

    # Otherwise, read from BigQuery and save
    sql = f"SELECT * FROM `physionet-data.mimiciv_3_1_hosp.{table_name}`"
    try:
        row_count = stream_query_to_csv(sql, path)
        print(f"Saved dictionary table {table_name} → {path} ({row_count} rows)")
    except Exception as e:
        print(f"Failed to extract {table_name}: {e}")

# Correct dictionary table name list
dict_tables = [
    "d_icd_diagnoses",
    "d_icd_procedures",
    "d_hcpcs"
]

for t in dict_tables:
    extract_dict_table(t)


# ## Data Consolidation and Data Leakage Prevention


import os
import pandas as pd
import numpy as np

# Data directory
DATA_DIR = 'mimiciv_data'

# 1. Read patient ID list (updated to use the new selection file)
subjects = pd.read_csv(os.path.join(DATA_DIR, 'selected_subject_ids.csv'))
print(f"Loaded {len(subjects)} subjects from SQL-based selection criteria")

# Validate that we have the expected column structure
assert 'subject_id' in subjects.columns, "subject_id column not found in selected subjects file"
assert len(subjects) > 0, "No subjects found in selection file"
selected_subject_ids = set(subjects['subject_id'].astype(int))

# 2. Load admissions, calculate index admission
adm = pd.read_csv(
    os.path.join(DATA_DIR, 'mimiciv_3_1_hosp_admissions.csv'),
    parse_dates=['admittime','dischtime', 'deathtime']
)
adm = adm[adm['subject_id'].isin(selected_subject_ids)].copy()
adm['los'] = (adm['dischtime'] - adm['admittime']).dt.total_seconds() / 86400

# Sort by admission time, take the first record for each subject as index admission
index_adm = adm.sort_values(['subject_id','admittime']).groupby('subject_id', as_index=False).first()

# Restrict the post-discharge prediction cohort to patients who survived the
# index admission. The mortality task below remains death_after_discharge_180d.
initial_subject_count = len(subjects)
index_adm = index_adm[index_adm['hospital_expire_flag'].fillna(0).astype(int) == 0].copy()
subjects = subjects[subjects['subject_id'].astype(int).isin(index_adm['subject_id'].astype(int))].copy()
selected_subject_ids = set(subjects['subject_id'].astype(int))
adm = adm[adm['subject_id'].isin(selected_subject_ids)].copy()
print(f"Excluded {initial_subject_count - len(subjects)} subjects who died during the index admission")

# Build index admission features
idx_feat = index_adm.set_index('subject_id')[[
    'admission_type','insurance','race','los',
    'admission_location','discharge_location'
]].rename(columns=lambda c: f'idx_{c}')

# 3. Derive new labels: 30-day readmission, 90-day ICU during readmission, 180-day death after readmission

# 3.1 30-day readmission after index admission
def calculate_readmission_labels(adm_df, index_adm_df):
    """Calculate 30-day readmission labels based on index admission"""
    # Create a mapping of subject_id to index discharge time
    index_discharge = index_adm_df.set_index('subject_id')['dischtime']

    # Find all admissions after index admission for each patient
    adm_with_index = adm_df.merge(
        index_discharge.reset_index().rename(columns={'dischtime': 'index_dischtime'}),
        on='subject_id',
        how='inner'
    )

    # Filter to admissions after index discharge
    subsequent_adm = adm_with_index[
        adm_with_index['admittime'] > adm_with_index['index_dischtime']
    ].copy()

    # Calculate days from index discharge to subsequent admission
    subsequent_adm['days_from_index_discharge'] = (
        subsequent_adm['admittime'] - subsequent_adm['index_dischtime']
    ).dt.days

    # Identify 30-day readmissions
    readmit_30d = subsequent_adm[subsequent_adm['days_from_index_discharge'] <= 30]

    # Create binary label for each subject
    readmit_subjects = readmit_30d['subject_id'].unique()
    readmit_labels = pd.Series(
        index=index_adm_df['subject_id'],
        data=0,
        name='readmission_30d'
    )
    readmit_labels.loc[readmit_subjects] = 1

    return readmit_labels, subsequent_adm

# Calculate readmission labels
readmit_lbl, subsequent_admissions = calculate_readmission_labels(adm, index_adm)

# 3.2 ICU admission during readmission episodes within 90 days
def calculate_icu_during_readmission(subsequent_adm_df, icu_df, window_days=90):
    """Calculate ICU admission during readmission episodes within the specified window"""
    if len(subsequent_adm_df) == 0:
        return pd.Series(index=subjects['subject_id'], data=0, name='icu_need_after_discharge_90d')

    # Match ICU stays to subsequent admissions, then apply the ICU event-time window
    readmission_cols = ['subject_id', 'hadm_id', 'index_dischtime']
    icu_during_readmission = icu_df.merge(
        subsequent_adm_df[readmission_cols].drop_duplicates(),
        on=['subject_id', 'hadm_id'],
        how='inner'
    )
    if 'intime' in icu_during_readmission.columns:
        icu_during_readmission['intime'] = pd.to_datetime(icu_during_readmission['intime'], errors='coerce')
        icu_days = (
            icu_during_readmission['intime'] - icu_during_readmission['index_dischtime']
        ).dt.total_seconds() / 86400
        icu_during_readmission = icu_during_readmission[
            icu_during_readmission['intime'].notna() & (icu_days >= 0) & (icu_days <= window_days)
        ]
    else:
        readmission_window = subsequent_adm_df[
            subsequent_adm_df['days_from_index_discharge'] <= window_days
        ]
        icu_during_readmission = icu_during_readmission[
            icu_during_readmission['hadm_id'].isin(readmission_window['hadm_id'])
        ]

    # Identify subjects with ICU stays during readmissions
    icu_readmit_subjects = icu_during_readmission['subject_id'].unique()

    # Create binary labels
    icu_readmit_labels = pd.Series(
        index=subjects['subject_id'],
        data=0,
        name='icu_need_after_discharge_90d'
    )
    icu_readmit_labels.loc[icu_readmit_subjects] = 1

    return icu_readmit_labels

# Load ICU stays data
icu = pd.read_csv(os.path.join(DATA_DIR, 'mimiciv_3_1_icu_icustays.csv'), parse_dates=['intime', 'outtime'])
icu_readmit_lbl = calculate_icu_during_readmission(subsequent_admissions, icu)

# 3.3 Death after readmission within 180 days
def calculate_death_after_readmission(subsequent_adm_df, window_days=180):
    """Calculate death during readmission episodes within the specified window"""
    if len(subsequent_adm_df) == 0:
        return pd.Series(index=subjects['subject_id'], data=0, name='death_after_discharge_180d')

    # Find readmissions that resulted in death within the target event-time window
    death_during_readmission = subsequent_adm_df[
        subsequent_adm_df['hospital_expire_flag'] == 1
    ].copy()
    if 'deathtime' in death_during_readmission.columns:
        death_days = (
            death_during_readmission['deathtime'] - death_during_readmission['index_dischtime']
        ).dt.total_seconds() / 86400
        death_during_readmission = death_during_readmission[
            death_during_readmission['deathtime'].notna() & (death_days >= 0) & (death_days <= window_days)
        ]
    else:
        death_during_readmission = death_during_readmission[
            death_during_readmission['days_from_index_discharge'] <= window_days
        ]

    # Identify subjects who died during readmission
    death_readmit_subjects = death_during_readmission['subject_id'].unique()

    # Create binary labels
    death_readmit_labels = pd.Series(
        index=subjects['subject_id'],
        data=0,
        name='death_after_discharge_180d'
    )
    death_readmit_labels.loc[death_readmit_subjects] = 1

    return death_readmit_labels

death_readmit_lbl = calculate_death_after_readmission(subsequent_admissions)

# Combine all labels
labels = pd.concat([
    readmit_lbl.rename('readmission_30d'),
    icu_readmit_lbl,
    death_readmit_lbl
], axis=1).fillna(0).astype(int)
assert 'death_after_discharge_180d' in labels.columns, "Expected post-discharge mortality label is missing"

print(f"Label distribution:")
print(f"30-day readmission: {labels['readmission_30d'].sum()} / {len(labels)} ({labels['readmission_30d'].mean():.3f})")
print(f"ICU need after discharge 90d: {labels['icu_need_after_discharge_90d'].sum()} / {len(labels)} ({labels['icu_need_after_discharge_90d'].mean():.3f})")
print(f"Death after discharge 180d: {labels['death_after_discharge_180d'].sum()} / {len(labels)} ({labels['death_after_discharge_180d'].mean():.3f})")

# 4. Population features (patients)
pat = pd.read_csv(os.path.join(DATA_DIR, 'mimiciv_3_1_hosp_patients.csv')).set_index('subject_id')
pat = pat[pat.index.isin(selected_subject_ids)]
pat_feat = pd.get_dummies(
    pat[['gender','anchor_age','anchor_year_group']],
    columns=['gender','anchor_year_group'], dummy_na=True
)

# 5. Prepare (subject_id, hadm_id) pairs for index admission to use for filtering
idx_pairs = index_adm[['subject_id','hadm_id']]

# 6. Perform inner merge on detail tables by idx_pairs, then aggregate by subject_id
dfs = [idx_feat, labels, pat_feat]

def agg_by_index(df, id_col, agg_specs):
    """
    Aggregate data from detail tables by index admission
    df: Original detail table DataFrame
    id_col: Column name for hadm_id
    agg_specs: dict of output_col: (col, func)
    """
    tmp = df.merge(idx_pairs, on=['subject_id', id_col], how='inner')
    return tmp.groupby('subject_id').agg(**agg_specs)

def combine_sum_partials(partials, columns):
    if not partials:
        empty = pd.DataFrame(columns=columns)
        empty.index.name = 'subject_id'
        return empty
    return pd.concat(partials).groupby(level=0).sum()

def agg_by_index_chunked(csv_path, id_col, agg_specs, usecols, transform=None, chunksize=None):
    """
    Chunked version of agg_by_index for large CSV files.
    Only count/sum aggregations are combined across chunks.
    """
    chunksize = chunksize or ARGS.csv_chunksize
    partials = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        if transform is not None:
            chunk = transform(chunk)
        tmp = chunk.merge(idx_pairs, on=['subject_id', id_col], how='inner')
        if tmp.empty:
            continue
        partials.append(tmp.groupby('subject_id').agg(**agg_specs))
    return combine_sum_partials(partials, list(agg_specs.keys()))

# 6.1 Diagnoses ICD
dx = pd.read_csv(os.path.join(DATA_DIR, 'mimiciv_3_1_hosp_diagnoses_icd.csv'))
dx_feat = agg_by_index(dx, 'hadm_id', {
    'idx_dx_count': ('icd_code','count'),
    'idx_unique_dx': ('icd_code','nunique')
})
dfs.append(dx_feat)

# 6.2 Procedures ICD
pr = pd.read_csv(os.path.join(DATA_DIR, 'mimiciv_3_1_hosp_procedures_icd.csv'))
pr_feat = agg_by_index(pr, 'hadm_id', {
    'idx_proc_count': ('icd_code','count'),
    'idx_unique_proc': ('icd_code','nunique')
})
dfs.append(pr_feat)

# 6.3 DRG codes
drg = pd.read_csv(os.path.join(DATA_DIR, 'mimiciv_3_1_hosp_drgcodes.csv'))
drg['drg_code'] = drg['drg_code'].astype(str)
dr_pairs = idx_pairs.copy()
dr = drg.merge(dr_pairs, on=['subject_id','hadm_id'], how='inner')
dr_feat = dr.groupby('subject_id').agg(
    idx_drg_count=('drg_code','count'),
    idx_severity_mode=('drg_severity', lambda x: x.mode().iat[0] if not x.mode().empty else pd.NA)
)
dfs.append(dr_feat)

# 6.4 Laboratory events
def add_lab_abnormal(chunk):
    chunk['abnormal'] = chunk['flag'].fillna('NORMAL') != 'NORMAL'
    return chunk

lab_feat = agg_by_index_chunked(
    os.path.join(DATA_DIR, 'mimiciv_3_1_hosp_labevents.csv'),
    'hadm_id',
    {
    'idx_lab_count': ('labevent_id','count'),
    'idx_lab_abn': ('abnormal','sum')
    },
    usecols=['subject_id', 'hadm_id', 'labevent_id', 'flag'],
    transform=add_lab_abnormal
)
dfs.append(lab_feat)

# 6.5 Microbiology events
micro = pd.read_csv(os.path.join(DATA_DIR, 'mimiciv_3_1_hosp_microbiologyevents.csv'))
micro['pos'] = micro['interpretation'] == 'Positive'
micro_feat = agg_by_index(micro, 'hadm_id', {
    'idx_micro_count': ('microevent_id','count'),
    'idx_micro_pos': ('pos','sum')
})
dfs.append(micro_feat)

# 6.6 Prescriptions and EMAR
def aggregate_medication_features():
    partials = []
    med_specs = {
        'idx_med_count': ('poe_id','count'),
        'idx_med_abx': ('abx','sum')
    }
    sources = [
        (os.path.join(DATA_DIR, 'mimiciv_3_1_hosp_prescriptions.csv'), 'drug_type'),
        (os.path.join(DATA_DIR, 'mimiciv_3_1_hosp_emar.csv'), 'event_txt')
    ]
    for path, drug_col in sources:
        usecols = ['subject_id', 'hadm_id', 'poe_id', drug_col]
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=ARGS.csv_chunksize, low_memory=False):
            if drug_col != 'drug_type':
                chunk = chunk.rename(columns={drug_col: 'drug_type'})
            chunk['abx'] = chunk['drug_type'].str.contains('ANTIBIOTIC', na=False)
            tmp = chunk.merge(idx_pairs, on=['subject_id', 'hadm_id'], how='inner')
            if tmp.empty:
                continue
            partials.append(tmp.groupby('subject_id').agg(**med_specs))
    return combine_sum_partials(partials, list(med_specs.keys()))

med_feat = aggregate_medication_features()
dfs.append(med_feat)

# 6.7 Services and transfers
serv = pd.read_csv(os.path.join(DATA_DIR, 'mimiciv_3_1_hosp_services.csv'))
serv_feat = agg_by_index(serv, 'hadm_id', {
    'idx_serv_changes': ('transfertime','count'),
    'idx_uniq_serv': ('curr_service','nunique')
})
trans = pd.read_csv(os.path.join(DATA_DIR, 'mimiciv_3_1_hosp_transfers.csv'))
trans_feat = agg_by_index(trans, 'hadm_id', {
    'idx_trans_count': ('transfer_id','count')
})
dfs.extend([serv_feat, trans_feat])

# 6.8 ICU stays during index admission
icu_events = pd.read_csv(os.path.join(DATA_DIR, 'mimiciv_3_1_icu_icustays.csv'))
icu_feat = agg_by_index(icu_events, 'hadm_id', {
    'idx_icu_flag': ('stay_id', lambda s: 1),
    'idx_icu_los': ('los','sum')
})
dfs.append(icu_feat)

# 6.9 Chart events
char_feat = agg_by_index_chunked(
    os.path.join(DATA_DIR, 'mimiciv_3_1_icu_chartevents.csv'),
    'hadm_id',
    {'idx_char_count': ('stay_id','count')},
    usecols=['subject_id', 'hadm_id', 'stay_id']
)
dfs.append(char_feat)

# 6.10 Datetime events
dt_feat = agg_by_index_chunked(
    os.path.join(DATA_DIR, 'mimiciv_3_1_icu_datetimeevents.csv'),
    'hadm_id',
    {'idx_dt_count': ('itemid','count')},
    usecols=['subject_id', 'hadm_id', 'itemid']
)
dfs.append(dt_feat)

# 6.11 Fluid balance
inp_feat = agg_by_index_chunked(
    os.path.join(DATA_DIR, 'mimiciv_3_1_icu_inputevents.csv'),
    'hadm_id',
    {'idx_in_vol': ('totalamount','sum')},
    usecols=['subject_id', 'hadm_id', 'totalamount']
)
out_feat = agg_by_index_chunked(
    os.path.join(DATA_DIR, 'mimiciv_3_1_icu_outputevents.csv'),
    'hadm_id',
    {'idx_out_vol': ('value','sum')},
    usecols=['subject_id', 'hadm_id', 'value']
)
fluids = inp_feat.join(out_feat, how='outer').fillna(0)
fluids['idx_net_fluid'] = fluids['idx_in_vol'] - fluids['idx_out_vol']
dfs.append(fluids)

# 7. Merge all features and labels
features = subjects.set_index('subject_id').join(dfs, how='left').fillna(0)

# Handle any remaining object columns that should be numeric
features = features.infer_objects(copy=False)

# 8. Output
out_file = os.path.join(DATA_DIR, 'patient_features_noleak.csv')
features.reset_index().to_csv(out_file, index=False)

print(f'\nGenerated no-leak feature file: {out_file}')
print(f'Final dataset shape: {features.shape}')
print(f'\nFinal label distribution:')
for label_col in ['readmission_30d', 'icu_need_after_discharge_90d', 'death_after_discharge_180d']:
    if label_col in features.columns:
        count = features[label_col].sum()
        rate = features[label_col].mean()
        print(f'{label_col}: {count} / {len(features)} ({rate:.3f})')


# # Notes Embedding


import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import re
import warnings
from tqdm import tqdm
import gc
from typing import List, Optional, Dict
import json

warnings.filterwarnings('ignore')

class ClinicalNotesEmbedding:
    """
    Advanced clinical notes embedding generator using domain-specific models
    """

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", 
                 max_length: int = 512, batch_size: int = 16):
        """
        Initialize the embedding generator with clinical domain model

        Args:
            model_name: Pre-trained clinical model name
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading clinical model: {model_name}")
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully")

    def preprocess_clinical_text(self, text: str) -> str:
        """
        Advanced preprocessing for clinical notes

        Args:
            text: Raw clinical note text

        Returns:
            Preprocessed clinical text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Remove common artifacts and standardize format
        text = str(text).strip()

        # Remove excessive whitespace and line breaks
        text = re.sub(r'\s+', ' ', text)

        # Remove common template artifacts
        text = re.sub(r'\[Report de-identified.*?\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\*\*[^*]*\*\*', '', text)  # Remove **enclosed** text

        # Standardize medical abbreviations spacing
        text = re.sub(r'([a-zA-Z])\.([a-zA-Z])', r'\1. \2', text)

        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)

        # Truncate very long texts while preserving sentence structure
        if len(text) > self.max_length * 4:  # Rough character limit
            sentences = text.split('. ')
            truncated_sentences = []
            char_count = 0

            for sentence in sentences:
                if char_count + len(sentence) < self.max_length * 3:
                    truncated_sentences.append(sentence)
                    char_count += len(sentence)
                else:
                    break

            text = '. '.join(truncated_sentences)
            if not text.endswith('.'):
                text += '.'

        return text.strip()

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts using advanced pooling strategy

        Args:
            texts: List of preprocessed clinical texts

        Returns:
            Array of embeddings with shape (batch_size, embedding_dim)
        """
        if not texts or all(not text.strip() for text in texts):
            return np.zeros((len(texts), 768))  # ClinicalBERT embedding dimension

        # Tokenize with proper handling of long sequences
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

            # Advanced pooling strategy: weighted average of last 4 layers
            # This captures both local and global semantic information
            hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
            attention_mask = inputs['attention_mask']

            # Apply attention mask to ignore padding tokens
            masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1)

            # Calculate mean pooling with attention weighting
            sum_embeddings = torch.sum(masked_hidden_states, dim=1)
            sum_mask = torch.sum(attention_mask, dim=1, keepdim=True)

            # Avoid division by zero
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask

            # Apply layer normalization for stability
            embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def aggregate_note_embeddings(self, embeddings_list: List[np.ndarray], 
                                  strategy: str = "weighted_mean") -> np.ndarray:
        """
        Aggregate multiple note embeddings for a single patient

        Args:
            embeddings_list: List of embedding arrays for individual notes
            strategy: Aggregation strategy ("weighted_mean", "max_pool", "attention")

        Returns:
            Single aggregated embedding
        """
        if not embeddings_list:
            return np.zeros(768)  # Return zero vector for empty list

        if len(embeddings_list) == 1:
            return embeddings_list[0]

        embeddings_array = np.array(embeddings_list)

        if strategy == "weighted_mean":
            # Weight by embedding magnitude (represents information content)
            weights = np.linalg.norm(embeddings_array, axis=1)
            weights = weights / (np.sum(weights) + 1e-9)
            return np.average(embeddings_array, weights=weights, axis=0)

        elif strategy == "max_pool":
            return np.max(embeddings_array, axis=0)

        else:  # Default to simple mean
            return np.mean(embeddings_array, axis=0)

    def process_notes(self, notes_df: pd.DataFrame, note_type: str) -> Dict[int, np.ndarray]:
        """
        Process all notes of a specific type and generate patient-level embeddings

        Args:
            notes_df: DataFrame containing clinical notes
            note_type: Type of notes being processed ("discharge" or "radiology")

        Returns:
            Dictionary mapping subject_id to aggregated embedding
        """
        print(f"Processing {note_type} notes...")

        if notes_df.empty:
            print(f"No {note_type} notes found")
            return {}

        # Group notes by subject_id
        subject_embeddings = {}

        for subject_id, group in tqdm(notes_df.groupby('subject_id'), 
                                      desc=f"Processing {note_type} notes"):

            # Preprocess all notes for this subject
            texts = [self.preprocess_clinical_text(text) for text in group['text']]

            # Filter out empty texts
            valid_texts = [text for text in texts if text.strip()]

            if not valid_texts:
                # No valid text for this subject
                subject_embeddings[subject_id] = np.zeros(768)
                continue

            # Process in batches
            all_embeddings = []
            for i in range(0, len(valid_texts), self.batch_size):
                batch_texts = valid_texts[i:i + self.batch_size]
                batch_embeddings = self.get_embeddings_batch(batch_texts)
                all_embeddings.extend(batch_embeddings)

            # Aggregate embeddings for this subject
            if all_embeddings:
                aggregated_embedding = self.aggregate_note_embeddings(all_embeddings)
                subject_embeddings[subject_id] = aggregated_embedding
            else:
                subject_embeddings[subject_id] = np.zeros(768)

        print(f"Completed processing {len(subject_embeddings)} subjects for {note_type} notes")
        return subject_embeddings

def load_index_admission_metadata(data_dir: str, selected_subject_ids: set) -> pd.DataFrame:
    """
    Rebuild the index-admission lookup used by structured preprocessing.
    Notes are filtered against this lookup to avoid using future readmission notes.
    """
    admissions = pd.read_csv(
        os.path.join(data_dir, 'mimiciv_3_1_hosp_admissions.csv'),
        parse_dates=['admittime', 'dischtime']
    )
    admissions = admissions[admissions['subject_id'].isin(selected_subject_ids)].copy()
    index_adm = admissions.sort_values(['subject_id', 'admittime']).groupby('subject_id', as_index=False).first()
    return index_adm[['subject_id', 'hadm_id', 'admittime', 'dischtime']]

def load_prediction_subject_ids(data_dir: str) -> List[int]:
    """
    Load the final post-discharge prediction cohort.
    Prefer patient_features_noleak.csv because it already excludes index-admission deaths.
    """
    features_path = os.path.join(data_dir, 'patient_features_noleak.csv')
    if os.path.exists(features_path):
        return pd.read_csv(features_path, usecols=['subject_id'])['subject_id'].astype(int).tolist()
    return pd.read_csv(os.path.join(data_dir, 'selected_subject_ids.csv'))['subject_id'].astype(int).tolist()

def filter_notes_to_index_admission(notes_df: pd.DataFrame, index_adm: pd.DataFrame, note_type: str) -> pd.DataFrame:
    """
    Keep only notes attached to each patient's index admission.
    A one-day grace period after discharge keeps discharge summaries whose chart date
    or storage time lands shortly after discharge while still excluding readmissions.
    """
    if notes_df.empty:
        return notes_df

    required_cols = {'subject_id', 'hadm_id'}
    if not required_cols.issubset(notes_df.columns):
        missing = sorted(required_cols - set(notes_df.columns))
        print(f"Warning: {note_type} notes missing {missing}; skipping notes to avoid future-note leakage")
        return pd.DataFrame(columns=notes_df.columns)

    notes = notes_df.copy()
    index_lookup = index_adm.copy()
    for col in ['subject_id', 'hadm_id']:
        notes[col] = pd.to_numeric(notes[col], errors='coerce').astype('Int64')
        index_lookup[col] = pd.to_numeric(index_lookup[col], errors='coerce').astype('Int64')

    notes = notes.dropna(subset=['subject_id', 'hadm_id'])
    index_lookup = index_lookup.dropna(subset=['subject_id', 'hadm_id'])

    filtered = notes.merge(index_lookup, on=['subject_id', 'hadm_id'], how='inner')
    if 'charttime' in filtered.columns:
        charttime = pd.to_datetime(filtered['charttime'], errors='coerce')
        time_mask = (
            charttime.isna() |
            ((charttime >= filtered['admittime']) & (charttime <= filtered['dischtime'] + pd.Timedelta(days=1)))
        )
        filtered = filtered[time_mask].copy()

    filtered = filtered.drop(columns=['admittime', 'dischtime'], errors='ignore')
    print(f"Filtered {note_type} notes to index admission: {len(notes_df)} -> {len(filtered)}")
    return filtered

def main():
    """
    Main function to generate patient-level text embeddings
    """
    # Configuration
    DATA_DIR = 'mimiciv_data'
    NOTE_DATA_DIR = 'mimiciv_note'
    OUTPUT_FILE = os.path.join(DATA_DIR, 'patient_text_embeddings_nan.csv')

    # Initialize embedding generator with advanced clinical model
    # Alternative high-quality models:
    # - "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    # - "dmis-lab/biobert-base-cased-v1.2"
    # - "allenai/scibert_scivocab_uncased"
    embedder = ClinicalNotesEmbedding(
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        max_length=512,
        batch_size=8  # Adjust based on GPU memory
    )

    # Load selected subject IDs
    print("Loading selected subject IDs...")
    selected_subject_ids = set(load_prediction_subject_ids(DATA_DIR))
    print(f"Found {len(selected_subject_ids)} selected subjects")
    index_adm = load_index_admission_metadata(DATA_DIR, selected_subject_ids)

    # Load and filter discharge notes
    print("Loading discharge notes...")
    try:
        discharge_df = pd.read_csv(os.path.join(NOTE_DATA_DIR, 'discharge.csv'))
        discharge_df = discharge_df[discharge_df['subject_id'].isin(selected_subject_ids)]
        discharge_df = filter_notes_to_index_admission(discharge_df, index_adm, "discharge")
        print(f"Loaded {len(discharge_df)} discharge notes for selected subjects")
    except FileNotFoundError:
        print(f"{os.path.join(NOTE_DATA_DIR, 'discharge.csv')} not found, creating empty DataFrame")
        discharge_df = pd.DataFrame()

    # Load and filter radiology notes
    print("Loading radiology notes...")
    try:
        radiology_df = pd.read_csv(os.path.join(NOTE_DATA_DIR, 'radiology.csv'))
        radiology_df = radiology_df[radiology_df['subject_id'].isin(selected_subject_ids)]
        radiology_df = filter_notes_to_index_admission(radiology_df, index_adm, "radiology")
        print(f"Loaded {len(radiology_df)} radiology notes for selected subjects")
    except FileNotFoundError:
        print(f"{os.path.join(NOTE_DATA_DIR, 'radiology.csv')} not found, creating empty DataFrame")
        radiology_df = pd.DataFrame()

    # Process discharge notes
    discharge_embeddings = embedder.process_notes(discharge_df, "discharge")

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Process radiology notes
    radiology_embeddings = embedder.process_notes(radiology_df, "radiology")

    # Create final DataFrame
    print("Creating final embedding dataset...")
    results = []

    for subject_id in selected_subject_ids:
        # Get discharge embedding
        discharge_emb = discharge_embeddings.get(subject_id)
        if discharge_emb is not None and np.any(discharge_emb != 0):
            discharge_emb_str = json.dumps(discharge_emb.tolist())
        else:
            discharge_emb_str = np.nan

        # Get radiology embedding
        radiology_emb = radiology_embeddings.get(subject_id)
        if radiology_emb is not None and np.any(radiology_emb != 0):
            radiology_emb_str = json.dumps(radiology_emb.tolist())
        else:
            radiology_emb_str = np.nan

        results.append({
            'subject_id': subject_id,
            'discharge_embedding': discharge_emb_str,
            'radiology_embedding': radiology_emb_str
        })

    # Create final DataFrame and save
    final_df = pd.DataFrame(results)
    final_df = final_df.sort_values('subject_id').reset_index(drop=True)

    # Save to CSV
    final_df.to_csv(OUTPUT_FILE, index=False)

    # Print summary statistics
    discharge_valid = final_df['discharge_embedding'].notna().sum()
    radiology_valid = final_df['radiology_embedding'].notna().sum()

    print(f"\nEmbedding generation completed!")
    print(f"Output saved to: {OUTPUT_FILE}")
    print(f"Total subjects: {len(final_df)}")
    print(f"Subjects with discharge embeddings: {discharge_valid} ({discharge_valid/len(final_df)*100:.1f}%)")
    print(f"Subjects with radiology embeddings: {radiology_valid} ({radiology_valid/len(final_df)*100:.1f}%)")
    print(f"Subjects with both embeddings: {((final_df['discharge_embedding'].notna()) & (final_df['radiology_embedding'].notna())).sum()}")

if __name__ == "__main__":
    main()


# # CXR Embedding


import os
import pandas as pd
# TensorFlow only parses TFRecords here; keep it on CPU to avoid CUDA init failures.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import json
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple
import numpy as np

# ========================================
# Configuration Section
# ========================================
DATA_DIR = 'mimiciv_data'
EMB_BASE_DIR = 'mimiciv_cxr'
INTERMEDIATE_CSV = 'cxr_embeddings_by_file.csv'
OUTPUT_CSV = 'mimiciv_data/cxr_embeddings_aggregated.csv'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========================================
# CXR Embedding Extraction Functions
# ========================================

def subject_folder(sid: int) -> str:
    """
    Generate the embedding directory path for a given subject_id

    Args:
        sid: Subject ID

    Returns:
        Directory path containing the subject's embedding files
    """
    prefix = sid // 1_000_000
    return os.path.join(EMB_BASE_DIR, f'p{prefix}', f'p{sid}')

def read_embedding_from_tfrecord(path: str) -> np.ndarray:
    """
    Read embedding vector from a single TFRecord file

    Args:
        path: Path to the TFRecord file

    Returns:
        Numpy array containing the embedding vector, or None if failed
    """
    try:
        ds = tf.data.TFRecordDataset([path])
        feature_spec = {'embedding': tf.io.VarLenFeature(tf.float32)}
        for raw in ds.take(1):
            ex = tf.io.parse_single_example(raw, feature_spec)
            return tf.sparse.to_dense(ex['embedding']).numpy()
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return None

def extract_subject_embeddings(subject_ids: List[int]) -> Tuple[List[Tuple[int, List[List[float]]]], int]:
    """
    Extract all CXR embeddings for given subjects

    Args:
        subject_ids: List of subject IDs to process

    Returns:
        Tuple of (subject_data_list, max_files_count)
    """
    rows = []
    max_files = 0

    print(f"Processing {len(subject_ids)} subjects for CXR embeddings...")

    for i, sid in enumerate(subject_ids):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(subject_ids)} subjects processed")

        folder = subject_folder(sid)
        if not os.path.isdir(folder):
            # Skip subjects without downloaded data
            continue

        # Collect all tfrecord file embeddings for this subject
        embeddings = []
        # Navigate through two-level subdirectory structure: files/pXX/p<sid>/<study>/*.tfrecord
        for study in os.listdir(folder):
            study_dir = os.path.join(folder, study)
            if not os.path.isdir(study_dir):
                continue
            for filename in os.listdir(study_dir):
                if filename.endswith('.tfrecord'):
                    path = os.path.join(study_dir, filename)
                    vec = read_embedding_from_tfrecord(path)
                    if vec is not None:
                        embeddings.append(vec.tolist())

        if len(embeddings) > max_files:
            max_files = len(embeddings)

        rows.append([sid, embeddings])

    print(f"Extraction completed. Found embeddings for {len(rows)} subjects")
    print(f"Maximum number of CXR files per subject: {max_files}")

    return rows, max_files

def save_intermediate_embeddings(rows: List[Tuple[int, List[List[float]]]], max_files: int, output_path: str):
    """
    Save extracted embeddings to intermediate CSV file

    Args:
        rows: List of (subject_id, embeddings_list) tuples
        max_files: Maximum number of files per subject
        output_path: Path to save the CSV file
    """
    # Create DataFrame with dynamic column names
    col_names = ['subject_id'] + [f'file_{i}' for i in range(max_files)]
    data = []

    for sid, embs in rows:
        # Pad with None if subject has fewer tfrecord files than max_files
        padded = embs + [None] * (max_files - len(embs))
        data.append([sid] + padded)

    df = pd.DataFrame(data, columns=col_names)

    # Convert embedding lists to string format for CSV storage
    for c in col_names[1:]:
        df[c] = df[c].apply(lambda x: '' if x is None else '[' + ','.join(f'{v:.6f}' for v in x) + ']')

    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Intermediate embeddings saved to: {output_path}")

# ========================================
# Mean Aggregation Module
# ========================================

class MeanAggregator(nn.Module):
    """
    Deterministic mean-pooling aggregator for multiple CXR embeddings per subject
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through mean-pooling aggregator

        Args:
            x: Input embeddings tensor (batch_size, num_files, emb_dim)

        Returns:
            Aggregated embedding tensor (batch_size, emb_dim)
        """
        return x.mean(dim=1)

# ========================================
# Dataset Class for Aggregation
# ========================================

class CXREmbeddingDataset(Dataset):
    """
    Dataset class for loading CXR embeddings from intermediate CSV
    """

    def __init__(self, csv_path: str):
        """
        Initialize dataset

        Args:
            csv_path: Path to the intermediate CSV file
        """
        self.rows = []
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                sid = row[0]
                # Parse JSON-formatted embedding lists from CSV
                embeds = [json.loads(cell) for cell in row[1:] if cell.strip() != '']
                if embeds:  # Only include subjects with at least one embedding
                    self.rows.append((sid, embeds))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        sid, embeds = self.rows[idx]
        return sid, torch.tensor(embeds, dtype=torch.float32)

# ========================================
# Aggregation Pipeline
# ========================================

def aggregate_embeddings(input_csv: str, output_csv: str) -> None:
    """
    Aggregate multiple CXR embeddings per subject using deterministic mean pooling

    Args:
        input_csv: Path to intermediate CSV with individual embeddings
        output_csv: Path to save aggregated embeddings
    """
    print("Starting embedding aggregation...")

    # Load dataset
    ds = CXREmbeddingDataset(input_csv)
    if len(ds) == 0:
        print("No valid embeddings found in input CSV")
        return

    # Infer embedding dimension from first sample
    _, first = ds[0]
    emb_dim = first.size(1)
    print(f"Detected embedding dimension: {emb_dim}")

    dl = DataLoader(ds, batch_size=1, shuffle=False)

    # Initialize deterministic mean-pooling aggregator
    agg = MeanAggregator().to(DEVICE)
    agg.eval()

    print(f"Using device: {DEVICE}")
    print("Aggregating embeddings with mean pooling...")

    # Process and save aggregated embeddings
    with open(output_csv, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(["subject_id", "agg_embedding"])

        with torch.no_grad():
            for i, (sid, embeds) in enumerate(dl):
                if i % 100 == 0:
                    print(f"Aggregation progress: {i}/{len(dl)} subjects processed")

                embeds = embeds.to(DEVICE)  # Shape: (1, num_files, emb_dim)
                out = agg(embeds)           # Shape: (1, emb_dim)
                vec = out.cpu().numpy().reshape(-1).tolist()
                writer.writerow([sid[0], json.dumps(vec)])

    print(f"Aggregation completed. Output saved to: {output_csv}")

# ========================================
# Main Execution Pipeline
# ========================================

def main():
    """
    Main execution pipeline for CXR embedding extraction and aggregation
    """
    print("=== CXR Embedding Generation Pipeline ===")

    # Load selected subject IDs
    print("Loading selected subject IDs...")
    subject_ids = load_prediction_subject_ids(DATA_DIR)
    print(f"Found {len(subject_ids)} selected subjects")

    # Step 1: Extract individual embeddings from TFRecord files
    print("\n--- Step 1: Extracting individual CXR embeddings ---")
    rows, max_files = extract_subject_embeddings(subject_ids)

    if not rows:
        print("No CXR embeddings found for any subjects. Please check the data directory.")
        return

    # Step 2: Save intermediate results
    print("\n--- Step 2: Saving intermediate embeddings ---")
    save_intermediate_embeddings(rows, max_files, INTERMEDIATE_CSV)

    # Step 3: Aggregate embeddings using deterministic mean pooling
    print("\n--- Step 3: Aggregating embeddings with mean pooling ---")
    aggregate_embeddings(INTERMEDIATE_CSV, OUTPUT_CSV)

    # Final summary
    print("\n=== Pipeline Completed Successfully ===")
    print(f"Processed {len(rows)} subjects with CXR data")
    print(f"Maximum CXR files per subject: {max_files}")
    print(f"Final aggregated embeddings saved to: {OUTPUT_CSV}")

    # Clean up intermediate file (optional)
    # os.remove(INTERMEDIATE_CSV)
    # print(f"Intermediate file {INTERMEDIATE_CSV} removed")

if __name__ == "__main__":
    main()
