"""Derive 90-day all-cause mortality after a surviving index discharge."""

import pandas as pd


MORTALITY_LABEL = "mortality_after_discharge_90d"
MORTALITY_DEFINITION = {
    "label": MORTALITY_LABEL,
    "prediction_landmark": "index_discharge",
    "horizon_days": 90,
    "death_date_source": "hosp.patients.dod",
    "date_resolution": "day",
    "inclusive_day_range": [0, 90],
    "missing_death_date": "no_recorded_death",
}


def derive_mortality_labels(cohort: pd.DataFrame, patients: pd.DataFrame) -> pd.Series:
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
