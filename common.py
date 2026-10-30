import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SNAPSHOT = Path(os.environ.get("MIMIC_SNAPSHOT", ROOT / "data/mimic"))
SUPPLEMENT = Path(os.environ.get("MIMIC_SUPPLEMENT", SNAPSHOT / "supplement"))
NOTE_DIR = Path(os.environ.get("MIMIC_NOTE_DIR", ROOT / "data/mimic-note"))
CXR_DIR = Path(os.environ.get("MIMIC_CXR_DIR", ROOT / "data/mimic-cxr"))
POOL = ROOT / "data/pool"
OUT = ROOT / "outputs"
TASKS = ("readmission", "icu", "mortality")
LABEL = {"readmission": "readmission_30d", "icu": "icu_need_after_discharge_90d", "mortality": "mortality_after_discharge_90d"}
SPLIT_SEED = 42
LGB = {"lgb_a": dict(learning_rate=0.03, num_leaves=63, min_child_samples=100, feature_fraction=0.5,
                     bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0)}
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:7.1f}s]", *args, flush=True)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_columns(df, required, source):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def filter_notes(notes, cohort, kind, note_grace_hours):
    require_columns(notes, {"subject_id", "hadm_id", "charttime", "storetime", "text"}, kind)
    joined = notes.merge(cohort[["subject_id", "hadm_id", "admittime", "dischtime"]],
                         on=["subject_id", "hadm_id"], how="inner", validate="many_to_one")
    chart = pd.to_datetime(joined.charttime, errors="coerce")
    source = kind.split("_")[0]
    cutoff = joined.dischtime + pd.to_timedelta(note_grace_hours, unit="h")
    start = joined.admittime if source == "discharge" else joined.admittime - pd.Timedelta(hours=6)
    valid = chart.notna() & chart.between(start, cutoff, inclusive="both")
    return joined.loc[valid, notes.columns].copy()


def notes_by_patient(source, window, wanted=None, sep="\n", chunksize=50000):
    parts = []
    for notes in pd.read_csv(NOTE_DIR / (source + ".csv"), chunksize=chunksize,
                             usecols=["subject_id", "hadm_id", "charttime", "storetime", "text"], low_memory=False):
        if wanted is not None:
            notes = notes[notes.subject_id.isin(wanted)]
        kept = filter_notes(notes, window, source, 24)
        if len(kept):
            parts.append(kept[["subject_id", "charttime", "text"]])
    frame = pd.concat(parts).sort_values(["subject_id", "charttime"])
    return frame.groupby("subject_id").text.apply(lambda s: sep.join(map(str, s)))


class Cohort:
    def __init__(self):
        pool = pd.read_parquet(POOL / "cohort.parquet")
        with np.load(POOL / "cohort_split.npz") as z:
            self.ids = z["subject_id"]
            self.splits = {k: z[k] for k in ("train", "validation", "test", "selection", "calibration")}
        with np.load(POOL / "cohort_flags.npz") as z:
            if not np.array_equal(z["subject_id"], self.ids):
                raise ValueError("Observation flags have a different patient order")
            self.flags = z["flags"]
        self.pattern = (self.flags * np.array([8, 4, 2, 1], dtype=np.float32)).sum(1).astype(np.int64)
        by_id = pool.set_index("subject_id").loc[self.ids]
        self.labels = {t: by_id[LABEL[t]].to_numpy().astype(np.float32) for t in TASKS}
        self.split_sha256 = digest(POOL / "cohort_split.npz")

    def target(self, task, rows, *, final_evaluation=False):
        rows = np.asarray(rows, dtype=int)
        if np.intersect1d(rows, self.splits["test"]).size and not final_evaluation:
            raise ValueError("Test outcomes are unavailable in development")
        return self.labels[task][rows].astype(int)
