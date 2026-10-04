#!/usr/bin/env python3
"""Train readmission, ICU and independent ICU-landmark mortality models."""

from __future__ import annotations

import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset, Sampler

from rectifier import json_safe, rectify


TASKS = ("readmission", "icu", "mortality")
LABELS = {
    "readmission": "readmission_30d",
    "icu": "icu_need_after_discharge_90d",
    "mortality": "in_hospital_mortality",
}
AUXILIARY_LABELS = {"readmission_90d": "readmission_90d"}
MODALITIES = ("struct", "img", "text", "rad")


@dataclass
class TrainConfig:
    task: str
    seed: int = 42
    post_view: str = "complete"
    objective: str = "focal"
    modality_set: str = "sitr"
    batch_size: int = 128
    pretrain_epochs: int = 20
    epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    attention_heads: int = 8
    dropout: float = 0.2
    workers: int = 4
    aux_weight: float = 1.0
    aux_anneal_epochs: int = 10
    missing_weight: float = 0.5
    reconstruction_weight: float = 1.0
    contrastive_weight: float = 0.3
    contrastive_temperature: float = 0.1
    initial_concat_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"Unknown task: {self.task}")
        root = Path(__file__).resolve().parent
        mortality = self.task == "mortality"
        self.data_dir = str(root / "data" / ("mortality" if mortality else "post_discharge"))
        self.output_dir = str(root / "outputs" / "model" / self.task)
        self.patience = 20 if mortality else 100
        self.relationship_weight = 0.0 if mortality else 0.3
        self.conditional_icu_weight = 0.0 if mortality else 0.5
        if not mortality:
            self.class_keep = (0.297, 0.347) if self.task == "readmission" else (0.666, 0.666)
            self.index_seed = 20261004
            self.epoch_size = 16000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_vector(value: object, expected: Optional[int] = None) -> Optional[np.ndarray]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        vector = np.asarray(json.loads(value), dtype=np.float32).reshape(-1)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed embedding JSON: {str(value)[:80]}") from exc
    if not np.isfinite(vector).all():
        raise ValueError("Embedding contains NaN or infinity")
    if expected is not None and vector.size != expected:
        raise ValueError(f"Embedding dimension mismatch: expected {expected}, found {vector.size}")
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def load_embedding_matrix(path: Path, ids: np.ndarray, column: str) -> Tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, usecols=["subject_id", column]).set_index("subject_id")
    if frame.index.duplicated().any():
        raise ValueError(f"{path} contains duplicate subject_id")
    dimension: Optional[int] = None
    for value in frame[column].dropna():
        parsed = parse_vector(value)
        if parsed is not None:
            dimension = int(parsed.size)
            break
    if dimension is None:
        raise ValueError(f"No valid vectors found in {path}:{column}")
    matrix = np.zeros((len(ids), dimension), dtype=np.float32)
    observed = np.zeros(len(ids), dtype=np.float32)
    for row, subject_id in enumerate(ids):
        if int(subject_id) not in frame.index:
            continue
        vector = parse_vector(frame.at[int(subject_id), column], dimension)
        if vector is not None:
            matrix[row] = vector
            observed[row] = 1.0
    return matrix, observed


class DataBundle:
    def __init__(self, data_dir: str, task: Optional[str] = None, post_view: str = "complete"):
        root = Path(data_dir)
        self.post_view = post_view
        self.split_on_missingness = True
        post_metadata = root / f"patient_features_discharge_{post_view}.meta.json"
        if post_metadata.is_file():
            if task == "mortality":
                raise ValueError("Mortality must use the ICU+24h data directory")
            self._load_post_discharge(root, post_view)
            return
        view_specs = {
            "early": (root / "patient_features_early.csv", root / "patient_features_early.meta.json", "early_24h"),
            "discharge": (root / "patient_features_discharge.csv", root / "patient_features_discharge.meta.json", "discharge"),
        }
        metadata_by_view: Dict[str, Dict[str, object]] = {}
        for view, (_, metadata_path, expected_view) in view_specs.items():
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Missing preprocessing metadata: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("structured_pipeline") != "icu_dual_view" or metadata.get("view") != expected_view:
                raise ValueError(f"{metadata_path} is not the expected {view} view")
            metadata_by_view[view] = metadata
        fingerprint = metadata_by_view["early"].get("cohort_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("Structured preprocessing metadata has no cohort fingerprint")
        if metadata_by_view["discharge"].get("cohort_fingerprint") != fingerprint:
            raise ValueError("Early and discharge structured views belong to different cohorts")
        self.fingerprint = fingerprint
        for artifact in (
            "patient_text_embeddings.meta.json",
            "cxr_embeddings_aggregated.meta.json",
            "cxr_embeddings_early_aggregated.meta.json",
        ):
            artifact_path = root / artifact
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Missing embedding metadata: {artifact_path}")
            artifact_metadata = json.loads(artifact_path.read_text(encoding="utf-8"))
            if artifact_metadata.get("cohort_fingerprint") != fingerprint:
                raise ValueError(f"{artifact} belongs to a different cohort")
            if (
                artifact == "patient_text_embeddings.meta.json"
                and (
                    artifact_metadata.get("text_pipeline") != "icu_dual_view"
                    or artifact_metadata.get("post_discharge_cutoff")
                    != "charttime_and_storetime_at_or_before_index_discharge"
                )
            ):
                raise ValueError("Text embeddings are stale; rerun the  text stage")
        required = {"subject_id", "post_discharge_observed", *LABELS.values()}
        frames: Dict[str, pd.DataFrame] = {}
        for view, (data_path, _, _) in view_specs.items():
            structured = pd.read_csv(data_path).sort_values("subject_id").reset_index(drop=True)
            missing = sorted(required - set(structured.columns))
            if missing:
                raise ValueError(f"{data_path} is missing required columns: {missing}")
            if structured.subject_id.duplicated().any():
                raise ValueError(f"{data_path} contains duplicate subjects")
            forbidden_tokens = ["hospital_expire_flag", "deathtime"]
            if view == "early":
                forbidden_tokens.extend(["discharge_location", "hospital_duration", "icu_duration"])
            forbidden = [c for c in structured if any(x in c.lower() for x in forbidden_tokens)]
            if forbidden:
                raise ValueError(f"Forbidden {view}-view features: {forbidden}")
            frames[view] = structured
        invariants = ["subject_id", "post_discharge_observed", *LABELS.values()]
        if not frames["early"][invariants].equals(frames["discharge"][invariants]):
            raise ValueError("Early and discharge structured views have different subjects or labels")

        reference = frames["early"]
        self.ids = reference.subject_id.astype(np.int64).to_numpy()
        self.labels = {task: reference[column].astype(np.float32).to_numpy() for task, column in LABELS.items()}
        # The legacy  cohort predates the explicit 90-day readmission label.
        # Keep a conservative compatibility target; the relationship-aware
        # holdout pipeline requires the native  label below.
        self.auxiliary_labels = {
            "readmission_90d": np.maximum(
                self.labels["readmission"], self.labels["icu"],
            ).astype(np.float32),
        }
        self.post_observed = reference.post_discharge_observed.astype(np.float32).to_numpy()
        for name, values in {**self.labels, "post_observed": self.post_observed}.items():
            unique = set(np.unique(values).tolist())
            if not np.isfinite(values).all() or not unique.issubset({0.0, 1.0}):
                raise ValueError(f"{name} must be a finite binary variable; found {sorted(unique)}")
        if not np.array_equal(self.post_observed, 1.0 - self.labels["mortality"]):
            raise ValueError("post_discharge_observed must equal 1 - in_hospital_mortality")
        exclude = {"subject_id", "post_discharge_observed", *LABELS.values(), *AUXILIARY_LABELS.values()}
        self.feature_names_by_view: Dict[str, List[str]] = {}
        self.struct_raw_by_view: Dict[str, np.ndarray] = {}
        for view, structured in frames.items():
            feature_frame = structured[[c for c in structured.columns if c not in exclude]].copy()
            object_columns = feature_frame.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
            if object_columns:
                raise ValueError(f"{view} structured view contains non-numeric columns: {object_columns[:10]}")
            self.feature_names_by_view[view] = feature_frame.columns.tolist()
            self.struct_raw_by_view[view] = feature_frame.astype(np.float32).to_numpy()
        # Compatibility aliases point to the post-discharge view.  New code
        # must call structured_view_for_task/fit_struct_transform with a task.
        self.feature_names = self.feature_names_by_view["discharge"]
        self.struct_raw = self.struct_raw_by_view["discharge"]

        self.img, has_img = load_embedding_matrix(root / "cxr_embeddings_aggregated.csv", self.ids, "agg_embedding")
        self.img_early, has_img_early = load_embedding_matrix(root / "cxr_embeddings_early_aggregated.csv", self.ids, "agg_embedding")
        self.text, has_text = load_embedding_matrix(root / "patient_text_embeddings.csv", self.ids, "discharge_embedding")
        self.rad, has_rad = load_embedding_matrix(root / "patient_text_embeddings.csv", self.ids, "radiology_embedding")
        self.rad_early, has_rad_early = load_embedding_matrix(root / "patient_text_embeddings.csv", self.ids, "radiology_early_embedding")
        if self.img_early.shape[1] != self.img.shape[1] or self.rad_early.shape[1] != self.rad.shape[1]:
            raise ValueError("Early and hospital-course embedding dimensions must match")
        if (has_img_early > has_img).any() or (has_rad_early > has_rad).any():
            raise ValueError("Landmark image/report availability must be a subset of hospital-course availability")
        self.flags = np.column_stack([np.ones(len(self.ids)), has_img, has_text, has_rad]).astype(np.float32)
        self.mortality_flags = np.column_stack([
            np.ones(len(self.ids)), has_img_early, np.zeros(len(self.ids)), has_rad_early,
        ]).astype(np.float32)
        self.pattern = (self.flags * np.array([8, 4, 2, 1], dtype=np.float32)).sum(1).astype(np.int64)
        self.mortality_pattern = (self.mortality_flags * np.array([8, 4, 2, 1], dtype=np.float32)).sum(1).astype(np.int64)
        print(
            f"Loaded {len(self.ids)} subjects; structured_dim="
            f"early:{self.struct_raw_by_view['early'].shape[1]}/"
            f"discharge:{self.struct_raw_by_view['discharge'].shape[1]}, "
            f"CXR={int(has_img.sum())}/{int(has_img_early.sum())} post/early, "
            f"discharge={int(has_text.sum())}, "
            f"radiology={int(has_rad.sum())}/{int(has_rad_early.sum())} post/early"
        )

    def _load_post_discharge(self, root: Path, post_view: str) -> None:
        # All post-discharge tasks share the same patient split.
        self.split_on_missingness = False
        metadata_path = root / f"patient_features_discharge_{post_view}.meta.json"
        view_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            view_metadata.get("structured_pipeline") != "post_discharge_dual_view"
            or view_metadata.get("view") != post_view
        ):
            raise ValueError(f"{metadata_path} is not the expected post-discharge view")
        fingerprint = view_metadata.get("cohort_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("Post-discharge preprocessing metadata has no fingerprint")
        text_metadata_path = root / "patient_text_embeddings_post.meta.json"
        cxr_metadata_path = root / "cxr_embeddings_aggregated.meta.json"
        for artifact_path in (text_metadata_path, cxr_metadata_path):
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            if artifact.get("cohort_fingerprint") != fingerprint:
                raise ValueError(f"{artifact_path} belongs to another cohort")
        text_metadata = json.loads(text_metadata_path.read_text(encoding="utf-8"))
        if text_metadata.get("text_pipeline") != "post_discharge_dual_view":
            raise ValueError("Post-discharge text embeddings are stale")

        data_path = root / f"patient_features_discharge_{post_view}.csv"
        structured = pd.read_csv(data_path).sort_values("subject_id").reset_index(drop=True)
        required = {
            "subject_id", "post_discharge_observed", *LABELS.values(),
            *AUXILIARY_LABELS.values(),
        }
        missing = sorted(required - set(structured.columns))
        if missing:
            raise ValueError(f"{data_path} is missing required columns: {missing}")
        if structured.subject_id.duplicated().any():
            raise ValueError(f"{data_path} contains duplicate subjects")
        forbidden = [
            column for column in structured
            if any(token in column.lower() for token in ("deathtime", "hospital_expire_flag"))
        ]
        if forbidden:
            raise ValueError(f"Post-discharge view contains forbidden columns: {forbidden}")
        self.fingerprint = fingerprint
        self.ids = structured.subject_id.astype(np.int64).to_numpy()
        self.labels = {
            task_name: structured[column].astype(np.float32).to_numpy()
            for task_name, column in LABELS.items()
        }
        self.auxiliary_labels = {
            name: structured[column].astype(np.float32).to_numpy()
            for name, column in AUXILIARY_LABELS.items()
        }
        if (self.labels["icu"] > self.auxiliary_labels["readmission_90d"]).any():
            raise ValueError("ICU labels must be a subset of 90-day readmission labels")
        self.post_observed = structured.post_discharge_observed.astype(np.float32).to_numpy()
        if not np.all(self.post_observed == 1):
            raise ValueError("Every post-discharge  row must be in the fixed-landmark risk set")
        exclude = {
            "subject_id", "post_discharge_observed", *LABELS.values(),
            *AUXILIARY_LABELS.values(),
        }
        feature_frame = structured[[column for column in structured if column not in exclude]].copy()
        object_columns = feature_frame.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
        if object_columns:
            raise ValueError(f"Post-discharge view contains non-numeric columns: {object_columns[:10]}")
        self.feature_names_by_view = {post_view: feature_frame.columns.tolist()}
        self.struct_raw_by_view = {post_view: feature_frame.astype(np.float32).to_numpy()}
        self.feature_names = self.feature_names_by_view[post_view]
        self.struct_raw = self.struct_raw_by_view[post_view]

        self.img, has_img = load_embedding_matrix(
            root / "cxr_embeddings_aggregated.csv", self.ids, "agg_embedding",
        )
        text_path = root / "patient_text_embeddings_post.csv"
        self.text, has_text = load_embedding_matrix(
            text_path, self.ids, f"discharge_{post_view}_embedding",
        )
        self.rad, has_rad = load_embedding_matrix(
            text_path, self.ids, f"radiology_{post_view}_embedding",
        )
        # FoldDataset retains the four-array contract. These aliases are never
        # used by a post-discharge active task as mortality uses another data directory.
        self.img_early = self.img.copy()
        self.rad_early = self.rad.copy()
        self.flags = np.column_stack([
            np.ones(len(self.ids)), has_img, has_text, has_rad,
        ]).astype(np.float32)
        self.mortality_flags = np.column_stack([
            np.ones(len(self.ids)), np.zeros((len(self.ids), 3), dtype=np.float32),
        ]).astype(np.float32)
        self.pattern = (self.flags * np.array([8, 4, 2, 1], dtype=np.float32)).sum(1).astype(np.int64)
        self.mortality_pattern = np.full(len(self.ids), 8, dtype=np.int64)
        print(
            f"Loaded post-discharge  {post_view} view: {len(self.ids)} subjects; "
            f"structured_dim={self.struct_raw.shape[1]}, CXR={int(has_img.sum())}, "
            f"discharge={int(has_text.sum())}, radiology={int(has_rad.sum())}"
        )

    def structured_view_for_task(self, task: str) -> str:
        if task not in TASKS:
            raise ValueError(f"Unknown task: {task}")
        if task == "mortality":
            return "early"
        return self.post_view if self.post_view in self.struct_raw_by_view else "discharge"

    def raw_structured_for_task(self, task: str) -> Tuple[np.ndarray, List[str]]:
        view = self.structured_view_for_task(task)
        return self.struct_raw_by_view[view], self.feature_names_by_view[view]

    def fit_struct_transform(
        self, train_indices: np.ndarray, task: str = "readmission",
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        raw, feature_names = self.raw_structured_for_task(task)
        train = raw[train_indices].astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            medians = np.nanmedian(train, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(np.isfinite(raw), raw, medians)
        keep = np.ptp(filled[train_indices], axis=0) > 1e-8
        if not keep.any():
            raise ValueError("No non-constant structured features exist in the training fold")
        dropped = [name for name, retain in zip(feature_names, keep) if not retain]
        kept_names = [name for name, retain in zip(feature_names, keep) if retain]
        filled = filled[:, keep]
        medians = medians[keep]
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        scaler.fit(filled[train_indices])
        transformed = scaler.transform(filled).astype(np.float32)
        transformed = np.clip(transformed, -10.0, 10.0)
        state = {
            "view": self.structured_view_for_task(task),
            "task": task,
            "feature_names": kept_names,
            "dropped_train_constant_features": dropped,
            "medians": medians.tolist(),
            "center": scaler.center_.tolist(),
            "scale": scaler.scale_.tolist(),
        }
        return transformed, state


class FoldDataset(Dataset):
    def __init__(self, bundle: DataBundle, struct: np.ndarray, indices: Sequence[int]):
        self.bundle = bundle
        self.struct = struct
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def storage_key(self) -> Tuple[int, int]:
        return self.struct.__array_interface__["data"][0], self.struct.shape[1]

    def row_weights(self) -> np.ndarray:
        targets = np.column_stack([self.bundle.labels[task][self.indices] for task in TASKS])
        labels = targets[:, 0].astype(int)
        patterns = self.bundle.pattern[self.indices]
        total = len(labels)
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        sample_weights = total / (len(unique_labels) * label_counts)
        sample_weights = sample_weights[labels] if len(unique_labels) > 1 else np.ones(total)
        unique_patterns, pattern_counts = np.unique(patterns, return_counts=True)
        pattern_weights = dict(zip(unique_patterns.tolist(), (total / (len(unique_patterns) * pattern_counts)).tolist()))
        pattern_sample_weights = np.array([pattern_weights[pattern] for pattern in patterns.tolist()])
        combined = 0.8 * sample_weights + 0.2 * pattern_sample_weights
        return combined / np.sum(combined) * len(combined)

    def epoch_view(self, positions: Sequence[int], context: FoldDataset, cfg: TrainConfig):
        if cfg.task == "mortality":
            raise ValueError("Epoch views require a post-discharge task")
        weights = self.row_weights()
        reference = context.indices
        labels = self.bundle.labels[cfg.task]
        rng = np.random.default_rng(cfg.index_seed)
        retained = []
        for label in (1, 0):
            candidates = np.sort(reference[labels[reference] == label])
            ordered = rng.permutation(candidates)
            fraction = cfg.class_keep[label]
            size = int(np.floor(len(candidates) * fraction + .5))
            retained.extend(ordered[:size].tolist())
        retained = np.asarray(sorted(retained), dtype=np.int64)
        bounds = len(weights)
        offsets = np.asarray(positions, dtype=np.int64)
        rows = np.sort(np.concatenate([offsets[offsets < bounds], retained]))
        row_weights = np.take(np.asarray(weights, dtype=np.float64), rows, mode="wrap")
        index = EpochIndex(rows, row_weights, cfg.epoch_size, cfg.seed)
        dataset = FoldDataset(self.bundle, self.struct, np.arange(len(self.bundle.ids)))
        return dataset, index

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        return {
            "row_index": torch.tensor(idx, dtype=torch.long),
            "subject_id": torch.tensor(int(self.bundle.ids[idx]), dtype=torch.long),
            "struct": torch.from_numpy(self.struct[idx]),
            "img": torch.from_numpy(self.bundle.img[idx]),
            "img_early": torch.from_numpy(self.bundle.img_early[idx]),
            "text": torch.from_numpy(self.bundle.text[idx]),
            "rad": torch.from_numpy(self.bundle.rad[idx]),
            "rad_early": torch.from_numpy(self.bundle.rad_early[idx]),
            "flags": torch.from_numpy(self.bundle.flags[idx]),
            "mortality_flags": torch.from_numpy(self.bundle.mortality_flags[idx]),
            "readmission": torch.tensor([self.bundle.labels["readmission"][idx]], dtype=torch.float32),
            "readmission_90d": torch.tensor(
                [self.bundle.auxiliary_labels["readmission_90d"][idx]], dtype=torch.float32,
            ),
            "icu": torch.tensor([self.bundle.labels["icu"][idx]], dtype=torch.float32),
            "mortality": torch.tensor([self.bundle.labels["mortality"][idx]], dtype=torch.float32),
            "post_observed": torch.tensor([self.bundle.post_observed[idx]], dtype=torch.float32),
        }


class EpochIndex(Sampler):
    """Iterate an indexed view with a fixed epoch size."""
    def __init__(self, rows, weights, draws, seed):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.draws = int(draws)
        if len(np.unique(self.rows)) != len(self.rows) or not len(self.rows):
            raise ValueError("Row index must be unique and nonempty")
        if self.weights.shape != (len(self.rows),) or not torch.isfinite(self.weights).all() or not (self.weights > 0).all():
            raise ValueError("Invalid row weights")
        if self.draws < len(self.rows):
            raise ValueError("Draw budget must cover every allowed row once")
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self): return self.draws

    def __iter__(self):
        repeats = torch.multinomial(self.weights, self.draws - len(self.rows), replacement=True, generator=self.generator) if self.draws > len(self.rows) else torch.empty(0, dtype=torch.long)
        positions = torch.cat([torch.arange(len(self.rows)), repeats])
        positions = positions[torch.randperm(len(positions), generator=self.generator)].tolist()
        for position in positions:
            yield int(self.rows[position])


class DataViews:
    """Manage dataset views and their batch readers."""

    def __init__(
        self, bundle: DataBundle, struct: np.ndarray, partitions: Sequence[np.ndarray],
        active: np.ndarray, cfg: TrainConfig,
    ):
        self.cfg = cfg
        self._options = {
            "num_workers": cfg.workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": cfg.workers > 0,
        }
        self._reader_cache = {}
        self._readers = []
        for position, rows in enumerate(partitions):
            dataset = FoldDataset(bundle, struct, active if position == 0 else rows)
            self._readers.append(self._make_reader(dataset, position == 0))
        self.supervised = self._readers[0]
        self.index = None
        if cfg.task != "mortality":
            local = self._readers[1].dataset
            context = self._reader_cache[(local.storage_key, False)].dataset
            positions = np.concatenate([self._readers[0].dataset.indices, local.indices])
            dataset, self.index = self._readers[0].dataset.epoch_view(positions, context, cfg)
            self.supervised = self._make_reader(dataset, True, sampler=self.index)

    def _make_reader(self, dataset: FoldDataset, training: bool, sampler=None) -> DataLoader:
        reader = DataLoader(
            dataset, batch_size=self.cfg.batch_size,
            shuffle=training and sampler is None, sampler=sampler, **self._options,
        )
        self._reader_cache[(dataset.storage_key, training)] = reader
        return reader

    @property
    def warmup(self) -> DataLoader:
        return self._readers[0]

    def validation(self, model, device) -> Tuple[Dict[str, float], float]:
        reader = self._readers[1]
        metrics, _ = evaluate_task(model, reader, device, self.cfg, seed_offset=17)
        score_metrics = metrics
        if self.index is not None:
            reader = self._reader_cache[(reader.dataset.storage_key, False)]
            score_metrics, _ = evaluate_task(model, reader, device, self.cfg, seed_offset=29)
        return metrics, checkpoint_score(score_metrics, self.cfg.task)

    def predictions(self, model, device):
        return tuple(
            evaluate_task(model, reader, device, self.cfg, seed_offset=offset)
            for reader, offset in zip(self._readers[1:], (23, 29))
        )


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class MissingnessAwareFusion(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.missing_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        # The paper evaluates recovery of the original 16 modality patterns;
        # use a categorical decoder rather than four independent bit losses.
        self.missing_decoder = nn.Linear(hidden_dim, 16)
        self.gates = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in MODALITIES])
        self.modality_tokens = nn.Parameter(torch.randn(4, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, modalities: Sequence[torch.Tensor], flags: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(modalities) != 4 or flags.shape[1] != 4:
            raise ValueError("Fusion expects exactly four modalities in S/I/T/R order")
        z = self.missing_encoder(flags)
        tokens = []
        for index, value in enumerate(modalities):
            gate = torch.sigmoid(self.gates[index](z))
            token = (value * gate + self.modality_tokens[index]) * flags[:, index:index + 1]
            tokens.append(token)
        stacked = torch.stack(tokens, dim=1)
        padding = ~flags.bool()
        attended, _ = self.attention(stacked, stacked, stacked, key_padding_mask=padding, need_weights=False)
        attended = self.norm1(stacked + attended)
        attended = self.norm2(attended + self.ffn(attended))
        weights = flags.unsqueeze(-1)
        h = (attended * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return h, z, self.missing_decoder(z)


class CRLMMNARFinal(nn.Module):
    def __init__(self, input_dims: Sequence[int], cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleList([ModalityEncoder(dim, cfg.hidden_dim, cfg.dropout) for dim in input_dims])
        self.fusion = MissingnessAwareFusion(cfg.hidden_dim, cfg.attention_heads, cfg.dropout)
        # Preserve the official model's direct [S, I, T, R, z] path.  A
        # learnable residual blend lets the existing attention representation
        # remain the stable starting point while the concatenation path earns a
        # larger or smaller contribution during training.
        self.initial_fusion = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 5, cfg.hidden_dim * 2),
            nn.LayerNorm(cfg.hidden_dim * 2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim),
        )
        self.concat_enabled = cfg.initial_concat_weight > 0.0
        safe_concat_weight = max(cfg.initial_concat_weight, 1e-6)
        initial_logit = math.log(safe_concat_weight / (1.0 - safe_concat_weight))
        self.initial_concat_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        self.reconstructors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2), nn.GELU(),
                nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            )
            for _ in input_dims
        ])
        projection_dim = max(32, cfg.hidden_dim // 2)
        self.original_projections = nn.ModuleList([nn.Linear(cfg.hidden_dim, projection_dim) for _ in input_dims])
        self.reconstructed_projections = nn.ModuleList([nn.Linear(cfg.hidden_dim, projection_dim) for _ in input_dims])
        self.task_adapters = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim),
                nn.GELU(), nn.Dropout(cfg.dropout),
            )
            for task in TASKS
        })
        self.heads = nn.ModuleDict({
            task: nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2), nn.GELU(), nn.Dropout(cfg.dropout / 2), nn.Linear(cfg.hidden_dim // 2, 1))
            for task in TASKS
        })
        # Relationship heads do not merge the task-specific training runs.
        # Each task model learns its own 90-day readmission gate and an ICU
        # severity conditional, which supplies clinically structured auxiliary
        # supervision without sharing parameters with the mortality model.
        self.relationship_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2), nn.GELU(),
                nn.Dropout(cfg.dropout / 2), nn.Linear(cfg.hidden_dim // 2, 1),
            )
            for name in ("readmission_90d", "icu_given_readmission")
        })
        self.conditional_icu_blend_logit = nn.Parameter(torch.tensor(0.0))

    def encode(self, values: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return [encoder(value) for encoder, value in zip(self.encoders, values)]

    def task_logit(self, task: str, h: torch.Tensor) -> torch.Tensor:
        # In the manuscript's simplified outcome model the base head is g(h);
        # direct pattern effects are reserved for the cross-fitted rectifier.
        adapted = self.task_adapters[task](h)
        return self.heads[task](adapted)

    @staticmethod
    def probability_to_logit(probability: torch.Tensor) -> torch.Tensor:
        probability = probability.clamp(1e-6, 1.0 - 1e-6)
        return torch.log(probability) - torch.log1p(-probability)

    def relationship_outputs(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        readmission_90d = self.relationship_heads["readmission_90d"](h)
        icu_given_readmission = self.relationship_heads["icu_given_readmission"](h)
        direct_icu = self.task_logit("icu", h)
        direct_readmission = self.task_logit("readmission", h)
        factorized_probability = torch.sigmoid(readmission_90d) * torch.sigmoid(icu_given_readmission)
        blend = torch.sigmoid(self.conditional_icu_blend_logit)
        icu_probability = (
            (1.0 - blend) * torch.sigmoid(direct_icu)
            + blend * factorized_probability
        )
        return {
            "readmission_90d": readmission_90d,
            "icu_given_readmission": icu_given_readmission,
            "direct_icu": direct_icu,
            "direct_readmission": direct_readmission,
            "factorized_icu": self.probability_to_logit(factorized_probability),
            "blended_icu": self.probability_to_logit(icu_probability),
            "conditional_blend": blend,
        }

    def fuse_encoded(
        self, encoded: Sequence[torch.Tensor], flags: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse encoded modalities through attention and direct concatenation."""
        masked = [value * flags[:, index:index + 1] for index, value in enumerate(encoded)]
        attended, missing_repr, missing_logits = self.fusion(masked, flags)
        if not self.concat_enabled:
            return attended, missing_logits
        concatenated = self.initial_fusion(torch.cat([*masked, missing_repr], dim=1))
        blend = torch.sigmoid(self.initial_concat_logit).to(dtype=attended.dtype)
        return attended + blend * (concatenated - attended), missing_logits

    def forward(
        self,
        values: Sequence[torch.Tensor],
        flags: torch.Tensor,
        mortality_values: Sequence[torch.Tensor],
        mortality_flags: torch.Tensor,
        compute_aux: bool = False,
    ) -> Dict[str, object]:
        encoded = self.encode(values)
        h_post, missing_logits = self.fuse_encoded(encoded, flags)
        outputs: Dict[str, object] = {
            "readmission": self.task_logit("readmission", h_post),
            "icu": self.task_logit("icu", h_post),
            "missing_logits": missing_logits,
        }
        # Discharge summaries occur after the in-hospital mortality outcome and
        # are therefore never visible to the mortality branch.
        mortality_encoded = self.encode(mortality_values)
        h_mort, _ = self.fuse_encoded(mortality_encoded, mortality_flags)
        outputs["mortality"] = self.task_logit("mortality", h_mort)
        if compute_aux:
            outputs["auxiliary"] = self.auxiliary(encoded, flags, missing_logits)
        return outputs

    def forward_active(
        self,
        values: Sequence[torch.Tensor],
        flags: torch.Tensor,
        task: str,
        compute_aux: bool = False,
    ) -> Dict[str, object]:
        """Run one task instance without changing the shared model topology."""
        if task not in TASKS:
            raise ValueError(task)
        encoded = self.encode(values)
        fused, missing_logits = self.fuse_encoded(encoded, flags)
        relationship = self.relationship_outputs(fused)
        if task == "readmission":
            primary_logit = relationship["direct_readmission"]
        elif task == "icu":
            primary_logit = relationship["direct_icu"]
        else:
            primary_logit = self.task_logit(task, fused)
        if task == "icu" and self.cfg.conditional_icu_weight > 0.0:
            primary_logit = relationship["blended_icu"]
        outputs: Dict[str, object] = {
            task: primary_logit,
            "missing_logits": missing_logits,
            "relationship": relationship,
        }
        if compute_aux:
            outputs["auxiliary"] = self.auxiliary(encoded, flags, missing_logits)
        return outputs

    def auxiliary(self, encoded: Sequence[torch.Tensor], flags: torch.Tensor, missing_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        pattern_weights = flags.new_tensor([8, 4, 2, 1])
        pattern_targets = (flags * pattern_weights).sum(1).long()
        missing_loss = F.cross_entropy(missing_logits, pattern_targets)
        eligible = flags.sum(1).ge(2)
        reconstruction_terms: List[torch.Tensor] = []
        contrastive_terms: List[torch.Tensor] = []
        for modality in range(4):
            # Exhaustive leave-one-observed-modality-out is the deterministic
            # minibatch estimator of Algorithm 2 and gives rare CXR patterns
            # enough positives for stable InfoNCE.
            selected = eligible & flags[:, modality].bool()
            if not selected.any():
                continue
            partial_flags = flags.clone()
            partial_flags[:, modality] = 0.0
            partial_h, _ = self.fuse_encoded(encoded, partial_flags)
            reconstructed = self.reconstructors[modality](partial_h)
            reconstruction_terms.append(F.mse_loss(reconstructed[selected], encoded[modality][selected]))
            if selected.sum() >= 2:
                original_z = F.normalize(self.original_projections[modality](encoded[modality][selected]), dim=1)
                reconstructed_z = F.normalize(self.reconstructed_projections[modality](reconstructed[selected]), dim=1)
                logits = original_z @ reconstructed_z.T / self.cfg.contrastive_temperature
                targets = torch.arange(logits.shape[0], device=logits.device)
                contrastive_terms.append((F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2)
        zero = missing_loss * 0.0
        reconstruction = torch.stack(reconstruction_terms).mean() if reconstruction_terms else zero
        contrastive = torch.stack(contrastive_terms).mean() if contrastive_terms else zero
        total = (
            self.cfg.missing_weight * missing_loss
            + self.cfg.reconstruction_weight * reconstruction
            + self.cfg.contrastive_weight * contrastive
        )
        return {"total": total, "missing": missing_loss, "reconstruction": reconstruction, "contrastive": contrastive}


class BinaryFocalLoss(nn.Module):
    def __init__(self, positive_alpha: float, gamma: float = 2.0):
        super().__init__()
        self.positive_alpha = float(positive_alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits, targets = logits.view(-1), targets.view(-1)
        if mask is not None:
            mask = mask.view(-1).bool()
            if not mask.any():
                return logits.sum() * 0.0
            logits, targets = logits[mask], targets[mask]
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probability = torch.sigmoid(logits)
        p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
        alpha_t = self.positive_alpha * targets + (1.0 - self.positive_alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - p_t).pow(self.gamma) * bce).mean()


def active_batch_to_device(
    batch: Dict[str, torch.Tensor], device: torch.device, cfg: TrainConfig,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    struct = batch["struct"].to(device, non_blocking=True)
    if cfg.task == "mortality":
        values = [
            struct,
            batch["img_early"].to(device, non_blocking=True),
            torch.zeros_like(batch["text"].to(device, non_blocking=True)),
            batch["rad_early"].to(device, non_blocking=True),
        ]
        flags = batch["mortality_flags"].to(device, non_blocking=True).clone()
    else:
        values = [
            struct,
            batch["img"].to(device, non_blocking=True),
            batch["text"].to(device, non_blocking=True),
            batch["rad"].to(device, non_blocking=True),
        ]
        flags = batch["flags"].to(device, non_blocking=True).clone()
    allowed = set(cfg.modality_set)
    for index, modality in enumerate("sitr"):
        if modality not in allowed:
            flags[:, index] = 0.0
            values[index] = torch.zeros_like(values[index])
    if not flags[:, 0].bool().all():
        raise ValueError("Structured modality must remain available")
    return values, flags


def evaluate_task(
    model: CRLMMNARFinal,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    seed_offset: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    records: List[Dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            values, flags = active_batch_to_device(batch, device, cfg)
            logits = model.forward_active(values, flags, cfg.task, compute_aux=False)[cfg.task]
            probabilities = torch.sigmoid(logits).view(-1).cpu().numpy()
            for index in range(len(probabilities)):
                record: Dict[str, object] = {
                    "subject_id": int(batch["subject_id"][index]),
                    "row_index": int(batch["row_index"][index]),
                    "post_discharge_observed": int(batch["post_observed"][index, 0]),
                    "has_struct": int(flags[index, 0].cpu()),
                    "has_img": int(flags[index, 1].cpu()),
                    "has_text": int(flags[index, 2].cpu()),
                    "has_rad": int(flags[index, 3].cpu()),
                    "missing_pattern": int((flags[index].cpu() * torch.tensor([8, 4, 2, 1])).sum()),
                    "task": cfg.task,
                    "objective": cfg.objective,
                    "modality_set": cfg.modality_set,
                    f"{cfg.task}_prediction": float(probabilities[index]),
                }
                for task in TASKS:
                    record[f"{task}_target"] = int(batch[task][index, 0])
                record["readmission_90d_target"] = int(batch["readmission_90d"][index, 0])
                records.append(record)
    frame = pd.DataFrame.from_records(records).sort_values("subject_id").reset_index(drop=True)
    valid = np.ones(len(frame), dtype=bool)
    if cfg.task != "mortality":
        valid = frame.post_discharge_observed.eq(1).to_numpy()
    targets = frame.loc[valid, f"{cfg.task}_target"].to_numpy()
    predictions = frame.loc[valid, f"{cfg.task}_prediction"].to_numpy()
    metrics = {
        f"{cfg.task}_auc": safe_metric(roc_auc_score, targets, predictions),
        f"{cfg.task}_auprc": safe_metric(average_precision_score, targets, predictions),
        f"{cfg.task}_brier": float(brier_score_loss(targets, predictions)),
    }
    return metrics, frame


def checkpoint_score(metrics: Dict[str, float], task: str) -> float:
    return 0.7 * metrics[f"{task}_auc"] + 0.3 * metrics[f"{task}_auprc"]


def safe_metric(function, targets: np.ndarray, predictions: np.ndarray) -> float:
    if len(np.unique(targets)) < 2:
        return float("nan")
    return float(function(targets, predictions))


def composite_strata(
    bundle: DataBundle, indices: np.ndarray, minimum: int, task: str,
) -> np.ndarray:
    """Build task-specific strata while retaining risk-set and MMNAR balance."""
    target = bundle.labels[task][indices].astype(int)
    observed = (
        np.ones(len(indices), dtype=int)
        if task == "mortality" else bundle.post_observed[indices].astype(int)
    )
    pattern = bundle.mortality_pattern[indices] if task == "mortality" else bundle.pattern[indices]
    complete_pattern = pattern == (13 if task == "mortality" else 15)
    candidates = []
    if bundle.split_on_missingness:
        candidates.append(np.column_stack([target, observed, complete_pattern.astype(int)]))
    candidates.extend([
        np.column_stack([target, observed]), target.reshape(-1, 1),
    ])
    for components in candidates:
        _, labels = np.unique(components, axis=0, return_inverse=True)
        if np.bincount(labels).min() >= minimum:
            return labels
    raise ValueError("No statistically viable stratification exists")


def relationship_strata(bundle: DataBundle, indices: np.ndarray, minimum: int) -> np.ndarray:
    """Common post-discharge strata so every task and expert shares one holdout."""
    readmission = bundle.labels["readmission"][indices].astype(int)
    readmission_90d = bundle.auxiliary_labels["readmission_90d"][indices].astype(int)
    icu = bundle.labels["icu"][indices].astype(int)
    candidates = [
        np.column_stack([readmission, readmission_90d, icu]),
        np.column_stack([readmission, icu]),
        np.column_stack([readmission_90d, icu]),
        readmission.reshape(-1, 1),
    ]
    for components in candidates:
        _, labels = np.unique(components, axis=0, return_inverse=True)
        if np.bincount(labels).min() >= minimum:
            return labels
    raise ValueError("No statistically viable relationship-aware stratification exists")


def make_splits(bundle: DataBundle, cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_indices = np.arange(len(bundle.ids))
    if cfg.task == "mortality":
        strata = composite_strata(bundle, all_indices, minimum=5, task=cfg.task)
    else:
        strata = relationship_strata(bundle, all_indices, minimum=5)
    train_val, test = train_test_split(
        all_indices, test_size=0.20, random_state=cfg.seed, stratify=strata,
    )
    if cfg.task == "mortality":
        inner_strata = composite_strata(bundle, train_val, minimum=2, task=cfg.task)
    else:
        inner_strata = relationship_strata(bundle, train_val, minimum=2)
    train, validation = train_test_split(
        train_val, test_size=0.15, random_state=cfg.seed, stratify=inner_strata,
    )
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise RuntimeError("Subject split overlap detected")
    for split_name, split in (("train", train), ("validation", validation), ("test", test)):
        tasks = ("mortality",) if cfg.task == "mortality" else ("readmission", "icu")
        for task in tasks:
            if set(np.unique(bundle.labels[task][split]).tolist()) != {0.0, 1.0}:
                raise ValueError(f"{split_name} holdout has fewer than two {task} classes")
    return np.asarray(train), np.asarray(validation), np.asarray(test)


def pretrain(model: CRLMMNARFinal, loader: DataLoader, cfg: TrainConfig, device: torch.device) -> None:
    if cfg.pretrain_epochs <= 0:
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    for epoch in range(1, cfg.pretrain_epochs + 1):
        model.train(); total = 0.0; seen = 0
        for batch in loader:
            values, flags = active_batch_to_device(batch, device, cfg)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                encoded = model.encode(values)
                _, missing_logits = model.fuse_encoded(encoded, flags)
                loss = model.auxiliary(encoded, flags, missing_logits)["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total += float(loss.detach()) * len(flags); seen += len(flags)
        print(f"pretrain epoch {epoch:03d}: loss={total / max(seen, 1):.6f}")


def train_final(model: CRLMMNARFinal, train_loader: DataLoader, views: DataViews, bundle: DataBundle, train_indices: np.ndarray, cfg: TrainConfig, device: torch.device, checkpoint: Path) -> Dict[str, float]:
    task_valid = np.ones(len(train_indices), dtype=bool)
    if cfg.task != "mortality":
        task_valid = bundle.post_observed[train_indices].astype(bool)
    prevalence = float(bundle.labels[cfg.task][train_indices][task_valid].mean())
    focal_loss = BinaryFocalLoss(float(np.clip(1.0 - prevalence, 0.5, 0.95))).to(device)

    def classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return focal_loss(logits, targets)

    def relationship_loss(outputs: Dict[str, object], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        relationship = outputs["relationship"]
        y_readmission_30d = batch["readmission"].to(device).view(-1)
        y_readmission_90d = batch["readmission_90d"].to(device).view(-1)
        y_icu = batch["icu"].to(device).view(-1)
        readmission_90d_loss = F.binary_cross_entropy_with_logits(
            relationship["readmission_90d"].view(-1), y_readmission_90d,
        )
        if cfg.task == "readmission":
            cross_task_icu = F.binary_cross_entropy_with_logits(
                relationship["direct_icu"].view(-1), y_icu,
            )
            return cfg.relationship_weight * (readmission_90d_loss + 0.5 * cross_task_icu)
        if cfg.task == "icu":
            eligible = y_readmission_90d.bool()
            conditional = relationship["icu_given_readmission"].view(-1)
            conditional_loss = conditional.sum() * 0.0
            if eligible.any():
                conditional_loss = F.binary_cross_entropy_with_logits(
                    conditional[eligible], y_icu[eligible],
                )
            cross_task_readmission = F.binary_cross_entropy_with_logits(
                relationship["direct_readmission"].view(-1), y_readmission_30d,
            )
            return (
                cfg.relationship_weight * (readmission_90d_loss + 0.25 * cross_task_readmission)
                + cfg.conditional_icu_weight * conditional_loss
            )
        return outputs[cfg.task].sum() * 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.learning_rate * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_score = -math.inf; no_improvement = 0; best_metrics: Dict[str, float] = {}
    for epoch in range(1, cfg.epochs + 1):
        model.train(); running = 0.0; relationship_running = 0.0; seen = 0
        for batch in train_loader:
            values, flags = active_batch_to_device(batch, device, cfg)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                compute_aux = cfg.aux_weight > 0.0 and cfg.aux_anneal_epochs > 0 and epoch <= cfg.aux_anneal_epochs
                outputs = model.forward_active(values, flags, cfg.task, compute_aux=compute_aux)
                targets = batch[cfg.task].to(device)
                prediction_loss = classification_loss(outputs[cfg.task], targets)
                if cfg.aux_anneal_epochs <= 0:
                    auxiliary_weight = 0.0
                else:
                    auxiliary_weight = cfg.aux_weight * max(0.0, 1.0 - (epoch - 1) / cfg.aux_anneal_epochs)
                loss = prediction_loss
                related_loss = relationship_loss(outputs, batch)
                loss = loss + related_loss
                if compute_aux:
                    loss = loss + auxiliary_weight * outputs["auxiliary"]["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            running += float(loss.detach()) * len(flags); seen += len(flags)
            relationship_running += float(related_loss.detach()) * len(flags)
        scheduler.step()
        metrics, score = views.validation(model, device)
        print(
            f"epoch {epoch:03d}: loss={running / max(seen, 1):.6f}, "
            f"aux_weight={auxiliary_weight:.6f}, lr={scheduler.get_last_lr()[0]:.8f}, "
            f"relationship_loss={relationship_running / max(seen, 1):.6f}, "
            f"metrics={json.dumps(metrics)}"
        )
        if score > best_score + 1e-5:
            best_score, no_improvement, best_metrics = score, 0, metrics
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            payload = {"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch, "validation_metrics": metrics}
            torch.save(payload, checkpoint)
        else:
            no_improvement += 1
            if no_improvement >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    if not checkpoint.exists():
        raise RuntimeError("Training did not produce a checkpoint")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    return best_metrics


@dataclass
class TaskResult:
    metrics: Dict[str, object]
    prediction_views: Tuple[pd.DataFrame, pd.DataFrame]


def run_task(cfg: TrainConfig) -> TaskResult:
    set_seed(cfg.seed)
    bundle = DataBundle(cfg.data_dir, cfg.task, cfg.post_view)
    partitions = make_splits(bundle, cfg)
    task_train_indices = partitions[0]
    if cfg.task != "mortality":
        task_train_indices = task_train_indices[bundle.post_observed[task_train_indices].astype(bool)]
    struct, preprocessing = bundle.fit_struct_transform(task_train_indices, cfg.task)
    run_dir = Path(cfg.output_dir) / f"seed_{cfg.seed}" / "holdout"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "structured_preprocessing.json").write_text(json.dumps(preprocessing), encoding="utf-8")
    views = DataViews(bundle, struct, partitions, task_train_indices, cfg)
    input_dims = [struct.shape[1], bundle.img.shape[1], bundle.text.shape[1], bundle.rad.shape[1]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRLMMNARFinal(input_dims, cfg).to(device)
    pretrain(model, views.warmup, cfg, device)
    print(
        f"Training task={cfg.task}, objective={cfg.objective}, modalities={cfg.modality_set}, "
        f"split=holdout, seed={cfg.seed}, device={device}, "
        f"parameters={sum(p.numel() for p in model.parameters()):,}"
    )
    best_validation = train_final(
        model, views.supervised, views, bundle, task_train_indices,
        cfg, device, run_dir / "best_model.pt",
    )
    results = views.predictions(model, device)
    for (_, frame), name in zip(results, ("validation", "test")):
        frame["seed"] = cfg.seed
        frame["data_fingerprint"] = bundle.fingerprint
        frame["split"] = name
    metric_payload = dict(zip(
        ("best_validation", "reloaded_validation", "test_holdout"),
        (best_validation, *(metrics for metrics, _ in results)),
    ))
    (run_dir / "metrics.json").write_text(json.dumps(metric_payload, indent=2), encoding="utf-8")
    print(json.dumps(results[-1][0], indent=2))
    return TaskResult(metric_payload, tuple(frame for _, frame in results))


def main() -> None:
    summary = {}
    completed = []
    root = Path(__file__).resolve().parent
    for task in TASKS:
        cfg = TrainConfig(task=task)
        result = run_task(cfg)
        completed.append((cfg, result))
        summary[task] = {"metrics": result.metrics["test_holdout"]}
        destination = root / "outputs" / "model" / "metrics.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    rectifier_summary = {}
    for cfg, result in completed:
        _, payload = rectify(
            *result.prediction_views, cfg.task, cfg.seed,
        )
        output = root / "outputs" / "rectifier" / cfg.task
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
        rectifier_summary[cfg.task] = {key: payload[key] for key in ("base", "rectified")}
        (output.parent / "metrics.json").write_text(json.dumps(json_safe(rectifier_summary), indent=2), encoding="utf-8")
        print(json.dumps(json_safe({"task": cfg.task, **rectifier_summary[cfg.task]}), indent=2), flush=True)


if __name__ == "__main__":
    main()
