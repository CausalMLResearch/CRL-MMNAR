"""Rectify in-memory model predictions for the training entry point."""
from __future__ import annotations
import math
from typing import Dict, List, Mapping, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

KAPPAS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
CORRECTION_STRENGTHS = tuple(np.linspace(0.0, 1.0, 11))
RANK_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def logit(values):
    clipped = np.clip(np.asarray(values, dtype=float), 1e-5, 1.0 - 1e-5)
    return np.log(clipped) - np.log1p(-clipped)


def sigmoid(values):
    values = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def safe_logit(values):
    return logit(np.asarray(values, dtype=float))


def composite_strata(targets: np.ndarray, patterns: np.ndarray, n_splits: int) -> np.ndarray:
    combined = targets.astype(int) * 16 + patterns.astype(int)
    _, encoded = np.unique(combined, return_inverse=True)
    if np.bincount(encoded).min() >= n_splits:
        return encoded
    return targets.astype(int)


def metric_dict(targets: np.ndarray, predictions: np.ndarray, weights: np.ndarray | None = None) -> Dict[str, float]:
    return {
        "auc": float(roc_auc_score(targets, predictions, sample_weight=weights)),
        "auprc": float(average_precision_score(targets, predictions, sample_weight=weights)),
        "brier": float(np.average(np.square(targets - predictions), weights=weights)),
        "prevalence": float(np.average(targets, weights=weights)),
    }


def metric_score(result: Mapping[str, float]) -> float:
    normalized_ap = (result["auprc"] - result["prevalence"]) / max(1.0 - result["prevalence"], 1e-12)
    return 0.5 * result["auc"] + 0.5 * normalized_ap


def domain_ratio_weights(
    validation_matrix: np.ndarray,
    test_matrix: np.ndarray,
    validation_patterns: np.ndarray,
    test_patterns: np.ndarray,
    validation_counts: np.ndarray,
    test_counts: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    one_hot = np.eye(16, dtype=float)
    validation_features = np.column_stack([
        validation_matrix,
        validation_matrix.mean(1), validation_matrix.std(1),
        validation_counts,
        one_hot[validation_patterns],
    ])
    test_features = np.column_stack([
        test_matrix,
        test_matrix.mean(1), test_matrix.std(1),
        test_counts,
        one_hot[test_patterns],
    ])
    features = np.vstack([validation_features, test_features])
    domain = np.concatenate([np.zeros(len(validation_features)), np.ones(len(test_features))]).astype(int)
    probabilities = np.zeros(len(features), dtype=float)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, held_out in splitter.split(features, domain):
        scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=seed)
        model.fit(scaler.transform(features[train]), domain[train])
        probabilities[held_out] = model.predict_proba(scaler.transform(features[held_out]))[:, 1]
    probability = np.clip(probabilities[:len(validation_features)], 1e-4, 1.0 - 1e-4)
    odds = probability / (1.0 - probability)
    prior_ratio = len(validation_features) / len(test_features)
    raw = odds * prior_ratio
    weights = np.clip(raw, 0.2, 5.0)
    weights /= np.mean(weights)
    effective_size = float(np.square(weights.sum()) / np.square(weights).sum())
    return weights, {
        "domain_auc": float(roc_auc_score(domain, probabilities)),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "effective_sample_size": effective_size,
    }


def weighted_residual_stat(
    targets: np.ndarray, predictions: np.ndarray, weights: np.ndarray,
) -> Dict[str, float]:
    weights = np.asarray(weights, dtype=float)
    residuals = targets - predictions
    total = float(weights.sum())
    mean = float(np.sum(weights * residuals) / max(total, 1e-12))
    effective_size = float(total * total / max(np.square(weights).sum(), 1e-12))
    variance = float(np.sum(weights * np.square(residuals - mean)) / max(total, 1e-12))
    return {
        "raw": mean,
        "shrunk": mean,
        "se": math.sqrt(max(variance, 1e-8) / max(effective_size, 1.0)),
        "samples": int(len(targets)),
        "effective_samples": effective_size,
        "positives": int(targets.sum()),
        "negatives": int(len(targets) - targets.sum()),
    }


def between_variance(stats: Sequence[Mapping[str, float]], parent_values: Sequence[float]) -> float:
    if len(stats) <= 1:
        return 1e-8
    deviations = np.array([item["raw"] - parent for item, parent in zip(stats, parent_values)], dtype=float)
    noise = np.array([item["se"] ** 2 for item in stats], dtype=float)
    weights = np.array([item["effective_samples"] for item in stats], dtype=float)
    variance = float(np.average(np.square(deviations), weights=weights) - np.average(noise, weights=weights))
    return max(variance, 1e-8)


def fit_residual_hierarchy(
    targets: np.ndarray,
    predictions: np.ndarray,
    patterns: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, object]:
    global_stat = weighted_residual_stat(targets, predictions, weights)
    count_stats: Dict[int, Dict[str, float]] = {}
    for count in sorted({int(value).bit_count() for value in np.unique(patterns)}):
        mask = np.array([int(value).bit_count() == count for value in patterns])
        count_stats[count] = weighted_residual_stat(targets[mask], predictions[mask], weights[mask])
    count_list = list(count_stats.values())
    count_between = between_variance(count_list, [global_stat["raw"]] * len(count_list))
    for stat in count_list:
        shrink = count_between / (count_between + stat["se"] ** 2)
        stat["shrinkage_weight"] = float(shrink)
        stat["shrunk"] = float(shrink * stat["raw"] + (1.0 - shrink) * global_stat["raw"])
        stat["shrunk_se"] = float(shrink * stat["se"])

    pattern_stats: Dict[int, Dict[str, float]] = {}
    for pattern in sorted(np.unique(patterns)):
        mask = patterns == pattern
        pattern_stats[int(pattern)] = weighted_residual_stat(
            targets[mask], predictions[mask], weights[mask],
        )
    pattern_list = list(pattern_stats.values())
    pattern_parents = [count_stats[int(pattern).bit_count()]["shrunk"] for pattern in pattern_stats]
    pattern_between = between_variance(pattern_list, pattern_parents)
    for pattern, stat in pattern_stats.items():
        parent = count_stats[int(pattern).bit_count()]["shrunk"]
        shrink = pattern_between / (pattern_between + stat["se"] ** 2)
        stat["shrinkage_weight"] = float(shrink)
        stat["shrunk"] = float(shrink * stat["raw"] + (1.0 - shrink) * parent)
        stat["shrunk_se"] = float(shrink * stat["se"])
    global_stat["shrunk_se"] = global_stat["se"]
    global_stat["shrinkage_weight"] = 1.0
    return {
        "global": global_stat,
        "counts": count_stats,
        "patterns": pattern_stats,
        "between_variance_count": count_between,
        "between_variance_pattern": pattern_between,
    }


def enough_data(stat: Mapping[str, float], task: str, level: str) -> bool:
    positive_minimum = {"readmission": 15, "icu": 8, "mortality": 10}[task]
    sample_minimum = 100
    if level == "count":
        positive_minimum = max(positive_minimum, 10)
        sample_minimum = 150
    if level == "global":
        positive_minimum = 1
        sample_minimum = 1
    return (
        stat["samples"] >= sample_minimum
        and stat["effective_samples"] >= sample_minimum * 0.7
        and stat["positives"] >= positive_minimum
        and stat["negatives"] >= 30
    )


def passes_gate(stat: Mapping[str, float], task: str, level: str, kappa: float, z_value: float) -> bool:
    lower_magnitude = abs(stat["shrunk"]) - z_value * stat["shrunk_se"]
    return enough_data(stat, task, level) and lower_magnitude > kappa


def hierarchy_tau(
    hierarchy: Mapping[str, object],
    patterns: np.ndarray,
    task: str,
    kappa: float,
    z_value: float,
) -> Tuple[np.ndarray, List[str]]:
    tau = np.zeros(len(patterns), dtype=float)
    source: List[str] = []
    for row, pattern_value in enumerate(patterns):
        pattern = int(pattern_value)
        count = pattern.bit_count()
        pattern_stat = hierarchy["patterns"].get(pattern)
        count_stat = hierarchy["counts"].get(count)
        global_stat = hierarchy["global"]
        if pattern_stat is not None and passes_gate(pattern_stat, task, "pattern", kappa, z_value):
            tau[row] = pattern_stat["shrunk"]
            source.append(f"pattern:{pattern}")
        elif count_stat is not None and passes_gate(count_stat, task, "count", kappa, z_value):
            tau[row] = count_stat["shrunk"]
            source.append(f"count:{count}")
        elif passes_gate(global_stat, task, "global", kappa, z_value):
            tau[row] = global_stat["shrunk"]
            source.append("global")
        else:
            source.append("none")
    return tau, source


def correction_splits(
    targets: np.ndarray, patterns: np.ndarray, folds: int, repeats: int, seed: int,
) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    result = []
    indices = np.arange(len(targets))
    strata = composite_strata(targets, patterns, folds)
    for repeat in range(repeats):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 3571 * repeat)
        for fold, (train, held_out) in enumerate(splitter.split(indices, strata)):
            result.append((repeat, fold, train, held_out))
    return result


def summarize_candidate_deltas(rows: Sequence[Mapping[str, float]], lcb_z: float) -> Dict[str, float]:
    keys = ("auc", "auprc", "brier", "score")
    summary: Dict[str, float] = {}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=float)
        se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        summary[f"mean_delta_{key}"] = float(values.mean())
        summary[f"se_delta_{key}"] = se
        summary[f"lcb_delta_{key}"] = float(values.mean() - lcb_z * se)
    summary["nondegrade_fraction_auc"] = float(np.mean([row["auc"] >= -1e-12 for row in rows]))
    summary["nondegrade_fraction_auprc"] = float(np.mean([row["auprc"] >= -1e-12 for row in rows]))
    return summary


def tune_constant_rectifier(
    targets: np.ndarray,
    predictions: np.ndarray,
    patterns: np.ndarray,
    weights: np.ndarray,
    task: str,
    splits: Sequence[Tuple[int, int, np.ndarray, np.ndarray]],
    uncertainty_z: float,
    lcb_z: float,
) -> Tuple[float, float, List[Dict[str, object]], Dict[Tuple[int, int, float], np.ndarray]]:
    candidate_rows: Dict[Tuple[float, float], List[Dict[str, float]]] = {
        (kappa, strength): [] for kappa in KAPPAS for strength in CORRECTION_STRENGTHS
    }
    cached_tau: Dict[Tuple[int, int, float], np.ndarray] = {}
    for repeat, fold, train, held_out in splits:
        hierarchy = fit_residual_hierarchy(
            targets[train], predictions[train], patterns[train], weights[train],
        )
        base_metrics = metric_dict(targets[held_out], predictions[held_out], weights[held_out])
        base_score = metric_score(base_metrics)
        for kappa in KAPPAS:
            tau, _ = hierarchy_tau(hierarchy, patterns[held_out], task, kappa, uncertainty_z)
            cached_tau[(repeat, fold, kappa)] = tau
            for strength in CORRECTION_STRENGTHS:
                corrected = np.clip(predictions[held_out] + strength * tau, 0.0, 1.0)
                result = metric_dict(targets[held_out], corrected, weights[held_out])
                candidate_rows[(kappa, strength)].append({
                    "auc": result["auc"] - base_metrics["auc"],
                    "auprc": result["auprc"] - base_metrics["auprc"],
                    "brier": result["brier"] - base_metrics["brier"],
                    "score": metric_score(result) - base_score,
                })
    payload: List[Dict[str, object]] = []
    best: Tuple[float, float, float] | None = None
    for (kappa, strength), rows in candidate_rows.items():
        summary = summarize_candidate_deltas(rows, lcb_z)
        admissible = (
            summary["lcb_delta_auc"] >= -1e-8
            and summary["lcb_delta_auprc"] >= -1e-8
            and summary["mean_delta_brier"] <= 1e-5
            and summary["nondegrade_fraction_auc"] >= 0.8
            and summary["nondegrade_fraction_auprc"] >= 0.8
        )
        item = {"kappa": kappa, "lambda": strength, "admissible": admissible, **summary}
        payload.append(item)
        objective = summary["lcb_delta_score"] if admissible else -np.inf
        if best is None or objective > best[0] + 1e-12 or (
            abs(objective - best[0]) <= 1e-12 and strength < best[2]
        ):
            best = (objective, kappa, strength)
    assert best is not None
    if not np.isfinite(best[0]):
        return 0.05, 0.0, payload, cached_tau
    return float(best[1]), float(best[2]), payload, cached_tau


def crossfit_constant_prediction(
    predictions: np.ndarray,
    splits: Sequence[Tuple[int, int, np.ndarray, np.ndarray]],
    cached_tau: Mapping[Tuple[int, int, float], np.ndarray],
    kappa: float,
    strength: float,
) -> np.ndarray:
    repeats = max(item[0] for item in splits) + 1
    repeated = np.zeros((repeats, len(predictions)), dtype=float)
    for repeat, fold, _, held_out in splits:
        tau = cached_tau[(repeat, fold, kappa)]
        repeated[repeat, held_out] = np.clip(predictions[held_out] + strength * tau, 0.0, 1.0)
    return repeated.mean(axis=0)


def ranking_design(
    scores: np.ndarray,
    patterns: np.ndarray,
    pattern_means: Mapping[int, float] | None = None,
) -> Tuple[np.ndarray, Dict[int, float]]:
    if pattern_means is None:
        pattern_means = {
            pattern: float(np.mean(scores[patterns == pattern]))
            for pattern in sorted(np.unique(patterns))
        }
    design = np.zeros((len(scores), 37), dtype=float)
    counts = np.array([int(value).bit_count() for value in patterns], dtype=int)
    design[np.arange(len(scores)), counts] = 1.0
    design[np.arange(len(scores)), 5 + patterns] = 1.0
    centered = np.array([score - pattern_means.get(int(pattern), 0.0) for score, pattern in zip(scores, patterns)])
    design[np.arange(len(scores)), 21 + patterns] = centered
    return design, dict(pattern_means)


def fit_ranking_rectifier(
    targets: np.ndarray,
    predictions: np.ndarray,
    patterns: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> Dict[str, object]:
    scores = safe_logit(predictions)
    design, pattern_means = ranking_design(scores, patterns)
    positives = np.flatnonzero(targets == 1)
    negatives = np.flatnonzero(targets == 0)
    rng = np.random.default_rng(seed)
    pair_count = min(50_000, max(len(positives) * 20, 5_000))
    positive_pair = rng.choice(positives, size=pair_count, replace=True)
    negative_pair = rng.choice(negatives, size=pair_count, replace=True)
    pair_design = design[positive_pair] - design[negative_pair]
    base_margin = scores[positive_pair] - scores[negative_pair]
    pair_weight = weights[positive_pair] * weights[negative_pair]
    pair_weight *= 1.0 + 2.0 * sigmoid(-base_margin)
    pair_weight /= pair_weight.sum()

    positive_weight = max(float(weights[targets == 1].sum()), 1e-12)
    negative_weight = max(float(weights[targets == 0].sum()), 1e-12)
    sample_weight = weights * np.where(targets == 1, 0.5 / positive_weight, 0.5 / negative_weight)
    pattern_counts = np.bincount(patterns, minlength=16)
    penalties = np.zeros(design.shape[1], dtype=float)
    penalties[:5] = 0.05
    penalties[5:21] = 0.15 * np.maximum(1.0, 300.0 / np.maximum(pattern_counts, 1))
    penalties[21:37] = 0.30 * np.maximum(1.0, 300.0 / np.maximum(pattern_counts, 1))

    def objective(parameters: np.ndarray) -> Tuple[float, np.ndarray]:
        corrected_scores = scores + design @ parameters
        probabilities = sigmoid(corrected_scores)
        bce_loss = np.sum(sample_weight * (np.logaddexp(0.0, corrected_scores) - targets * corrected_scores))
        bce_gradient = design.T @ (sample_weight * (probabilities - targets))
        margin = base_margin + pair_design @ parameters
        pair_loss = np.sum(pair_weight * np.logaddexp(0.0, -margin))
        pair_gradient = pair_design.T @ (-pair_weight * sigmoid(-margin))
        penalty = 0.5 * np.sum(penalties * np.square(parameters))
        gradient = 0.25 * bce_gradient + 0.75 * pair_gradient + penalties * parameters
        return float(0.25 * bce_loss + 0.75 * pair_loss + penalty), gradient

    bounds = [(-1.5, 1.5)] * 21 + [(-0.75, 1.5)] * 16
    result = minimize(
        objective, np.zeros(design.shape[1], dtype=float), method="L-BFGS-B", jac=True,
        bounds=bounds, options={"maxiter": 500, "ftol": 1e-11},
    )
    if not result.success:
        raise RuntimeError(f"Ranking rectifier failed: {result.message}")
    return {
        "parameters": np.asarray(result.x, dtype=float),
        "pattern_means": pattern_means,
        "objective": float(result.fun),
        "iterations": int(result.nit),
    }


def ranking_delta(model: Mapping[str, object], predictions: np.ndarray, patterns: np.ndarray) -> np.ndarray:
    scores = safe_logit(predictions)
    design, _ = ranking_design(scores, patterns, model["pattern_means"])
    return design @ np.asarray(model["parameters"], dtype=float)


def tune_ranking_rectifier(
    targets: np.ndarray,
    base_predictions: np.ndarray,
    constant_predictions: np.ndarray,
    patterns: np.ndarray,
    weights: np.ndarray,
    splits: Sequence[Tuple[int, int, np.ndarray, np.ndarray]],
    seed: int,
    lcb_z: float,
) -> Tuple[float, List[Dict[str, object]], np.ndarray]:
    candidate_rows: Dict[float, List[Dict[str, float]]] = {strength: [] for strength in RANK_LAMBDAS}
    repeats = max(item[0] for item in splits) + 1
    repeated_delta = np.zeros((repeats, len(targets)), dtype=float)
    for repeat, fold, train, held_out in splits:
        model = fit_ranking_rectifier(
            targets[train], base_predictions[train], patterns[train], weights[train],
            seed + repeat * 1009 + fold * 101,
        )
        delta = ranking_delta(model, base_predictions[held_out], patterns[held_out])
        repeated_delta[repeat, held_out] = delta
        baseline = metric_dict(targets[held_out], base_predictions[held_out], weights[held_out])
        baseline_score = metric_score(baseline)
        constant_scores = safe_logit(constant_predictions[held_out])
        for strength in RANK_LAMBDAS:
            prediction = sigmoid(constant_scores + strength * delta)
            result = metric_dict(targets[held_out], prediction, weights[held_out])
            candidate_rows[strength].append({
                "auc": result["auc"] - baseline["auc"],
                "auprc": result["auprc"] - baseline["auprc"],
                "brier": result["brier"] - baseline["brier"],
                "score": metric_score(result) - baseline_score,
            })
    payload: List[Dict[str, object]] = []
    best: Tuple[float, float] | None = None
    for strength, rows in candidate_rows.items():
        summary = summarize_candidate_deltas(rows, lcb_z)
        admissible = (
            summary["lcb_delta_auc"] >= -1e-8
            and summary["lcb_delta_auprc"] >= -1e-8
            and summary["mean_delta_brier"] <= 1e-5
            and summary["nondegrade_fraction_auc"] >= 0.8
            and summary["nondegrade_fraction_auprc"] >= 0.8
        )
        payload.append({"lambda": strength, "admissible": admissible, **summary})
        objective = summary["lcb_delta_score"] if admissible else -np.inf
        if best is None or objective > best[0] + 1e-12 or (
            abs(objective - best[0]) <= 1e-12 and strength < best[1]
        ):
            best = (objective, strength)
    assert best is not None
    selected = float(best[1]) if np.isfinite(best[0]) else 0.0
    return selected, payload, repeated_delta.mean(axis=0)


def json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value



def rectify(validation, evaluation, task, seed):
    validation = validation.sort_values("subject_id").reset_index(drop=True)
    evaluation = evaluation.sort_values("subject_id").reset_index(drop=True)
    required = {"subject_id", "missing_pattern", "data_fingerprint", f"{task}_target", f"{task}_prediction"}
    for frame in (validation, evaluation):
        if required - set(frame.columns) or frame.subject_id.duplicated().any():
            raise ValueError(f"Invalid {task} prediction table")
    if set(validation.subject_id) & set(evaluation.subject_id):
        raise ValueError("Validation and evaluation patients overlap")
    if set(validation.data_fingerprint) != set(evaluation.data_fingerprint):
        raise ValueError("Validation and evaluation cohorts differ")
    y = validation[f"{task}_target"].to_numpy(dtype=int)
    p = validation[f"{task}_prediction"].to_numpy(dtype=float)
    pattern = validation.missing_pattern.to_numpy(dtype=int)
    p_test = evaluation[f"{task}_prediction"].to_numpy(dtype=float)
    pattern_test = evaluation.missing_pattern.to_numpy(dtype=int)
    count = np.array([int(value).bit_count() for value in pattern])
    count_test = np.array([int(value).bit_count() for value in pattern_test])
    if task == "mortality":
        weights = np.ones(len(validation), dtype=float)
        shift = None
    else:
        weights, shift = domain_ratio_weights(
            safe_logit(p)[:, None], safe_logit(p_test)[:, None],
            pattern, pattern_test, count, count_test, seed,
        )
    splits = correction_splits(y, pattern, 5, 3, seed + 17)
    kappa, strength, constant_candidates, cached_tau = tune_constant_rectifier(
        y, p, pattern, weights, task, splits, 1.0, 1.0,
    )
    constant_oof = crossfit_constant_prediction(p, splits, cached_tau, kappa, strength)
    rank_strength, rank_candidates, _ = tune_ranking_rectifier(
        y, p, constant_oof, pattern, weights, splits, seed + 31, 1.0,
    )
    hierarchy = fit_residual_hierarchy(y, p, pattern, weights)
    tau, sources = hierarchy_tau(hierarchy, pattern_test, task, kappa, 1.0)
    constant_test = np.clip(p_test + strength * tau, 0.0, 1.0)
    ranking = fit_ranking_rectifier(y, p, pattern, weights, seed + 43)
    delta = ranking_delta(ranking, p_test, pattern_test)
    corrected = sigmoid(safe_logit(constant_test) + rank_strength * delta)
    output = evaluation.copy()
    output[f"{task}_prediction_base"] = p_test
    output[f"{task}_prediction_rectified"] = corrected
    output["rectifier_tau"] = tau
    output["rectifier_source"] = sources
    output["rectifier_ranking_delta"] = delta
    targets = evaluation[f"{task}_target"].to_numpy(dtype=int)
    payload = {
        "task": task,
        "base": metric_dict(targets, p_test),
        "rectified": metric_dict(targets, corrected),
        "selected": {"kappa": kappa, "constant_strength": strength, "ranking_strength": rank_strength},
        "residual_hierarchy": hierarchy,
        "constant_candidates": constant_candidates,
        "ranking_candidates": rank_candidates,
        "covariate_shift": shift,
    }
    return output, payload
