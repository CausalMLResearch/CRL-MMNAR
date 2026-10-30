import argparse
import copy
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import LABEL, LGB, OUT, POOL, SPLIT_SEED, TASKS, Cohort, digest, log, notes_by_patient

HEAD = {"readmission": 0, "icu": 1, "mortality": 2}
KAPPAS = (0.03, 0.05)
PATTERN_NAMES = {8: "S", 9: "S+R", 10: "S+T", 11: "S+T+R", 12: "S+I", 13: "S+I+R", 14: "S+I+T", 15: "S+I+T+R"}
CODE = ["common.py", "data.py", "experts.py", "model.py"]
B = 2000
EXPERTS = ["S", "FT", "R", "CXR"]
MODS = ("S", "I", "T", "R")
CV_SEEDS, FIT_SEEDS = (0, 1, 2), (100, 101, 102)
LAM_REP = 0.1


def scores(y, p, boot=300):
    out = {"n": int(len(y)), "events": int(y.sum()), "auc": float(roc_auc_score(y, p)),
           "ap": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p))}
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(y), len(y)) for _ in range(boot)]
    draws = [i for i in draws if 0 < y[i].sum() < len(i)]
    if draws:
        out["auc_ci"] = np.percentile([roc_auc_score(y[i], p[i]) for i in draws], [2.5, 97.5]).tolist()
        out["ap_ci"] = np.percentile([average_precision_score(y[i], p[i]) for i in draws], [2.5, 97.5]).tolist()
    return out


def metrics(y, p):
    return {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "n": int(len(y)), "events": int(y.sum()),
            "observed": float(y.mean()), "predicted": float(p.mean())}


def pattern_tau(y, p, pattern):
    return {int(g): float(np.mean(y[pattern == g] - p[pattern == g])) for g in np.unique(pattern)}


def apply_tau(p, pattern, tau, kappa):
    adj = np.array([tau.get(int(g), 0.0) if abs(tau.get(int(g), 0.0)) > kappa else 0.0 for g in pattern])
    return np.clip(p + adj, 1e-6, 1 - 1e-6)


def ppi_table(y, p, pattern):
    out = {}
    for g in range(8, 16):
        m = pattern == g
        if m.sum() == 0:
            out[PATTERN_NAMES[g]] = {"n": 0}
            continue
        r = y[m] - p[m]
        se = r.std(ddof=1) / np.sqrt(m.sum()) if m.sum() > 1 else float("nan")
        out[PATTERN_NAMES[g]] = {"n": int(m.sum()), "events": int(y[m].sum()), "observed": float(y[m].mean()),
                                 "mean_risk": float(p[m].mean()), "tau_hat_pp": float(100 * r.mean()),
                                 "ci95_pp": [float(100 * (r.mean() - 1.96 * se)), float(100 * (r.mean() + 1.96 * se))]}
    return out


def pattern_adjustment(y, stages, pattern):
    out = {}
    for g in range(8, 16):
        m = pattern == g
        row = {"n": int(m.sum()), "events": int(y[m].sum()) if m.any() else 0}
        if m.any():
            row["observed"] = float(y[m].mean())
            for name, p in stages.items():
                row["mean_risk_" + name] = float(p[m].mean())
                row["brier_" + name] = float(brier_score_loss(y[m], p[m]))
            first, last = list(stages)[0], list(stages)[-1]
            row["risk_change_pp"] = 100 * (row["mean_risk_" + last] - row["mean_risk_" + first])
            row["brier_change"] = row["brier_" + last] - row["brier_" + first]
        out[PATTERN_NAMES[g]] = row
    return out


def lgb_seed_models(X, y, seeds=(0, 1, 2)):
    import lightgbm as lgb
    models = []
    for seed in seeds:
        stop = np.random.default_rng(seed).random(len(y)) < 0.1
        train = lgb.Dataset(X[~stop], y[~stop], free_raw_data=True)
        valid = lgb.Dataset(X[stop], y[stop], reference=train)
        models.append(lgb.train({"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": seed,
                                 "num_threads": 16, **LGB["lgb_a"]}, train, num_boost_round=6000, valid_sets=[valid],
                                callbacks=[lgb.early_stopping(150, verbose=False)]))
    return models


def build_h(dims, d=16, heads=4):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.ModuleList([nn.Linear(k, d) for k in dims])
            self.zmlp = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 16))
            self.zdec = nn.Linear(16, 8)
            self.gate = nn.ModuleList([nn.Linear(16, d) for _ in dims])
            self.bias = nn.Linear(16, d)
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.heads = nn.Linear(d, 3)
            self.dec = nn.ModuleList([nn.Linear(d, k) for k in dims])

        def fuse(self, e, delta, z):
            tok = torch.stack([delta[:, m, None] * torch.sigmoid(self.gate[m](z)) * e[m] for m in range(len(e))], 1)
            a, _ = self.attn(tok, tok, tok, key_padding_mask=delta < 0.5)
            tok = (tok + a) * delta[:, :, None]
            return tok.sum(1) + self.bias(z)

    return Net()


def fit_h(X, delta, Y, fit, pred, seed, lam_miss=0.1, lam_rep=LAM_REP):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(fit)
    n_stop = int(0.15 * len(perm))
    stop, tr = perm[:n_stop], perm[n_stop:]
    Xs, norm = [], []
    for j, x in enumerate(X):
        o = tr[delta[tr, j] > 0.5]
        mu, sd = x[o].mean(0), x[o].std(0) + 1e-6
        norm.append((mu, sd))
        Xs.append(torch.from_numpy(np.where(delta[:, j:j + 1] > 0.5, (x - mu) / sd, 0.0).astype(np.float32)))
    D, Yt = torch.from_numpy(delta), torch.from_numpy(Y)
    pat = torch.from_numpy(((delta * np.array([8, 4, 2, 1])).sum(1) - 8).astype(np.int64))
    net = build_h([x.shape[1] for x in Xs])
    task_w = torch.tensor(np.ones(3), dtype=torch.float32)
    params = list(net.parameters())
    if lam_rep > 0:
        d = net.heads.in_features
        predictor = nn.ModuleList([nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, d)) for _ in Xs])
        target = copy.deepcopy(net.enc).requires_grad_(False)
        params += list(predictor.parameters())
    opt = torch.optim.Adam([{"params": params, "weight_decay": 1e-4}], lr=1e-3)
    gen = torch.Generator().manual_seed(seed + 7)

    def forward(idx):
        e = [f(x[idx]) for f, x in zip(net.enc, Xs)]
        z = net.zmlp(D[idx])
        return z, net.fuse(e, D[idx], z)

    def rep_loss(idx, e, dl):
        with torch.no_grad():
            tgt = [f(x[idx]) for f, x in zip(target, Xs)]
        l = 0.0
        for m in range(len(e)):
            sel = (dl[:, m] > 0.5) & (dl.sum(1) > 1.5) & (torch.rand(len(idx), generator=gen) < 0.5)
            if sel.sum() < 2:
                continue
            dm = dl.clone()
            dm[sel, m] = 0.0
            rec = predictor[m](net.fuse(e, dm, net.zmlp(dm))[sel])
            l = l + (2 - 2 * (F.normalize(rec, dim=1) * F.normalize(tgt[m][sel], dim=1)).sum(1)).mean()
        return l

    def loss(idx):
        dl = D[idx]
        e = [f(x[idx]) for f, x in zip(net.enc, Xs)]
        z = net.zmlp(dl)
        h = net.fuse(e, dl, z)
        l = (F.binary_cross_entropy_with_logits(net.heads(h), Yt[idx], reduction="none").mean(0) * task_w).mean() \
            + lam_miss * F.cross_entropy(net.zdec(z), pat[idx])
        return l + lam_rep * rep_loss(idx, e, dl) if lam_rep > 0 else l

    def step(l):
        opt.zero_grad(); l.backward(); opt.step()
        if lam_rep > 0:
            with torch.no_grad():
                for pt, p in zip(target.parameters(), net.enc.parameters()):
                    pt.mul_(0.99).add_(p.detach(), alpha=0.01)

    @torch.no_grad()
    def logits(idx):
        return net.heads(forward(torch.from_numpy(idx))[1]).numpy()

    best_t, bad_t, out_t = np.full(3, np.inf), np.zeros(3, int), np.zeros((len(pred), 3), np.float32)
    states = [None] * 3
    ep = -1
    for ep in range(300):
        order = rng.permutation(tr)
        for i in range(0, len(order), 256):
            step(loss(torch.from_numpy(order[i:i + 256])))
        p = 1 / (1 + np.exp(-logits(stop)))
        v = -np.mean(Y[stop] * np.log(np.clip(p, 1e-6, 1)) + (1 - Y[stop]) * np.log(np.clip(1 - p, 1e-6, 1)), axis=0)
        better = v < best_t - 1e-5
        if better.any():
            cur = logits(pred)
            out_t[:, better] = cur[:, better]
            for t in np.where(better)[0]:
                states[t] = {k: w.clone() for k, w in net.state_dict().items()}
        best_t = np.where(better, v, best_t)
        bad_t = np.where(better, 0, bad_t + 1)
        if (bad_t >= 20).all():
            break
    return out_t, ep + 1, states, norm


def fit(args):
    import torch
    torch.set_num_threads(4)
    experts = EXPERTS
    out = OUT / "final" / args.run_name
    if (out / "frozen_bundle.json").exists():
        raise FileExistsError("Frozen bundle already exists; choose a new run name")
    out.mkdir(parents=True, exist_ok=True)

    data = Cohort()
    b = data
    train_rows, meta_rows, test_rows = data.splits["train"], data.splits["validation"], data.splits["test"]
    sel_rows, cal_rows = data.splits["selection"], data.splits["calibration"]
    pool = pd.read_parquet(POOL / "cohort.parquet")
    extra = pool[pool.partition.eq("pool_extra")].sort_values("subject_id")
    ids = np.concatenate([b.ids[train_rows], extra.subject_id.to_numpy(), b.ids[meta_rows], b.ids[test_rows]])
    n_tr, n_ex, n_meta = len(train_rows), len(extra), len(meta_rows)
    universe = np.arange(n_tr + n_ex)
    meta_idx = np.arange(n_tr + n_ex, n_tr + n_ex + n_meta)
    test_idx = np.arange(n_tr + n_ex + n_meta, len(ids))
    targets = np.concatenate([meta_idx, test_idx])
    y_universe = {t: np.concatenate([data.target(t, train_rows), extra[LABEL[t]].to_numpy()]).astype(int) for t in TASKS}
    y_meta = {t: data.target(t, meta_rows).astype(int) for t in TASKS}
    y_oof = {t: data.target(t, train_rows).astype(int) for t in TASKS}
    S = np.hstack([pd.read_parquet(POOL / "features_concepts.parquet").set_index("subject_id").loc[ids].to_numpy(np.float32),
                   pd.read_parquet(POOL / "features_supplement.parquet").set_index("subject_id").loc[ids].to_numpy(np.float32)])
    log("universe", len(universe), "meta", n_meta, "test", len(test_idx), "S", S.shape, "experts", experts)

    window = pool[["subject_id", "hadm_id", "admittime", "dischtime"]]
    rows_all = np.concatenate([train_rows, meta_rows, test_rows])
    cohort_positions = np.concatenate([np.arange(n_tr), meta_idx, test_idx])
    has, vectorizers = {}, {}
    for name, source in (("text", "discharge"), ("rad", "radiology")):
        docs = notes_by_patient(source, window, set(ids.tolist())).reindex(ids).fillna("").to_numpy()
        has[name] = docs != ""
        if not np.array_equal(has[name][cohort_positions], b.flags[rows_all, 2 if name == "text" else 3].astype(bool)):
            raise ValueError(name + " availability differs from the observation flags")
        if name == "rad":
            vectorizers[name] = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9, max_features=200000,
                                                sublinear_tf=True, dtype=np.float32).fit(docs[universe])
            tfidf_rad = vectorizers[name].transform(docs)
            vectorizers[name].stop_words_ = None
        del docs
    log("text inputs ready")

    coverage = {"FT": has["text"][cohort_positions], "CXR": b.flags[rows_all, 1].astype(bool)}
    pool_runs = {}
    for name, run_dir in (("FT", OUT / "experts/ft"), ("CXR", OUT / "experts/cxr")):
        z = np.load(run_dir / "predictions.npz")
        pos = {int(s): i for i, s in enumerate(z["subject_id"])}
        if not np.array_equal(np.array([int(s) in pos for s in ids[cohort_positions]]), coverage[name]):
            raise ValueError(name + " predictions do not cover exactly the cohort patients with that modality")
        if json.loads((run_dir / "run.json").read_text())["train_partitions"] != "pool_extra":
            raise ValueError(name + " expert must come from a pool_extra-only model")
        evidence = np.full((len(ids), z["logits"].shape[1]), np.nan, np.float32)
        for i in cohort_positions:
            j = pos.get(int(ids[i]))
            if j is not None:
                evidence[i] = z["logits"][j]
        pool_runs[name] = {"evidence": evidence, "path": run_dir / "predictions.npz"}

    models, oof_hashes = {}, {}
    oof = {k: np.full((n_tr, 3), np.nan) for k in ("S", "R")}
    level1 = {k: np.full((len(targets), 3), np.nan) for k in ("S", "R")}
    for j, task in enumerate(TASKS):
        t = y_universe[task]
        rad_path, struct_path = OUT / "experts/radiology_oof" / f"oof_{task}.npz", OUT / "experts/structured_oof" / f"oof_{task}.npz"
        rad_oof, struct_oof = np.load(rad_path), np.load(struct_path)
        for z in (rad_oof, struct_oof):
            if not np.array_equal(z["rows"], train_rows):
                raise ValueError("OOF rows differ from the development patients")
        oof_hashes[task] = {"radiology": digest(rad_path), "structured": digest(struct_path)}
        task_models = {"S": lgb_seed_models(S[universe], t)}
        level1["S"][:, j] = np.mean([m.predict(S[targets], raw_score=True) for m in task_models["S"]], axis=0)
        oof["S"][:, j] = struct_oof["S"]
        fit_rows = universe[has["rad"][universe]]
        model = LogisticRegression(C=0.5, max_iter=3000, solver="liblinear").fit(tfidf_rad[fit_rows], t[fit_rows])
        obs = has["rad"][targets]
        level1["R"][obs, j] = model.decision_function(tfidf_rad[targets[obs]])
        oof["R"][:, j], task_models["R"] = rad_oof["R"], model
        models[task] = task_models
        log(task, "level-1 experts refit")

    dev_pos = np.arange(n_tr)
    tokens = {}
    for part, src, pos in (("dev", oof, dev_pos), ("target", level1, targets)):
        tok = {"S": src["S"], "I": pool_runs["CXR"]["evidence"][pos],
               "T": pool_runs["FT"]["evidence"][pos], "R": src["R"]}
        tokens[part] = {m: np.nan_to_num(v).astype(np.float32) for m, v in tok.items()}
    X = [np.vstack([tokens["dev"][m], tokens["target"][m]]) for m in MODS]
    delta = np.vstack([b.flags[train_rows], b.flags[meta_rows], b.flags[test_rows]]).astype(np.float32)
    Y = np.vstack([np.column_stack([y_oof[t] for t in TASKS]).astype(np.float32), np.zeros((len(targets), 3), np.float32)])
    strata = y_oof["readmission"] * 4 + y_oof["icu"] * 2 + y_oof["mortality"]
    fusion_folds = list(StratifiedKFold(5, shuffle=True, random_state=SPLIT_SEED + 1).split(train_rows, strata))
    cv = []
    for seed in CV_SEEDS:
        z = np.zeros((n_tr, 3), np.float32)
        for k, (fi, pi) in enumerate(fusion_folds):
            z[pi] = fit_h(X, delta, Y, fi, pi, seed * 10 + k)[0]
        cv.append(z)
        log("cross-validated shared-h fusion, seed", seed)
    p_cf_all = expit(np.mean(cv, axis=0))
    fits = [fit_h(X, delta, Y, dev_pos, np.arange(n_tr, len(Y)), s) for s in FIT_SEEDS]
    fused_logits = np.mean([f[0] for f in fits], axis=0)
    log("shared-h fusion fitted on", n_tr, "patients; seeds", FIT_SEEDS)

    pattern = {"oof": b.pattern[train_rows], "meta": b.pattern[meta_rows], "test": b.pattern[test_rows]}
    sel_pos, cal_pos = np.searchsorted(meta_rows, sel_rows), np.searchsorted(meta_rows, cal_rows)
    report, predictions = {"experts": experts, "fusion": "shared_h", "tasks": {}}, {}
    for j, task in enumerate(TASKS):
        task_models = models[task]
        fused_meta, fused_test = expit(fused_logits[:n_meta, j]), expit(fused_logits[n_meta:, j])
        p_cf = np.clip(p_cf_all[:, j], 1e-6, 1 - 1e-6)

        ym = y_meta[task]
        base_logit = logit(np.clip(fused_meta, 1e-6, 1 - 1e-6))
        cross_fitted, tau_folds = {}, {}
        for kappa in KAPPAS:
            q = np.clip(fused_meta, 1e-6, 1 - 1e-6)
            for name, est, app in (("selection", sel_pos, cal_pos), ("calibration", cal_pos, sel_pos)):
                tau_folds[name] = pattern_tau(ym[est], fused_meta[est], pattern["meta"][est])
                q[app] = apply_tau(fused_meta[app], pattern["meta"][app], tau_folds[name], kappa)
            cross_fitted[kappa] = q
        candidates = {kappa: metrics(ym, q) for kappa, q in cross_fitted.items()}
        merit = lambda m: m["ap"] + 0.2 * m["auc"] - m["brier"]
        chosen = max(KAPPAS, key=lambda kappa: (merit(candidates[kappa]), kappa))
        tau = pattern_tau(ym, fused_meta, pattern["meta"])
        rect_meta = logit(cross_fitted[chosen])
        rect_test = logit(apply_tau(fused_test, pattern["test"], tau, chosen))
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(rect_meta[cal_pos, None], ym[cal_pos])
        use_platt = brier_score_loss(ym[sel_pos], platt.predict_proba(rect_meta[sel_pos, None])[:, 1]) < brier_score_loss(ym[sel_pos], expit(rect_meta[sel_pos]))
        final = lambda z: platt.predict_proba(z[:, None])[:, 1] if use_platt else expit(z)
        final_meta, final_test = final(rect_meta), final(rect_test)
        global_platt = LogisticRegression(C=1e6, max_iter=1000).fit(base_logit[cal_pos, None], ym[cal_pos])
        global_meta = global_platt.predict_proba(base_logit[:, None])[:, 1]
        global_test = global_platt.predict_proba(logit(np.clip(fused_test, 1e-6, 1 - 1e-6))[:, None])[:, 1]
        task_models.update(rectifier_tau=tau, kappa=chosen, platt=platt if use_platt else None, global_platt=global_platt)

        expert_meta = {}
        for k in ("S", "R"):
            v = level1[k][:n_meta, j]
            m = np.isfinite(v)
            expert_meta["expert_" + k] = scores(ym[m], expit(v[m]), boot=0)
        report["tasks"][task] = {
            "meta_heldout_before_selection": {"fused": scores(ym, fused_meta), **expert_meta},
            "oof_cv_fused": scores(y_oof[task], p_cf, boot=0),
            "rectifier": {"kappa_cross_fitted": {str(k): v for k, v in candidates.items()}, "chosen_kappa": chosen,
                          "tau_pooled": tau, "tau_by_fold": tau_folds, "ppi_tau_on_meta": ppi_table(ym, fused_meta, pattern["meta"])},
            "calibration": {"use_platt": bool(use_platt), "platt": [float(platt.intercept_[0]), float(platt.coef_[0, 0])]},
            "selection_partition": {"fused": metrics(ym[sel_pos], fused_meta[sel_pos]), "global_only": metrics(ym[sel_pos], global_meta[sel_pos]),
                                    "final": metrics(ym[sel_pos], final_meta[sel_pos])},
            "calibration_partition": {"fused": metrics(ym[cal_pos], fused_meta[cal_pos]), "global_only": metrics(ym[cal_pos], global_meta[cal_pos]),
                                      "final": metrics(ym[cal_pos], final_meta[cal_pos])},
            "pattern_adjustment_meta": pattern_adjustment(ym, {"fused": fused_meta, "rectified": expit(rect_meta), "final": final_meta}, pattern["meta"]),
        }
        predictions[task] = {"fused_meta": fused_meta, "final_meta": final_meta, "global_meta": global_meta,
                             "fused_test": fused_test, "final_test": final_test, "global_test": global_test}
        r = report["tasks"][task]
        log(task, "meta held-out fused", json.dumps({k: round(r["meta_heldout_before_selection"]["fused"][k], 4) for k in ("auc", "ap", "brier")}),
            "| oof cv", round(r["oof_cv_fused"]["auc"], 4), round(r["oof_cv_fused"]["ap"], 4), "| kappa", chosen, "| platt", bool(use_platt))

    np.savez_compressed(out / "meta_predictions.npz", meta_rows=meta_rows,
                        **{f"{t}_{k}": v for t, d in predictions.items() for k, v in d.items() if k.endswith("_meta")})
    np.savez_compressed(out / "test_predictions.npz", test_rows=test_rows, test_subject_id=b.ids[test_rows],
                        **{f"{t}_{k}": v for t, d in predictions.items() for k, v in d.items() if "test" in k})
    joblib.dump({"models": models, "shared_h": {s: {"states": f[2], "standardization": f[3]} for s, f in zip(FIT_SEEDS, fits)},
                 "vectorizers": vectorizers, "experts": experts}, out / "models.joblib", compress=3)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    data_files = {"cohort.parquet": POOL / "cohort.parquet", "features_concepts.parquet": POOL / "features_concepts.parquet",
                  "features_supplement.parquet": POOL / "features_supplement.parquet"}
    for name, run in pool_runs.items():
        data_files[name.lower() + "_predictions.npz"] = run["path"]
    bundle = {"run_name": args.run_name, "experts": experts, "fusion": "shared_h", "split_manifest_sha256": data.split_sha256,
              "code_sha256": {c: digest(ROOT / c) for c in CODE},
              "data_sha256": {k: digest(v) for k, v in data_files.items()},
              "data_paths": {k: str(Path(v).relative_to(ROOT)) if Path(v).is_relative_to(ROOT) else str(v) for k, v in data_files.items()},
              "level1_oof_sha256": oof_hashes,
              "artifacts_sha256": {n: digest(out / n) for n in ("models.joblib", "test_predictions.npz", "meta_predictions.npz", "report.json")},
              "decisions": {t: {"kappa": models[t]["kappa"], "use_platt": report["tasks"][t]["calibration"]["use_platt"]} for t in TASKS}}
    (out / "frozen_bundle.json").write_text(json.dumps(bundle, indent=1, default=float))
    log("frozen", out / "frozen_bundle.json", "release key sha256:", digest(out / "frozen_bundle.json"))


def point(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], y)
    return {"auc": roc_auc_score(y, p), "ap": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "prevalence_brier": brier_score_loss(y, np.full(len(y), y.mean())),
            "ap_lift": average_precision_score(y, p) / y.mean(), "mean_risk": float(p.mean()), "observed": float(y.mean()),
            "calibration_intercept": float(cal.intercept_[0]), "calibration_slope": float(cal.coef_[0, 0]),
            "n": int(len(y)), "events": int(y.sum())}


def boot(y, preds, draws):
    names = list(preds)
    vals = {n: {"auc": [], "ap": [], "brier": []} for n in names}
    for i in draws:
        if not 0 < y[i].sum() < len(i):
            continue
        for n in names:
            vals[n]["auc"].append(roc_auc_score(y[i], preds[n][i]))
            vals[n]["ap"].append(average_precision_score(y[i], preds[n][i]))
            vals[n]["brier"].append(brier_score_loss(y[i], preds[n][i]))
    ci = {n: {m: np.percentile(v, [2.5, 97.5]).tolist() for m, v in vals[n].items()} for n in names}
    ref = names[0]
    deltas = {n: {m: [float(np.mean(np.subtract(vals[n][m], vals[ref][m]))),
                      *np.percentile(np.subtract(vals[n][m], vals[ref][m]), [2.5, 97.5]).tolist()] for m in ("auc", "ap", "brier")}
              for n in names[1:]}
    return ci, deltas


def evaluate(args):
    bdir = OUT / "final" / args.run_name
    lock = OUT / "final/TEST_RELEASED.json"
    bundle_path = bdir / "frozen_bundle.json"
    if digest(bundle_path) != args.release:
        raise SystemExit("Release key does not match frozen_bundle.json")
    bundle = json.loads(bundle_path.read_text())
    bad = [c for c, h in bundle["code_sha256"].items() if digest(ROOT / c) != h]
    bad += [n for n, h in bundle["artifacts_sha256"].items() if digest(bdir / n) != h]
    bad += [n for n, h in bundle["data_sha256"].items() if digest(ROOT / bundle["data_paths"][n]) != h]
    if bad:
        raise SystemExit(f"Frozen bundle no longer verifies: {bad}")
    if lock.exists():
        raise SystemExit(f"Test outcomes were already released: {lock.read_text()}")
    lock.write_text(json.dumps({"bundle": str(bundle_path.relative_to(ROOT)), "release_key": args.release,
                                "released_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))

    data = Cohort()
    if data.split_sha256 != bundle["split_manifest_sha256"]:
        raise SystemExit("Split manifest changed since freeze")
    pred = np.load(bdir / "test_predictions.npz")
    test_rows = data.splits["test"]
    if not np.array_equal(pred["test_rows"], test_rows):
        raise SystemExit("Pinned test predictions are for different patients")
    pattern = data.pattern[test_rows]
    rng = np.random.default_rng(20261022)
    draws = [rng.integers(0, len(test_rows), len(test_rows)) for _ in range(B)]
    report = {"bundle": str(bundle_path.relative_to(ROOT)), "release_key": args.release, "experts": bundle.get("experts"),
              "bootstrap": B, "n_test": int(len(test_rows)), "tasks": {}}
    for t in TASKS:
        y = data.target(t, test_rows, final_evaluation=True).astype(int)
        preds = {"global_only": pred[f"{t}_global_test"], "final": pred[f"{t}_final_test"], "fused_raw": pred[f"{t}_fused_test"]}
        ci, deltas = boot(y, preds, draws)
        res = {name: {**point(y, p), "ci": ci[name]} for name, p in preds.items()}
        res["delta_vs_global_only"] = deltas
        table = {}
        for g in range(8, 16):
            m = pattern == g
            row = {"n": int(m.sum()), "events": int(y[m].sum()) if m.any() else 0}
            if m.any():
                row.update(observed=float(y[m].mean()), mean_risk_fused=float(preds["fused_raw"][m].mean()),
                           mean_risk_global=float(preds["global_only"][m].mean()), mean_risk_final=float(preds["final"][m].mean()),
                           risk_change_pp_final_vs_global=float(100 * (preds["final"][m].mean() - preds["global_only"][m].mean())),
                           brier_global=float(brier_score_loss(y[m], preds["global_only"][m])),
                           brier_final=float(brier_score_loss(y[m], preds["final"][m])))
                if 0 < y[m].sum() < m.sum():
                    row.update(auc_final=float(roc_auc_score(y[m], preds["final"][m])),
                               ap_final=float(average_precision_score(y[m], preds["final"][m])))
            table[PATTERN_NAMES[g]] = row
        res["by_pattern"] = table
        report["tasks"][t] = res
        f = res["final"]
        print(f"{t}: final AUC {f['auc']:.4f} {np.round(f['ci']['auc'], 3)} AP {f['ap']:.4f} {np.round(f['ci']['ap'], 3)} "
              f"Brier {f['brier']:.4f} {np.round(f['ci']['brier'], 4)} | events {f['events']}", flush=True)
    (bdir / "test_report.json").write_text(json.dumps(report, indent=1, default=float))
    print("wrote", bdir / "test_report.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--run-name", default="final")
    p.set_defaults(func=fit)
    p = sub.add_parser("evaluate")
    p.add_argument("--run-name", default="final")
    p.add_argument("--release", required=True)
    p.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)
