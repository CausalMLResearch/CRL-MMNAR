import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABEL, LGB, OUT, POOL, SPLIT_SEED, TASKS, Cohort, log, notes_by_patient

FT_MODEL = "thomas-sounack/BioClinical-ModernBERT-base"
FT_LABELS = ["readmission_30d", "icu_need_after_discharge_90d", "mortality_after_discharge_90d", "readmission_90d"]


def rank_log(*args):
    if int(os.environ.get("RANK", 0)) == 0:
        print(time.strftime("%H:%M:%S"), *args, flush=True)


def tokenize_cache(path, model_name, max_len, pool):
    from transformers import AutoTokenizer
    if path.exists():
        return
    tok = AutoTokenizer.from_pretrained(model_name)
    window = pool[["subject_id", "hadm_id", "admittime", "dischtime"]]
    docs = notes_by_patient("discharge", window)
    body, head = max_len - 2, (max_len - 2) // 2
    ids, offsets, lengths = [], [0], []
    subjects = docs.index.to_numpy()
    for start in range(0, len(docs), 512):
        enc = tok(docs.iloc[start:start + 512].tolist(), add_special_tokens=False)["input_ids"]
        for seq in enc:
            lengths.append(len(seq))
            if len(seq) > body:
                seq = seq[:head] + seq[-(body - head):]
            seq = [tok.cls_token_id] + seq + [tok.sep_token_id]
            ids.extend(seq); offsets.append(offsets[-1] + len(seq))
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, subject_id=subjects, ids=np.asarray(ids, np.int32), offsets=np.asarray(offsets, np.int64),
             raw_lengths=np.asarray(lengths, np.int32), pad_id=tok.pad_token_id)
    tmp.replace(path)


def build_encoder(name, n_out):
    import torch
    from transformers import AutoModel

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = AutoModel.from_pretrained(name, attn_implementation="sdpa")
            self.body.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.drop = torch.nn.Dropout(0.1)
            self.head = torch.nn.Linear(self.body.config.hidden_size, n_out)

        def forward(self, ids, mask):
            h = self.body(input_ids=ids, attention_mask=mask).last_hidden_state
            pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            return self.head(self.drop(pooled))

    return Encoder()


def batches(index, lengths, batch, world, rank, seed, shuffle=True):
    rng = np.random.default_rng(seed)
    order = rng.permutation(index) if shuffle else np.array(index)
    chunks = [order[i:i + batch * 64] for i in range(0, len(order), batch * 64)]
    out = []
    for chunk in chunks:
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        out += [chunk[i:i + batch] for i in range(0, len(chunk), batch)]
    if shuffle:
        out = [out[i] for i in rng.permutation(len(out))]
    if shuffle:
        out = out[: len(out) // world * world]
    else:
        out += [np.array([], int)] * ((-len(out)) % world)
    return out[rank::world]


def collate(rows, cache, pad_id, device):
    import torch
    seqs = [cache["ids"][cache["offsets"][r]:cache["offsets"][r + 1]] for r in rows]
    width = max(len(s) for s in seqs)
    ids = np.full((len(seqs), width), pad_id, np.int64)
    mask = np.zeros((len(seqs), width), np.int64)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s; mask[i, :len(s)] = 1
    return torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device)


def ft_predict(model, rows, cache, lengths, pad_id, device, world, rank, batch):
    import torch
    import torch.distributed as dist
    model.eval()
    out_rows, out_logits = [], []
    with torch.inference_mode():
        for b in batches(rows, lengths, batch, world, rank, 0, shuffle=False):
            if len(b) == 0:
                continue
            ids, mask = collate(b, cache, pad_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out_logits.append(model(ids, mask).float().cpu().numpy()); out_rows.append(b)
    model.train()
    local = (np.concatenate(out_rows) if out_rows else np.array([], int),
             np.concatenate(out_logits) if out_logits else np.zeros((0, len(FT_LABELS)), np.float32))
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    rows_all = np.concatenate([g[0] for g in gathered]); logits_all = np.concatenate([g[1] for g in gathered])
    order = np.argsort(rows_all)
    return rows_all[order], logits_all[order]


def ft_cache_path(args):
    return OUT / "experts/ft" / f"tokens_{args.model.replace('/', '__')}_{args.max_len}.npz"


def tokenize(args):
    pool = pd.read_parquet(POOL / "cohort.parquet")
    cache_path = ft_cache_path(args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tokenize_cache(cache_path, args.model, args.max_len, pool)
    print("token cache", cache_path, flush=True)


def finetune(args):
    import torch
    import torch.distributed as dist
    from sklearn.metrics import roc_auc_score
    pool = pd.read_parquet(POOL / "cohort.parquet")
    cache_path = ft_cache_path(args)
    if not cache_path.exists():
        raise SystemExit("Build the token cache first with the tokenize stage")
    dist.init_process_group("nccl", timeout=timedelta(hours=1))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    out = OUT / "experts/ft"
    out.mkdir(parents=True, exist_ok=True)

    cache = dict(np.load(cache_path))
    pad_id = int(cache["pad_id"])
    lengths = np.diff(cache["offsets"])
    position = {int(s): i for i, s in enumerate(cache["subject_id"])}
    by_subject = pool.set_index("subject_id")
    train_ids = by_subject.index[by_subject.partition.isin(args.train_partitions.split(","))]
    train_ids = np.array([s for s in train_ids if s in position])
    predict_ids = by_subject.index[by_subject.partition.isin(args.predict_partitions.split(","))]
    predict_ids = np.array([s for s in predict_ids if s in position])
    if np.intersect1d(train_ids, predict_ids).size:
        raise ValueError("Train and predict partitions overlap")
    rng = np.random.default_rng(0)
    stop_mask = rng.random(len(train_ids)) < 0.03
    fit_rows = np.array([position[s] for s in train_ids[~stop_mask]])
    stop_rows = np.array([position[s] for s in train_ids[stop_mask]])
    predict_rows = np.array([position[s] for s in predict_ids])
    y_all = np.full((len(cache["subject_id"]), len(FT_LABELS)), np.nan, np.float32)
    lab = by_subject.loc[train_ids, FT_LABELS].to_numpy(np.float32)
    y_all[[position[s] for s in train_ids]] = lab
    rank_log(f"world {world}; fit {len(fit_rows)} stop {len(stop_rows)} predict {len(predict_rows)}; "
             f"notes >max_len: {(cache['raw_lengths'] > args.max_len - 2).mean():.3f}")

    model = build_encoder(args.model, len(FT_LABELS)).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    head_params = list(model.module.head.parameters())
    body_params = [p for n, p in model.named_parameters() if "head" not in n]
    opt = torch.optim.AdamW([{"params": body_params, "lr": args.lr}, {"params": head_params, "lr": 1e-3}], weight_decay=0.01)
    steps_per_epoch = len(batches(fit_rows, lengths, args.batch, world, rank, 0)) // args.accum
    total = int(steps_per_epoch * args.epochs)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)
    rank_log(f"optimizer steps {total}, eval every {eval_every}")

    best, history, step, epoch, t0 = -1.0, [], 0, 0, time.time()
    model.train()
    while step < total:
        plan = batches(fit_rows, lengths, args.batch, world, rank, seed=1000 + epoch)
        for i, b in enumerate(plan):
            ids, mask = collate(b, cache, pad_id, device)
            target = torch.from_numpy(y_all[b]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, mask)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), target) / args.accum
            sync = (i + 1) % args.accum == 0
            if sync:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            else:
                with model.no_sync():
                    loss.backward()
            if sync and step % 100 == 0:
                rank_log(f"step {step}/{total} loss {loss.item() * args.accum:.4f} {(time.time() - t0) / 60:.1f} min")
            if sync and (step % eval_every == 0 or step == total):
                rows, lg = ft_predict(model, stop_rows, cache, lengths, pad_id, device, world, rank, args.batch * 2)
                aucs = {FT_LABELS[k]: float(roc_auc_score(y_all[rows, k], lg[:, k])) for k in range(len(FT_LABELS))}
                score = float(np.mean([aucs[l] for l in FT_LABELS[:3]]))
                history.append({"step": step, "minutes": (time.time() - t0) / 60, "stop_auc": aucs, "mean_main_auc": score})
                rank_log("eval", history[-1])
                if score > best:
                    best = score
                    if rank == 0:
                        torch.save(model.module.state_dict(), out / "best.pt")
                dist.barrier()
            if step >= total:
                break
        epoch += 1

    model.module.load_state_dict(torch.load(out / "best.pt", map_location=device))
    rows, lg = ft_predict(model, predict_rows, cache, lengths, pad_id, device, world, rank, args.batch * 2)
    if rank == 0:
        np.savez_compressed(out / "predictions.npz", subject_id=cache["subject_id"][rows], logits=lg, labels=np.array(FT_LABELS))
        (out / "run.json").write_text(json.dumps({**{k: v for k, v in vars(args).items() if k != "func"}, "world": world,
            "fit_n": int(len(fit_rows)), "stop_n": int(len(stop_rows)), "predict_n": int(len(predict_rows)), "history": history,
            "best_mean_main_auc": best}, indent=1))
        rank_log("wrote", out / "predictions.npz")
    dist.destroy_process_group()


def linear(X, yv, fit, pred, mask, C, solver="liblinear"):
    from sklearn.linear_model import LogisticRegression
    out = np.full(len(pred), np.nan)
    fm, pm = fit[mask[fit]], pred[mask[pred]]
    m = LogisticRegression(C=C, max_iter=3000, solver=solver).fit(X[fm], yv[fm])
    out[mask[pred]] = m.decision_function(X[pm])
    return out


def radiology_oof(args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold
    out = OUT / "experts/radiology_oof"
    out.mkdir(parents=True, exist_ok=True)
    data = Cohort()
    rows = data.splits["train"]
    eval_ids = data.ids[rows]
    y_eval = {t: data.target(t, rows).astype(int) for t in TASKS}
    strata = y_eval["readmission"] * 4 + y_eval["icu"] * 2 + y_eval["mortality"]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SPLIT_SEED).split(rows, strata))

    pool = pd.read_parquet(POOL / "cohort.parquet")
    extra = pool[pool.partition.eq("pool_extra")].sort_values("subject_id")
    if not set(eval_ids) <= set(pool.loc[pool.partition.eq("cohort_train"), "subject_id"]):
        raise ValueError("Evaluation patients are not the pool's cohort_train partition")
    ids = np.concatenate([eval_ids, extra.subject_id.to_numpy()])
    n_eval = len(eval_ids)
    y = {t: np.concatenate([y_eval[t], extra[LABEL[t]].to_numpy().astype(int)]) for t in TASKS}
    log("training universe", len(ids), "eval", n_eval)

    window = pool[["subject_id", "hadm_id", "admittime", "dischtime"]]
    docs = notes_by_patient("radiology", window, set(ids.tolist())).reindex(ids).fillna("").to_numpy()
    if not np.array_equal(docs[:n_eval] != "", data.flags[rows, 3].astype(bool)):
        raise ValueError("rad availability differs from the observation flags on evaluation patients")
    has_rad = docs != ""
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9, max_features=200000,
                            sublinear_tf=True, dtype=np.float32).fit_transform(docs)
    log("tfidf rad", tfidf.shape)
    del docs

    extra_idx = np.arange(n_eval, len(ids))
    for task in TASKS:
        t = y[task]
        L = {"R": np.zeros(n_eval)}
        for fi, pi in folds:
            fit = np.concatenate([extra_idx, fi])
            L["R"][pi] = linear(tfidf, t, fit, pi, has_rad, 0.5)
            log(task, "fold done")
        np.savez_compressed(out / f"oof_{task}.npz", rows=rows, **L)
    log("wrote", out)


def fit_lgb(X, y, Xp, params, seed):
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    stop = rng.random(len(y)) < 0.1
    train = lgb.Dataset(X[~stop], y[~stop], free_raw_data=True)
    valid = lgb.Dataset(X[stop], y[stop], reference=train)
    model = lgb.train({"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": seed,
                       "num_threads": int(os.environ.get("OMP_NUM_THREADS", 8)), **params},
                      train, num_boost_round=6000, valid_sets=[valid],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
    return model.predict(Xp, raw_score=True), model.best_iteration


def structured_oof(args):
    from sklearn.model_selection import StratifiedKFold
    out = OUT / "experts/structured_oof"
    out.mkdir(parents=True, exist_ok=True)
    data = Cohort()
    rows = data.splits["train"]
    y_eval = {t: data.target(t, rows).astype(int) for t in TASKS}
    strata = y_eval["readmission"] * 4 + y_eval["icu"] * 2 + y_eval["mortality"]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SPLIT_SEED).split(rows, strata))
    pool = pd.read_parquet(POOL / "cohort.parquet")
    extra = pool[pool.partition.eq("pool_extra")].sort_values("subject_id")
    ids = np.concatenate([data.ids[rows], extra.subject_id.to_numpy()])
    n_eval = len(rows)
    y = np.concatenate([y_eval[args.task], extra[LABEL[args.task]].to_numpy()]).astype(int)
    base = pd.read_parquet(POOL / "features_concepts.parquet").set_index("subject_id").loc[ids]
    supp = pd.read_parquet(POOL / "features_supplement.parquet").set_index("subject_id").loc[ids]
    S = np.hstack([base.to_numpy(np.float32), supp.to_numpy(np.float32)])
    del base, supp
    extra_idx = np.arange(n_eval, len(ids))
    log(args.task, "features", S.shape)
    new = {}
    for seed in (0, 1, 2):
        z = np.zeros(n_eval)
        for fi, pi in folds:
            fit = np.concatenate([extra_idx, fi])
            z[pi], _ = fit_lgb(S[fit], y[fit], S[pi], LGB["lgb_a"], seed)
        new[f"lgb_a_s{seed}"] = z
        log("seed", seed, "done")
    np.savez_compressed(out / f"oof_{args.task}.npz", rows=rows, S=np.mean([new[k] for k in new], axis=0))
    log("wrote", out / f"oof_{args.task}.npz")


def last_study(img):
    df = pd.DataFrame({"sid": img["subject_id"], "study": img["study_id"], "h_dc": img["hours_to_discharge"]})
    study_h = df.groupby(["sid", "study"]).h_dc.transform("min")
    latest = study_h.eq(study_h.groupby(df.sid).transform("min"))
    last_id = df[latest].groupby("sid").study.first()
    rows = df.sid.map(last_id).eq(df.study).to_numpy()
    sel = df[rows]
    feats = pd.DataFrame(img["X"][rows]).groupby(sel.sid.to_numpy()).mean()
    return feats.index.to_numpy(), feats.to_numpy(np.float32)


def cxr(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    out = OUT / "experts/cxr"
    out.mkdir(parents=True, exist_ok=True)
    with np.load(POOL / "cxr_images.npz") as z:
        img = {k: z[k] for k in z.files}
    sid, X = last_study(img)
    pos = {int(s): i for i, s in enumerate(sid)}
    cohort = pd.read_parquet(POOL / "cohort.parquet")
    pool = cohort[cohort.partition.eq("pool_extra") & cohort.subject_id.isin(sid)].sort_values("subject_id")
    data = Cohort()
    rows = np.concatenate([data.splits[k] for k in ("train", "validation", "test")])
    target_ids, has_i = data.ids[rows], data.flags[rows, 1].astype(bool)
    rebuilt = np.array([int(s) in pos for s in target_ids])
    if not np.array_equal(rebuilt, has_i):
        raise ValueError(f"rebuilt CXR availability differs from the I flag on {int((rebuilt != has_i).sum())} cohort patients")
    Xp = X[[pos[int(s)] for s in pool.subject_id]]
    Xt = X[[pos[int(s)] for s in target_ids[has_i]]]
    scaler = StandardScaler().fit(Xp)
    Xp, Xt = scaler.transform(Xp), scaler.transform(Xt)
    logits = np.zeros((int(has_i.sum()), len(TASKS)), np.float32)
    for j, t in enumerate(TASKS):
        logits[:, j] = LogisticRegression(C=0.05, max_iter=3000, solver="liblinear").fit(Xp, pool[LABEL[t]].to_numpy()).decision_function(Xt)
    np.savez(out / "predictions.npz", subject_id=target_ids[has_i], logits=logits)
    run = {"expert": "CXR", "train_partitions": "pool_extra", "fit_n": int(len(pool)), "predict_n": int(has_i.sum())}
    (out / "run.json").write_text(json.dumps(run, indent=1))
    log(json.dumps(run))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    for name, func in (("tokenize", tokenize), ("finetune", finetune)):
        p = sub.add_parser(name)
        p.add_argument("--model", default=FT_MODEL)
        p.add_argument("--max-len", type=int, default=4096)
        p.add_argument("--epochs", type=float, default=2.0)
        p.add_argument("--lr", type=float, default=3e-5)
        p.add_argument("--batch", type=int, default=4)
        p.add_argument("--accum", type=int, default=2)
        p.add_argument("--evals-per-epoch", type=int, default=2)
        p.add_argument("--train-partitions", default="pool_extra")
        p.add_argument("--predict-partitions", default="cohort_train,cohort_validation,cohort_test")
        p.set_defaults(func=func)
    sub.add_parser("radiology-oof").set_defaults(func=radiology_oof)
    p = sub.add_parser("structured-oof")
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.set_defaults(func=structured_oof)
    sub.add_parser("cxr").set_defaults(func=cxr)
    args = parser.parse_args()
    args.func(args)
