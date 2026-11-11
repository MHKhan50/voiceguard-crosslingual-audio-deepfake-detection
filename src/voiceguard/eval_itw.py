"""Out-of-domain test: models trained on ASVspoof 2019 LA, evaluated on the In-the-Wild dataset.

In-the-Wild (Mueller et al., Interspeech 2022) holds real and fake speech of public figures collected online.
Nothing is trained or tuned on it: classifiers and calibrators are fitted on ASVspoof 2019 LA train/dev exactly as
in run_kaggle, then scored on every In-the-Wild clip (first 4 s, as in all other experiments).

python -m voiceguard.eval_itw --data /kaggle/input --out /kaggle/working/results_itw
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import numpy as np

from . import metrics as M
from . import pipeline
from .protocol import find_dir, find_file, read_protocol, stratified_sample
from .run_kaggle import find_reusable_features

ITW_REPO = "mueller91/In-The-Wild"
ITW_ZIP = "release_in_the_wild.zip"
ITW_URL = "https://huggingface.co/datasets/%s/resolve/main/%s" % (ITW_REPO, ITW_ZIP)
FAKE_AT, REAL_AT = 0.95, 0.05  # demo thresholds on the posterior at an even prior (demo/app.js)
SR, MAX_SEC = 16000, 4.0


# ---------------------------------------------------------------- data

def label_of(value: str) -> int:
    v = value.strip().lower()
    if v in ("spoof", "fake", "deepfake"):
        return 1
    if v in ("bona-fide", "bonafide", "bona fide", "real", "genuine"):
        return 0
    raise ValueError("unknown In-the-Wild label: %r" % value)


def read_meta(path: str) -> List[Dict[str, object]]:
    """Rows of meta.csv as {file, speaker, label}. Column names are detected, not assumed."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError("empty meta file: " + path)
    cols = list(rows[0])
    fcol = next(c for c in cols if str(rows[0][c]).lower().endswith(".wav"))
    lcol = next(c for c in cols if c != fcol and str(rows[0][c]).strip().lower() in
                ("spoof", "fake", "deepfake", "bona-fide", "bonafide", "bona fide", "real", "genuine"))
    scol = next((c for c in cols if c not in (fcol, lcol)), None)
    return [{"file": r[fcol], "speaker": r[scol] if scol else "", "label": label_of(r[lcol])} for r in rows]


def find_itw(root: str) -> str:
    """Folder holding meta.csv next to In-the-Wild wav files, or ""."""
    if not root or not os.path.isdir(root):
        return ""
    for dirpath, dirs, files in os.walk(root):
        if "meta.csv" in files and any(f.endswith(".wav") for f in files):
            return dirpath
        dirs[:] = [d for d in dirs if d not in ("flac", "features")]
    return ""


def download_itw(dest: str) -> str:
    """Download and unzip In-the-Wild from the authors' Hugging Face page; return the folder with meta.csv."""
    os.makedirs(dest, exist_ok=True)
    found = find_itw(dest)
    if found:
        return found
    zpath = os.path.join(dest, ITW_ZIP)
    t0 = time.time()
    try:
        from huggingface_hub import hf_hub_download
        zpath = hf_hub_download(ITW_REPO, ITW_ZIP, repo_type="dataset", local_dir=dest)
    except ImportError:
        urllib.request.urlretrieve(ITW_URL, zpath)
    print("downloaded %s (%.1f GB) in %.1f min" % (ITW_ZIP, os.path.getsize(zpath) / 1e9, (time.time() - t0) / 60))
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    os.remove(zpath)
    found = find_itw(dest)
    if not found:
        raise FileNotFoundError("meta.csv with wav files not found after unzipping " + ITW_ZIP)
    return found


def _prepare_one(job: Tuple[str, str]) -> Tuple[int, float]:
    """Write the first 4 s of a clip as 16 kHz mono 16-bit wav. Returns (source rate, source seconds)."""
    import soundfile as sf
    from scipy.signal import resample_poly
    src, dst = job
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    dur = len(x) / sr
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = x[: int(np.ceil(MAX_SEC * sr)) + sr // 10]
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    x = x[: int(MAX_SEC * SR)]
    sf.write(dst, np.clip(x, -1, 1), SR, subtype="PCM_16")
    return int(sr), float(dur)


def prepare_clips(itw_dir: str, rows: List[Dict], out_dir: str, workers: int = 4) -> Dict[str, object]:
    """16 kHz, 4 s copies of every clip, so the shared extractor sees the same input format as for ASVspoof."""
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(os.path.join(itw_dir, r["file"]), os.path.join(out_dir, r["file"])) for r in rows]
    todo = [j for j in jobs if not os.path.exists(j[1])]
    rates, durs = [], []
    with ProcessPoolExecutor(workers) as ex:
        for sr, d in ex.map(_prepare_one, todo, chunksize=64):
            rates.append(sr)
            durs.append(d)
    return {"source_rates": sorted(set(rates)), "seconds_mean": float(np.mean(durs)) if durs else None,
            "seconds_median": float(np.median(durs)) if durs else None,
            "share_shorter_than_4s": float(np.mean(np.asarray(durs) < MAX_SEC)) if durs else None}


# ---------------------------------------------------------------- scoring

def demo_verdicts(llr: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Share of each class that the public demo would call fake, real or uncertain."""
    from scipy.special import expit
    p = expit(llr)
    out = {}
    for name, cls in (("bona_fide", 0), ("spoof", 1)):
        q = p[y == cls]
        out[name] = {"fake": float(np.mean(q >= FAKE_AT)), "real": float(np.mean(q <= REAL_AT)),
                     "uncertain": float(np.mean((q > REAL_AT) & (q < FAKE_AT)))}
    return out


def threshold_transfer(dev_raw, dev_y, raw, y) -> Dict[str, float]:
    """Fix the threshold at the ASVspoof dev EER point, then measure both error rates on In-the-Wild."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(dev_y, dev_raw)
    i = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    t = float(thr[i])
    fa = float(np.mean(raw[y == 0] >= t))  # real speech called fake
    miss = float(np.mean(raw[y == 1] < t))  # fake speech called real
    return {"false_alarm": fa, "miss": miss, "hter": (fa + miss) / 2}


def score_model(model, cal, dev_raw, dev_y, x, y) -> Dict:
    raw = model.predict_proba(x)[:, 1]
    llr = cal.llr(raw)
    prior = float(np.mean(y))
    return {"eer": M.eer(y, raw), "eer_ci95": M.bootstrap_eer_ci(y, raw, n=200), "auc": M.auc(y, raw),
            "cllr": M.cllr(y, llr), "min_cllr": M.min_cllr(y, raw),
            "ece": M.ece(y, cal.posterior(raw, prior)), "target_prior": prior,
            "dev_eer_threshold": threshold_transfer(dev_raw, dev_y, raw, y),
            "demo_verdicts": demo_verdicts(llr, y)}


SUMMARY = ("eer", "auc", "cllr", "min_cllr", "ece")


def write_tables(res: Dict, out: str) -> None:
    keys = sorted(res["runs"])
    L = ["# In-the-Wild (trained on ASVspoof 2019 LA only)", "",
         "%d clips (%d bona fide, %d spoof), first 4 s of each. Mean over classifier seeds %s; "
         "sd in brackets." % (res["counts"]["clips"], res["counts"]["bona_fide"], res["counts"]["spoof"],
                              ", ".join(map(str, res["classifier_seeds"]))), "",
         "| Model | EER % | AUC | Cllr | minCllr | ECE |", "|---|---:|---:|---:|---:|---:|"]
    for k in keys:
        s = res["seed_summary"][k]
        L.append("| %s | %.1f (%.1f) | %.3f | %.2f | %.2f | %.3f |" % (
            k, 100 * s["eer"]["mean"], 100 * s["eer"]["std"], s["auc"]["mean"], s["cllr"]["mean"],
            s["min_cllr"]["mean"], s["ece"]["mean"]))
    s0 = res["classifier_seeds"][0]
    L += ["", "## Threshold fixed on ASVspoof dev (seed %d)" % s0, "",
          "| Model | Real called fake % | Fake called real % | HTER % |", "|---|---:|---:|---:|"]
    for k in keys:
        t = res["runs"][k]["dev_eer_threshold"]
        L.append("| %s | %.1f | %.1f | %.1f |" % (k, 100 * t["false_alarm"], 100 * t["miss"], 100 * t["hter"]))
    L += ["", "## Demo rule (fake if p >= %.2f, real if p <= %.2f, even prior; seed %d)" % (FAKE_AT, REAL_AT, s0), "",
          "| Model | Real: fake % | Real: uncertain % | Real: real % | Fake: fake % | Fake: uncertain % "
          "| Fake: real % |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k in keys:
        d = res["runs"][k]["demo_verdicts"]
        L.append("| %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |" % (
            k, 100 * d["bona_fide"]["fake"], 100 * d["bona_fide"]["uncertain"], 100 * d["bona_fide"]["real"],
            100 * d["spoof"]["fake"], 100 * d["spoof"]["uncertain"], 100 * d["spoof"]["real"]))
    with open(os.path.join(out, "itw_tables.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/input", help="ASVspoof 2019 LA (and optionally In-the-Wild)")
    ap.add_argument("--out", default="/kaggle/working/results_itw")
    ap.add_argument("--cache", default="/kaggle/working/features")
    ap.add_argument("--itw-dir", default="", help="folder with meta.csv; empty = search --data, then download")
    ap.add_argument("--download-to", default="/tmp/itw")
    ap.add_argument("--clips16k", default="/tmp/itw16k")
    ap.add_argument("--reuse-from", default="/kaggle/input")
    ap.add_argument("--seed", type=int, default=42, help="seed of the ASVspoof dev sample (as in run_kaggle)")
    ap.add_argument("--clf-seeds", default="42,43,44")
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips (for quick checks only)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    t_start = time.time()

    itw = args.itw_dir or find_itw(args.data) or download_itw(args.download_to)
    rows = read_meta(os.path.join(itw, "meta.csv"))
    if args.limit:
        rng = np.random.default_rng(0)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), args.limit, replace=False))]
    y = np.array([r["label"] for r in rows])
    print("In-the-Wild: %d clips (%d bona fide, %d spoof) in %s" % (len(rows), (y == 0).sum(), (y == 1).sum(), itw))
    info = prepare_clips(itw, rows, args.clips16k)
    print("clips prepared:", info, flush=True)

    proto = {s: read_protocol(find_file(args.data, "ASVspoof2019.LA.cm.%s.txt" % s)) for s in ("train.trn", "dev.trl")}
    dirs = {"train.trn": find_dir(args.data, "ASVspoof2019_LA_train"),
            "dev.trl": find_dir(args.data, "ASVspoof2019_LA_dev")}
    train = proto["train.trn"]
    dev = stratified_sample(proto["dev.trl"], {"bonafide": 1000}, 170, args.seed)
    reuse = find_reusable_features(args.reuse_from, {"train": [u.utt_id for u in train],
                                                     "dev": [u.utt_id for u in dev]})
    print("reusing ASVspoof features from", reuse or "nowhere (extracting)")
    ext = None

    def feats(fname, paths):
        nonlocal ext
        path = os.path.join(args.cache, fname)
        if os.path.exists(path):
            return dict(np.load(path))
        if reuse and os.path.exists(os.path.join(reuse, fname)):
            shutil.copy(os.path.join(reuse, fname), path)
            return dict(np.load(path))
        if ext is None:
            from .extract import Extractor  # torch only here
            ext = Extractor()
        f = ext.run(paths, "clean")
        np.savez(path, **f)
        return f

    Ftr = feats("train%d_clean_nopad.npz" % len(train),
                [os.path.join(dirs["train.trn"], u.utt_id + ".flac") for u in train])
    Fdv = feats("dev%d_clean_nopad.npz" % len(dev), [os.path.join(dirs["dev.trl"], u.utt_id + ".flac") for u in dev])
    Fit = feats("itw%d_clean_nopad.npz" % len(rows), [os.path.join(args.clips16k, r["file"]) for r in rows])
    print("features done in %.1f min" % ((time.time() - t_start) / 60), flush=True)

    ytr = np.array([u.label for u in train])
    ydv = np.array([u.label for u in dev])
    seeds = [int(s) for s in args.clf_seeds.split(",") if s.strip()]
    res = {"dataset": {"name": "In-the-Wild", "source": ITW_URL, "dir": itw, **info},
           "counts": {"clips": len(rows), "bona_fide": int((y == 0).sum()), "spoof": int((y == 1).sum()),
                      "speakers": len({r["speaker"] for r in rows}), "train": len(train), "dev": len(dev)},
           "classifier_seeds": seeds, "runs": {}, "seed_summary": {}}
    for feat in sorted(Ftr):
        for kind in ("logreg", "mlp"):
            key = "%s+%s" % (feat, kind)
            per = {}
            for s in seeds:
                model = pipeline.make_model(kind, s).fit(Ftr[feat].astype(np.float32), ytr)
                dev_raw = model.predict_proba(Fdv[feat].astype(np.float32))[:, 1]
                cal = pipeline.LLRCalibrator().fit(dev_raw, ydv)
                per[s] = score_model(model, cal, dev_raw, ydv, Fit[feat].astype(np.float32), y)
            res["runs"][key] = per[seeds[0]]
            res["seed_summary"][key] = {m: {"mean": float(np.mean([per[s][m] for s in seeds])),
                                            "std": float(np.std([per[s][m] for s in seeds], ddof=1))
                                            if len(seeds) > 1 else 0.0,
                                            "values": [float(per[s][m]) for s in seeds]} for m in SUMMARY}
            sm = res["seed_summary"][key]
            print("%-16s ITW EER %.1f%% (sd %.1f)  Cllr %.2f  minCllr %.2f" % (
                key, 100 * sm["eer"]["mean"], 100 * sm["eer"]["std"], sm["cllr"]["mean"], sm["min_cllr"]["mean"]),
                flush=True)
    res["minutes"] = round((time.time() - t_start) / 60, 1)
    with open(os.path.join(args.out, "itw_results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    write_tables(res, args.out)
    print("done in %.1f min; results in %s" % (res["minutes"], args.out))


if __name__ == "__main__":
    main()
