"""End-to-end run on Kaggle: sample ASVspoof 2019 LA, extract features, train, evaluate, plot.

python -m voiceguard.run_kaggle --data /kaggle/input --out /kaggle/working/results
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import pipeline
from .protocol import find_dir, find_file, read_protocol, stratified_sample

KNOWN_ATTACKS = ["A16", "A19"]  # same algorithms as A04 and A06 in training (ASVspoof 2019 LA design)
CONDITIONS = ["clean", "phone", "phone_noise15"]


def find_reusable_features(root: str, ids: dict) -> str:
    """Return a features/ folder from an earlier run whose results/manifest.json lists the same clips, else ""."""
    if not root or not os.path.isdir(root):
        return ""
    for dirpath, dirs, _files in os.walk(root):
        if "features" in dirs:
            man = os.path.join(dirpath, "results", "manifest.json")
            if os.path.exists(man):
                with open(man) as fh:
                    m = json.load(fh)
                if all(m.get(k) == v for k, v in ids.items()):
                    return os.path.join(dirpath, "features")
        dirs[:] = [d for d in dirs if d not in ("flac", "features")]
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working/results")
    ap.add_argument("--cache", default="/kaggle/working/features")
    ap.add_argument("--eval-per-attack", type=int, default=600)
    ap.add_argument("--eval-bonafide", type=int, default=2000)
    ap.add_argument("--train-limit", type=int, default=0, help="0 = all training clips")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-from", default="/kaggle/input",
                    help="search here for features saved by an earlier run with the same clip IDs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)

    from .extract import Extractor  # torch only here

    t_start = time.time()
    proto = {s: read_protocol(find_file(args.data, "ASVspoof2019.LA.cm.%s.txt" % s))
             for s in ("train.trn", "dev.trl", "eval.trl")}
    dirs = {"train.trn": find_dir(args.data, "ASVspoof2019_LA_train"),
            "dev.trl": find_dir(args.data, "ASVspoof2019_LA_dev"),
            "eval.trl": find_dir(args.data, "ASVspoof2019_LA_eval")}
    train = proto["train.trn"]
    if args.train_limit:
        train = stratified_sample(train, {}, args.train_limit // 7, args.seed)
    dev = stratified_sample(proto["dev.trl"], {"bonafide": 1000}, 170, args.seed)
    ev = stratified_sample(proto["eval.trl"], {"bonafide": args.eval_bonafide},
                           args.eval_per_attack, args.seed)
    print("clips: train %d, dev %d, eval %d x %d conditions" % (len(train), len(dev), len(ev), len(CONDITIONS)))
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump({"train": [u.utt_id for u in train], "dev": [u.utt_id for u in dev],
                   "eval": [u.utt_id for u in ev], "seed": args.seed}, fh)

    ids = {"train": [u.utt_id for u in train], "dev": [u.utt_id for u in dev], "eval": [u.utt_id for u in ev]}
    reuse = find_reusable_features(args.reuse_from, ids)
    if reuse:
        print("reusing features from", reuse)
    ext = None

    def feats(name, items, split, cond):
        nonlocal ext
        fname = "%s%d_%s.npz" % (name, len(items), cond)
        for d in (args.cache, reuse):
            if d and os.path.exists(os.path.join(d, fname)):
                return dict(np.load(os.path.join(d, fname)))
        if ext is None:
            ext = Extractor()
        path = os.path.join(args.cache, fname)
        paths = [os.path.join(dirs[split], u.utt_id + ".flac") for u in items]
        f = ext.run(paths, cond)
        np.savez(path, **f)
        return f

    F = {"train": feats("train", train, "train.trn", "clean"),
         "dev": feats("dev", dev, "dev.trl", "clean")}
    for c in CONDITIONS:
        F["eval_" + c] = feats("eval", ev, "eval.trl", c)
    print("features done in %.1f min" % ((time.time() - t_start) / 60))

    ytr = np.array([u.label for u in train])
    ydv = np.array([u.label for u in dev])
    yev = np.array([u.label for u in ev])
    atk = [u.attack for u in ev]
    results = {"counts": {"train": len(train), "dev": len(dev), "eval": len(ev),
                          "eval_bonafide": int((yev == 0).sum())},
               "calibration": "affine LLR map fitted on dev with equal class weights; posterior at the eval "
                              "sample's spoof share (target_prior); Platt on dev kept as ece_platt_dev",
               "majority_baseline": pipeline.majority_baseline(ytr, yev), "runs": {}}
    for feat in sorted(F["train"]):
        for kind in ("logreg", "mlp"):
            key = "%s+%s" % (feat, kind)
            sets = {c: {"x": F["eval_" + c][feat].astype(np.float32), "y": yev, "attacks": atk}
                    for c in CONDITIONS}
            results["runs"][key] = pipeline.run_experiment(
                F["train"][feat].astype(np.float32), ytr, F["dev"][feat].astype(np.float32), ydv,
                sets, kind, KNOWN_ATTACKS, args.seed)
            r = results["runs"][key]
            print("%-16s EER clean %.1f%%  phone %.1f%%  phone+noise %.1f%%  Cllr clean %.3f" % (
                key, 100 * r["clean"]["eer"], 100 * r["phone"]["eer"], 100 * r["phone_noise15"]["eer"],
                r["clean"]["cllr"]),
                flush=True)
    results["minutes"] = round((time.time() - t_start) / 60, 1)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    from .report import write_report
    write_report(results, args.out)
    print("done; results in", args.out)


if __name__ == "__main__":
    main()
