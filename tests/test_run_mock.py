"""Whole Kaggle runner with a fake feature extractor (no torch, no audio)."""
import json
import sys
import types

import numpy as np


class FakeExtractor:
    def run(self, paths, condition):
        rng = np.random.default_rng(len(paths))
        y = np.array([1 if "S" in p.rsplit("/", 1)[-1] else 0 for p in paths])
        noise = {"clean": 1.0, "phone": 2.0, "phone_noise15": 3.0}[condition]
        base = rng.standard_normal((len(paths), 8)) * noise + 2 * y[:, None]
        return {"wavlm_l3": base.astype(np.float16), "mfcc": (base[:, :4] * 0.5).astype(np.float32)}


def _layout(root):
    attacks = {"train": ["A01", "A02"], "dev": ["A01", "A02"], "eval": ["A07", "A16"]}
    names = {"train": "train.trn", "dev": "dev.trl", "eval": "eval.trl"}
    for split, atks in attacks.items():
        d = root / "LA" / ("ASVspoof2019_LA_" + split) / "flac"
        d.mkdir(parents=True)
        lines = ["SPK%d B_%s_%03d - - bonafide" % (i % 3, split, i) for i in range(40)]
        for a in atks:
            lines += ["SPK%d S_%s_%s_%03d - %s spoof" % (i % 3, split, a, i, a) for i in range(40)]
        pdir = root / "LA" / "ASVspoof2019_LA_cm_protocols"
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / ("ASVspoof2019.LA.cm.%s.txt" % names[split])).write_text("\n".join(lines) + "\n")


def test_runner_end_to_end(tmp_path, monkeypatch):
    _layout(tmp_path / "data")
    fake = types.ModuleType("voiceguard.extract")
    fake.Extractor = FakeExtractor
    monkeypatch.setitem(sys.modules, "voiceguard.extract", fake)
    from voiceguard import run_kaggle
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["run", "--data", str(tmp_path / "data"), "--out", str(out),
                                      "--cache", str(tmp_path / "cache"), "--eval-per-attack", "30",
                                      "--eval-bonafide", "30", "--reuse-from", "",
                                      "--export", "wavlm_l3+mlp"])
    run_kaggle.main()
    res = json.loads((out / "results.json").read_text())
    assert res["counts"]["eval"] == 90
    assert set(res["runs"]) == {"mfcc+logreg", "mfcc+mlp", "wavlm_l3+logreg", "wavlm_l3+mlp"}
    r = res["runs"]["wavlm_l3+logreg"]
    assert r["clean"]["eer"] <= r["phone_noise15"]["eer"]
    assert (out / "fig_eer_by_condition.png").exists() and (out / "manifest.json").exists()
    assert 0 <= r["clean"]["min_cllr"] <= r["clean"]["cllr"] + 1e-9
    assert "## Calibration" in (out / "tables.md").read_text()
    assert res["classifier_seeds"] == [42, 43, 44]
    sm = res["seed_summary"]["wavlm_l3+mlp"]["clean"]["eer"]
    assert len(sm["values"]) == 3 and sm["min"] <= sm["mean"] <= sm["max"]
    assert res["seed_summary"]["wavlm_l3+logreg"]["clean"]["eer"]["std"] < 1e-12
    assert "## Classifier seeds" in (out / "tables.md").read_text()
    import joblib
    card = json.loads((out / "model" / "model_card.json").read_text())
    assert card["eval_clean_eer_check"]["match"] and card["feature_recipe"]["hidden_state_index"] == 3
    bundle = joblib.load(out / "model" / (card["name"] + ".joblib"))
    assert bundle["calibrator"].llr(np.array([0.2, 0.8])).shape == (2,)


def test_runner_reuses_features_from_earlier_run(tmp_path, monkeypatch):
    _layout(tmp_path / "data")
    fake = types.ModuleType("voiceguard.extract")
    fake.Extractor = FakeExtractor
    monkeypatch.setitem(sys.modules, "voiceguard.extract", fake)
    from voiceguard import run_kaggle
    common = ["--data", str(tmp_path / "data"), "--eval-per-attack", "30", "--eval-bonafide", "30"]
    first = tmp_path / "earlier"
    monkeypatch.setattr(sys, "argv", ["run", "--out", str(first / "results"), "--cache", str(first / "features"),
                                      "--reuse-from", ""] + common)
    run_kaggle.main()

    class NoExtractor:
        def __init__(self):
            raise AssertionError("features should have been reused")

    fake.Extractor = NoExtractor
    monkeypatch.setattr(sys, "argv", ["run", "--out", str(tmp_path / "new"), "--cache", str(tmp_path / "cache2"),
                                      "--reuse-from", str(tmp_path / "earlier_parent")] + common)
    (tmp_path / "earlier_parent").mkdir()
    first.rename(tmp_path / "earlier_parent" / "earlier")
    run_kaggle.main()
    assert (tmp_path / "new" / "results.json").exists()
    assert sorted(p.name for p in (tmp_path / "cache2").iterdir()) == sorted(
        p.name for p in (tmp_path / "earlier_parent" / "earlier" / "features").iterdir())
