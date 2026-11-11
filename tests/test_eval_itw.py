"""In-the-Wild evaluation with fake features (no torch); the audio step is tested where soundfile exists."""
import json
import sys
import types

import numpy as np
import pytest

from test_run_mock import FakeExtractor, _layout


def _itw(root, n=60):
    root.mkdir(parents=True)
    lines = ["file,speaker,label"]
    for i in range(n):
        lines.append("%d.wav,Person %d,%s" % (i, i % 4, "spoof" if i % 2 else "bona-fide"))
        (root / ("%d.wav" % i)).write_bytes(b"")
    (root / "meta.csv").write_text("\n".join(lines) + "\n")


def test_read_meta_detects_columns(tmp_path):
    from voiceguard import eval_itw
    _itw(tmp_path / "itw", 6)
    rows = eval_itw.read_meta(str(tmp_path / "itw" / "meta.csv"))
    assert [r["label"] for r in rows] == [0, 1, 0, 1, 0, 1]
    assert rows[1] == {"file": "1.wav", "speaker": "Person 1", "label": 1}
    assert eval_itw.find_itw(str(tmp_path)) == str(tmp_path / "itw")
    with pytest.raises(ValueError):
        eval_itw.label_of("maybe")


def test_demo_verdicts_and_threshold_transfer():
    from voiceguard import eval_itw
    y = np.array([0, 0, 0, 1, 1, 1])
    llr = np.array([-5.0, 0.0, 5.0, -5.0, 0.0, 5.0])  # p = 0.007, 0.5, 0.993
    d = eval_itw.demo_verdicts(llr, y)
    for cls in ("bona_fide", "spoof"):
        assert d[cls]["fake"] == pytest.approx(1 / 3) and d[cls]["real"] == pytest.approx(1 / 3)
        assert d[cls]["uncertain"] == pytest.approx(1 / 3)
    dev_y = np.array([0, 0, 1, 1])
    t = eval_itw.threshold_transfer(np.array([0.1, 0.2, 0.8, 0.9]), dev_y,
                                    np.array([0.85, 0.1, 0.9, 0.3]), np.array([0, 0, 1, 1]))
    assert t == {"false_alarm": 0.5, "miss": 0.5, "hter": 0.5}


def test_eval_itw_end_to_end(tmp_path, monkeypatch):
    _layout(tmp_path / "data")
    _itw(tmp_path / "itw")
    fake = types.ModuleType("voiceguard.extract")

    class ItwAwareExtractor(FakeExtractor):
        def run(self, paths, condition):
            if any(p.endswith(".wav") for p in paths):  # In-the-Wild: odd file numbers are spoof
                paths = ["S" if int(p.rsplit("/", 1)[-1][:-4]) % 2 else "B" for p in paths]
            return super().run(paths, condition)

    fake.Extractor = ItwAwareExtractor
    monkeypatch.setitem(sys.modules, "voiceguard.extract", fake)
    from voiceguard import eval_itw
    monkeypatch.setattr(eval_itw, "prepare_clips", lambda *a, **k: {"source_rates": [16000]})
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["eval", "--data", str(tmp_path / "data"), "--itw-dir", str(tmp_path / "itw"),
                                      "--out", str(out), "--cache", str(tmp_path / "cache"), "--reuse-from", ""])
    eval_itw.main()
    res = json.loads((out / "itw_results.json").read_text())
    assert res["counts"] == {"clips": 60, "bona_fide": 30, "spoof": 30, "speakers": 4, "train": 120, "dev": 120}
    assert set(res["runs"]) == {"mfcc+logreg", "mfcc+mlp", "wavlm_l3+logreg", "wavlm_l3+mlp"}
    r = res["runs"]["wavlm_l3+logreg"]
    assert r["eer"] < 0.2 and 0 <= r["min_cllr"] <= r["cllr"] + 1e-9
    assert set(r["dev_eer_threshold"]) == {"false_alarm", "miss", "hter"}
    assert len(res["seed_summary"]["wavlm_l3+mlp"]["eer"]["values"]) == 3
    assert res["seed_summary"]["wavlm_l3+logreg"]["eer"]["std"] < 1e-12
    tables = (out / "itw_tables.md").read_text()
    assert "## Threshold fixed on ASVspoof dev" in tables and "## Demo rule" in tables
    assert (tmp_path / "cache" / "itw60_clean_nopad.npz").exists()


def test_prepare_one_resamples_and_crops(tmp_path):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("scipy")
    from voiceguard import eval_itw
    t = np.arange(int(6.0 * 44100)) / 44100
    sf.write(str(tmp_path / "a.wav"), np.stack([np.sin(2 * np.pi * 440 * t)] * 2, 1) * 0.5, 44100)
    sr, dur = eval_itw._prepare_one((str(tmp_path / "a.wav"), str(tmp_path / "b.wav")))
    x, sr2 = sf.read(str(tmp_path / "b.wav"))
    assert (sr, sr2, x.ndim, len(x)) == (44100, 16000, 1, 64000) and dur == pytest.approx(6.0)
