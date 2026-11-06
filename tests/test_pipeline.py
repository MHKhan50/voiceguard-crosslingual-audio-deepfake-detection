import json

import numpy as np

from voiceguard import pipeline
from voiceguard.report import write_report


def _data(n, shift, seed):
    rng = np.random.default_rng(seed)
    y = np.array([0] * (n // 4) + [1] * (n - n // 4))
    x = rng.standard_normal((n, 16)) + shift * y[:, None]
    return x.astype(np.float32), y


def test_end_to_end_mock_run(tmp_path):
    xtr, ytr = _data(400, 1.5, 0)
    xdv, ydv = _data(200, 1.5, 1)
    xev, yev = _data(300, 1.5, 2)
    att = ["bonafide" if v == 0 else ("A16" if i % 2 else "A07") for i, v in enumerate(yev)]
    noisy = xev + np.random.default_rng(3).standard_normal(xev.shape).astype(np.float32) * 2
    sets = {"clean": {"x": xev, "y": yev, "attacks": att},
            "phone": {"x": noisy, "y": yev, "attacks": att},
            "phone_noise15": {"x": noisy * 1.5, "y": yev, "attacks": att}}
    res = {"runs": {}}
    for kind in ("logreg", "mlp"):
        res["runs"]["toy+" + kind] = pipeline.run_experiment(xtr, ytr, xdv, ydv, sets, kind, ["A16"])
    r = res["runs"]["toy+logreg"]
    assert r["clean"]["eer"] < 0.15
    assert r["phone"]["eer"] > r["clean"]["eer"]
    assert set(r["clean"]["per_attack_eer"]) == {"A07", "A16"}
    assert r["clean"]["eer_ci95"][0] <= r["clean"]["eer"] <= r["clean"]["eer_ci95"][1]
    json.dumps(res)
    write_report(res, str(tmp_path))
    assert (tmp_path / "tables.md").exists() and (tmp_path / "fig_eer_per_attack.png").exists()
