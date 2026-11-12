"""EER by representation and channel condition for the website article, from results/results.json.

    python scripts/website_figures.py --out figures_web
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# House palette: blue, yellow, red, light green
CONDITIONS = [("clean", "Clean", "#1F4E9C"), ("phone", "Phone line", "#E3A800"),
              ("phone_noise15", "Phone line + noise (15 dB)", "#C8102E"), ("opus16", "Opus 16 kbps", "#5FA65A")]
MODELS = [("mfcc+mlp", "MFCC"), ("wavlm_l3+mlp", "WavLM\nlayer 3"), ("wavlm_l6+mlp", "WavLM\nlayer 6"),
          ("wavlm_l9+mlp", "WavLM\nlayer 9"), ("wavlm_l12+mlp", "WavLM\nlayer 12")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures_web")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    ss = json.loads((ROOT / "results" / "results.json").read_text())["seed_summary"]
    fig, ax = plt.subplots(figsize=(10.7, 4.9))
    x = np.arange(len(MODELS))
    w = 0.2
    for i, (key, label, colour) in enumerate(CONDITIONS):
        means = [100 * ss[m][key]["eer"]["mean"] for m, _ in MODELS]
        sds = [100 * ss[m][key]["eer"]["std"] for m, _ in MODELS]
        ax.bar(x + (i - 1.5) * w, means, w * 0.92, yerr=sds, capsize=2.5, color=colour, label=label,
               error_kw={"elinewidth": 1, "ecolor": "#222222"})
    ax.set_xticks(x, [name for _, name in MODELS])
    ax.set_ylabel("Equal error rate (%), lower is better")
    ax.grid(axis="y", color="#e4e6ec", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.1), frameon=False)
    fig.tight_layout()
    fig.savefig(out / "voiceguard-eer-by-layer.png", dpi=150)
    print("wrote", out / "voiceguard-eer-by-layer.png")


if __name__ == "__main__":
    main()
