"""Figures for the paper, drawn only from the saved result files.

python paper/make_figures.py   (from the repository root)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})


def eer_by_layer() -> None:
    res = json.load(open(os.path.join(ROOT, "results", "results.json")))
    itw = json.load(open(os.path.join(ROOT, "results", "in_the_wild", "itw_results.json")))
    layers = [3, 6, 9, 12]
    series = [("Clean", "#2a78d6", "o", lambda k: res["seed_summary"][k]["clean"]["eer"]["mean"]),
              ("Opus 16 kbps", "#1baf7a", "^", lambda k: res["seed_summary"][k]["opus16"]["eer"]["mean"]),
              ("Phone line", "#eda100", "s", lambda k: res["seed_summary"][k]["phone"]["eer"]["mean"]),
              ("Phone + noise", "#eb6834", "D", lambda k: res["seed_summary"][k]["phone_noise15"]["eer"]["mean"]),
              ("In-the-Wild", "#e87ba4", "v", lambda k: itw["seed_summary"][k]["eer"]["mean"])]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for name, col, mk, get in series:
        ys = [100 * get("wavlm_l%d+mlp" % l) for l in layers]
        ax.plot(layers, ys, color=col, linewidth=2, marker=mk, markersize=4.5, markeredgecolor="white",
                markeredgewidth=0.8, label=name)
    ax.set_xticks(layers)
    ax.set_xlabel("WavLM Base+ hidden layer (MLP head)")
    ax.set_ylabel("EER (%), mean of 3 seeds")
    ax.set_ylim(0, 32)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2, handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "eer_by_layer.pdf"))
    plt.close(fig)


def padding_effect() -> None:
    # Demo (one unpadded clip, float32) against the v0.3 batched features (zero-padded to the longest clip of
    # each batch of 48), on the 920 check clips; values from the v0.3 parity report (README, padding paragraph).
    groups = ["< 2 s", "2-3 s", "3-4 s", "4 s (cut)"]
    gap = [2.98, 1.07, 0.41, 0.01]
    agree = [89.0, 96.8, 99.1, 100.0]
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    bars = ax.bar(groups, gap, width=0.6, color="#2a78d6")
    for b, g, a in zip(bars, gap, agree):
        ax.text(b.get_x() + b.get_width() / 2, g + 0.06, "%.2f\n(%.1f%% agree)" % (g, a), ha="center",
                va="bottom", fontsize=6.5, color=INK)
    ax.set_ylim(0, 3.9)
    ax.set_ylabel("mean |$\\Delta$LLR|")
    ax.set_xlabel("clip length (zero-padded in the batch unless cut at 4 s)")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "padding_effect.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    eer_by_layer()
    padding_effect()
    print("figures written to", OUT)
