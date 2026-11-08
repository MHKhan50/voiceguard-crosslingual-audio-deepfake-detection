"""Tables and figures from results.json (no GPU needed)."""
from __future__ import annotations

import os
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COND_LABEL = {"clean": "Clean", "phone": "Phone line", "phone_noise15": "Phone + noise 15 dB",
              "opus16": "Opus 16 kbps"}


def table(results: Dict) -> str:
    rows = ["| Features + classifier | Condition | EER % [95% CI] | Unseen-attack EER % | AUC | ECE | Brier | Error at 80% coverage |",
            "|---|---|---|---|---|---|---|---|"]
    for key, r in results["runs"].items():
        for c, m in r.items():
            rc80 = [x["error"] for x in m["risk_coverage"] if abs(x["coverage"] - 0.8) < 1e-9][0]
            rows.append("| %s | %s | %.1f [%.1f, %.1f] | %.1f | %.3f | %.3f | %.3f | %.1f%% |" % (
                key, COND_LABEL[c], 100 * m["eer"], 100 * m["eer_ci95"][0], 100 * m["eer_ci95"][1],
                100 * m["eer_unseen_attacks"], m["auc"], m["ece"], m["brier"], 100 * rc80))
    return "\n".join(rows)


def calibration_table(results: Dict) -> str:
    rows = ["| Features + classifier | Condition | Cllr | minCllr | Calibration loss | ECE raw | ECE Platt (dev mix) | ECE LLR at eval prior |",
            "|---|---|---|---|---|---|---|---|"]
    for key, r in results["runs"].items():
        for c, m in r.items():
            rows.append("| %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
                key, COND_LABEL[c], m["cllr"], m["min_cllr"], m["cllr"] - m["min_cllr"],
                m["ece_uncalibrated"], m["ece_platt_dev"], m["ece"]))
    return "\n".join(rows)


def seed_table(results: Dict) -> str:
    rows = ["| Features + classifier | Condition | EER % mean (sd) | Cllr mean (sd) | minCllr mean (sd) | ECE mean (sd) |",
            "|---|---|---|---|---|---|"]
    for key, r in results["seed_summary"].items():
        for c, m in r.items():
            rows.append("| %s | %s | %.2f (%.2f) | %.3f (%.3f) | %.3f (%.3f) | %.3f (%.3f) |" % (
                key, COND_LABEL[c], 100 * m["eer"]["mean"], 100 * m["eer"]["std"], m["cllr"]["mean"],
                m["cllr"]["std"], m["min_cllr"]["mean"], m["min_cllr"]["std"], m["ece"]["mean"], m["ece"]["std"]))
    return "\n".join(rows)


def write_report(results: Dict, out: str) -> None:
    with open(os.path.join(out, "tables.md"), "w") as fh:
        fh.write("# VoiceGuard baseline results\n\n" + table(results) + "\n")
        fh.write("\n## Calibration\n\nCllr and minCllr in bits (lower is better) do not depend on the class mix. "
                 "ECE columns: raw classifier output, Platt fitted on dev (v0.2 method), and the LLR calibrator "
                 "at the eval sample's spoof share.\n\n" + calibration_table(results) + "\n")
        if results.get("seed_summary"):
            seeds = results.get("classifier_seeds", [])
            fh.write("\n## Classifier seeds\n\nSame clips and features; only the classifier seed changes (%s). "
                     "The tables above use seed %s. sd is the sample standard deviation over seeds. Logistic regression "
                     "is fitted deterministically, so its sd is expected to be zero; the MLP seed sets its initial "
                     "weights and early-stopping split.\n\n" % (", ".join(map(str, seeds)), seeds[0] if seeds else "?")
                     + seed_table(results) + "\n")
    keys = list(results["runs"])
    conds = [c for c in COND_LABEL if c in results["runs"][keys[0]]]
    x = np.arange(len(keys))
    w = 0.8 / len(conds)
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.9), 4.2))
    colors = ["#2a6f97", "#e09f3e", "#9e2a2b", "#5a7d2a"]
    for i, c in enumerate(conds):
        ax.bar(x + (i - (len(conds) - 1) / 2) * w, [100 * results["runs"][k][c]["eer"] for k in keys], w,
               label=COND_LABEL[c], color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("EER (%), lower is better")
    ax.set_title("ASVspoof 2019 LA eval sample under channel conditions")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_eer_by_condition.png"), dpi=160)
    plt.close(fig)

    best = min(keys, key=lambda k: results["runs"][k]["clean"]["eer"])
    attacks = sorted(results["runs"][best]["clean"]["per_attack_eer"])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    xa = np.arange(len(attacks))
    for i, c in enumerate(conds):
        ax.bar(xa + (i - (len(conds) - 1) / 2) * w, [100 * results["runs"][best][c]["per_attack_eer"][a] for a in attacks], w,
               label=COND_LABEL[c], color=colors[i])
    ax.set_xticks(xa)
    ax.set_xticklabels(attacks)
    ax.set_ylabel("EER (%)")
    ax.set_title("Per attack, best clean model: " + best)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_eer_per_attack.png"), dpi=160)
    plt.close(fig)
