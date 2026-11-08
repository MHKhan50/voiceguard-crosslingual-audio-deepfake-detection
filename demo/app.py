"""VoiceGuard research demo: is this voice clip AI-generated? Runs on CPU."""
from __future__ import annotations

import os

import gradio as gr
import joblib
import torch
from transformers import AutoModel

import core
from voiceguard.channel import telephone

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = joblib.load(os.path.join(HERE, "model", "voiceguard_wavlm_l6_mlp.joblib"))
RECIPE = BUNDLE["card"]["feature_recipe"]
torch.set_num_threads(max(1, os.cpu_count() or 1))
WAVLM = AutoModel.from_pretrained(RECIPE["wavlm_model"]).eval()

ABOUT = """
**Research demo, not a forensic tool.** The detector is frozen WavLM Base+ (layer 6) with a small classifier,
trained on the ASVspoof 2019 LA training set (English speech, spoofing systems from 2019) and calibrated on its
development set. On the ASVspoof 2019 LA evaluation sample it reached an equal error rate of 1.8% on clean audio,
2.0% over a simulated phone line and 9.3% with added noise. Newer voice-cloning tools, other languages, music,
heavy noise or real phone networks can fool it. Never use the result alone to make a decision about a person,
a payment or a call.

Code, results and limitations: [github.com/MHKhan50/voiceguard-crosslingual-audio-deepfake-detection](https://github.com/MHKhan50/voiceguard-crosslingual-audio-deepfake-detection)
"""


def describe(name: str, r: dict) -> str:
    return "| %s | %s | %.1f%% | %+.2f |" % (name, core.verdict(r["p_fake"]), 100 * r["p_fake"], r["llr"])


def analyse(audio, phone_check: bool) -> str:
    if audio is None:
        return "Please upload or record a clip first."
    sample_rate, data = audio
    x, error = core.check_clip(core.to_mono_16k(sample_rate, data), RECIPE["max_seconds"])
    if error:
        return error
    layer = RECIPE["hidden_state_index"]
    clean = core.score(BUNDLE, core.wavlm_features(WAVLM, x, layer))
    phone = core.score(BUNDLE, core.wavlm_features(WAVLM, telephone(x), layer)) if phone_check else None
    rows = ["| Check | Verdict | Probability of AI voice | Log-likelihood ratio |", "|---|---|---|---|",
            describe("As uploaded", clean)]
    if phone is not None:
        rows.append(describe("Simulated phone line", phone))
    used = len(x) / core.SR
    return ("### %s\n\n" % core.overall(clean, phone) + "\n".join(rows) +
            "\n\nThe model looked at the first %.1f s. Probabilities assume real and fake clips are equally likely "
            "before listening; a positive log-likelihood ratio points to an AI voice, a negative one to a real voice."
            % used)


with gr.Blocks(title="VoiceGuard: Fake Voice Check") as demo:
    gr.Markdown("# VoiceGuard: Fake Voice Check\nUpload or record 1 to 10 seconds of speech. "
                "The clip is processed in memory and is not stored.")
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Voice clip")
            phone_check = gr.Checkbox(value=True, label="Also check it over a simulated phone line (8 kHz, G.711)")
            button = gr.Button("Check voice", variant="primary")
        with gr.Column():
            result = gr.Markdown()
    gr.Markdown(ABOUT)
    button.click(analyse, inputs=[audio, phone_check], outputs=result)

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
