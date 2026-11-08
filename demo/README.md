---
title: VoiceGuard - Fake Voice Check
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Research demo - is this voice clip AI-generated?
---

# VoiceGuard: Fake Voice Check

Research demo of the VoiceGuard project: upload or record a short voice clip and the detector estimates whether it
is AI-generated, as uploaded and again after a simulated phone line (8 kHz, 300-3400 Hz, G.711 mu-law).

| Part | Detail |
|---|---|
| Features | Frozen [WavLM Base+](https://huggingface.co/microsoft/wavlm-base-plus), hidden state 6, mean-pooled over the first 4 s |
| Classifier | One-hidden-layer MLP (256 units), seed 42 |
| Calibration | Affine map to a log-likelihood ratio, fitted on the ASVspoof 2019 LA dev sample with equal class weights |
| Training data | ASVspoof 2019 LA train set (25,380 clips, clean audio) |
| Reported results | EER 1.8% clean, 2.0% simulated phone line, 9.3% phone line with noise at 15 dB SNR (ASVspoof 2019 LA eval sample, 9,800 clips) |
| Verdicts | Probability of an AI voice at an even prior: at least 95% "Likely AI-generated", at most 5% "Likely real", otherwise "Uncertain"; also "Uncertain" when the clean and phone-line checks disagree |

## Limitations

- Trained on English speech and on spoofing systems from 2019. Recent voice-cloning tools, other languages, music,
  strong noise and real phone networks were not part of training and can be misjudged.
- The phone line is simulated. Calibration fitted on clean audio is known to be over-confident on noisy phone audio
  (see the project results).
- Features are computed in float32 on CPU here and were computed in float16 on GPU in the experiments, so scores can
  differ slightly from the paper numbers.
- Not a forensic tool. Do not use the result alone to make a decision about a person, a payment or a call.

## Licences and credits

- ASVspoof 2019 database (Wang et al., 2020), University of Edinburgh, released under the Open Data Commons
  Attribution License: https://doi.org/10.7488/ds/2555
- WavLM Base+ (Chen et al., 2022), Microsoft, used under the licence in the
  [UniSpeech repository](https://github.com/microsoft/UniSpeech).

Code, experiments and results: https://github.com/MHKhan50/voiceguard-crosslingual-audio-deepfake-detection
