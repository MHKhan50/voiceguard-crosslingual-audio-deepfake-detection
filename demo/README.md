---
title: VoiceGuard - Fake Voice Check
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
short_description: Research demo - is this voice clip AI-generated?
---

# VoiceGuard: Fake Voice Check

Browser demo of the VoiceGuard speech deepfake detector. The clip is decoded, resampled to 16 kHz and scored on
the user's device with ONNX Runtime Web; no audio is uploaded.

| Part | Detail |
|---|---|
| Features | WavLM Base+ up to layer 6, mean-pooled over the first 4 s |
| Classifier | One-hidden-layer MLP (256 units), seed 42, trained on the ASVspoof 2019 LA train set (25,380 clips) |
| Calibration | Affine map to a log-likelihood ratio, fitted on the ASVspoof 2019 LA dev sample with equal class weights |
| Browser model | One ONNX graph ({precision}, {model_mb} MB): waveform in, log-likelihood ratio out |
| Experiment results | EER 1.8% clean, 2.0% simulated phone line, 9.3% phone line with noise at 15 dB SNR (ASVspoof 2019 LA eval sample, 9,800 clips) |
| Browser check | On {n_check} eval clips: EER {eer_demo}% for this model and {eer_batched}% for the experiment pipeline; same verdict on {agree}% of clips (float32). Details in `parity.json` |

The experiments computed features for batches of clips in float16 on a GPU; the browser computes one clip at a
time in {precision} on the CPU, which is why the two can differ slightly.

## Limitations

- Trained on English speech and on spoofing systems from 2019. Recent voice-cloning tools, other languages, music,
  strong noise and real phone networks can be misjudged.
- Calibration fitted on clean audio is over-confident on noisy phone audio (see the project results).
- Not a forensic tool. Do not use the result alone to make a decision about a person, a payment or a call.

## Credits

- ASVspoof 2019 database (Wang et al., 2020), University of Edinburgh, Open Data Commons Attribution License:
  https://doi.org/10.7488/ds/2555
- WavLM Base+ (Chen et al., 2022), Microsoft, used under the licence in the
  [UniSpeech repository](https://github.com/microsoft/UniSpeech).

Code, experiments and results: https://github.com/MHKhan50/voiceguard-crosslingual-audio-deepfake-detection
