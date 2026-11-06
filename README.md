# VoiceGuard: speech deepfake detection under phone-line conditions

**Working title:** Beyond Clean Audio: Calibrated Cross-Lingual Detection of AI-Generated Speech under Telephony Degradation and Unseen Voice Generators.

**Status:** v0.2.0 adds the first runnable baseline (code, tests and a Kaggle notebook). Results will be added only from completed runs. This is not yet a validated detector or a paper.

## Problem
AI-generated and cloned speech can support impersonation and social-engineering attacks. Detectors can fail when the generator, language, recording channel, codec, or noise differs from training data.

## Proposed application
A Hugging Face demo where a user uploads a short audio clip and receives a risk estimate (likely bona-fide / likely synthetic or manipulated / uncertain), calibrated confidence, audio-quality warnings, and limitations. It must not claim forensic certainty or replace bank or law-enforcement checks.

## Proposed contribution
Study generator-disjoint generalisation, cross-language variation, telephone-channel degradation, calibration and selective abstention, and resource-efficient inference. Novelty is not established: a systematic literature review and data audit must precede any novelty claim.

## Research questions
1. How much does performance degrade on unseen voice generators?
2. How do telephone filtering, codecs, background noise, and reverberation affect errors?
3. How stable are results across languages and speaker populations represented in the data?
4. Can calibration and abstention reduce high-confidence errors under distribution shift?
5. What accuracy/latency/memory trade-off is achievable with a compact model?

## Planned technical route
Start with frozen WavLM or wav2vec 2.0 representations plus a lightweight neural classifier. Compare with acoustic-feature baselines and, if feasible, AASIST. Use group-aware splits by speaker, corpus, and generator. Calibrate only on a separate validation set.

## Planned deliverables
Runnable training/evaluation code; documented dataset and split manifests; robustness benchmark; model card; Hugging Face demo; research manuscript with statistical analysis and limitations.

## First baseline (v0.2.0)
A pilot on one well-documented dataset before the cross-lingual stage.

| Part | Choice |
|---|---|
| Data | ASVspoof 2019 LA (Kaggle input `awsaf49/asvpoof-2019-dataset`; audio is not copied into this repo). Its train, dev and eval sets have different speakers. |
| Unseen generators | Training contains attacks A01-A06. Eval contains A07-A19; A16 and A19 reuse the A04 and A06 algorithms, so they are reported as known and left out of the unseen-attack EER. |
| Sample | All 25,380 training clips; dev 1,000 bona fide + up to 170 per attack (calibration only); eval up to 600 clips per attack + 2,000 bona fide, fixed seed 42. Clip IDs are saved in `manifest.json`. |
| Channel conditions | Clean; simulated phone line (8 kHz, 300-3400 Hz band, G.711-style mu-law); phone line with white noise at 15 dB SNR (`src/voiceguard/channel.py`). Training uses clean audio only. |
| Features | Frozen WavLM Base+ hidden states (layers 3, 6, 9, 12, mean-pooled over the first 4 s); MFCC mean and standard deviation as an acoustic baseline. |
| Classifiers | Logistic regression (balanced classes) and a one-hidden-layer MLP; Platt calibration fitted on dev only. |
| Metrics | EER with bootstrap 95% CI, EER per attack and on unseen attacks, ROC-AUC, ECE (before and after calibration), Brier score, error at 100/90/80/70/50% coverage (abstention). A majority-class baseline is included. |

### Run on Kaggle
1. New notebook, *File > Import notebook*, paste the GitHub link to `notebooks/voiceguard_kaggle_baseline.ipynb`.
2. Settings: GPU T4 x2 or P100, Internet on; *Add Input*: `awsaf49/asvpoof-2019-dataset`.
3. Run all. The pilot cell checks the whole pipeline on a small sample first; the full run prints progress and an ETA.
4. Download `voiceguard_results.zip` from the Output panel.

### Run the tests locally (no GPU, no audio)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```
The tests cover protocol parsing, the phone-line simulation, the metrics, the classifiers and a full mock run of the Kaggle runner with a fake feature extractor.

### Results
To be filled in from the Kaggle run.

### Limitations of this baseline
- One English corpus; the cross-lingual question needs MLAAD or another multilingual set after a licence and content audit.
- The phone line is simulated, not recorded over a real network.
- One seed per classifier in this first run.

## Hardware
Local machine: MacBook with Apple M4 and 16 GB unified memory. Feature extraction runs on a Kaggle GPU; local work is limited to tests and analysis.
