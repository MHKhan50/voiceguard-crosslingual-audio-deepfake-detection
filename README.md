# VoiceGuard: speech deepfake detection under phone-line conditions

**Working title:** Beyond Clean Audio: Calibrated Cross-Lingual Detection of AI-Generated Speech under Telephony Degradation and Unseen Voice Generators.

**Status:** v0.2.0 adds the first runnable baseline (code, tests and a Kaggle notebook). Results will be added only from completed runs. This is not yet a validated detector or a paper.

## Problem
AI-generated and cloned speech can support impersonation and social-engineering attacks. Detectors can fail when the generator, language, recording channel, codec, or noise differs from training data.

## Proposed application
A Hugging Face demo where a user uploads a short audio clip and receives a risk estimate (likely bona-fide / likely synthetic or manipulated / uncertain), calibrated confidence, audio-quality warnings, and limitations. It must not claim forensic certainty or replace bank or law-enforcement checks.

## Demo
A research demo of the layer 6 + MLP model (upload or record a clip; it is also checked over a simulated phone line)
runs as a Hugging Face Space. Its source is in [`demo/`](demo/); the model file is stored in the Space, not in git.

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
| Classifiers | Logistic regression (balanced classes) and a one-hidden-layer MLP. v0.2: Platt calibration fitted on dev only. v0.3: an affine map to log-likelihood ratios fitted on dev with equal class weights, then a probability at the stated spoof share of the eval sample (Platt kept for comparison). |
| Metrics | EER with bootstrap 95% CI, Cllr and minCllr (v0.3; independent of the class mix), EER per attack and on unseen attacks, ROC-AUC, ECE (before and after calibration), Brier score, error at 100/90/80/70/50% coverage (abstention). A majority-class baseline is included. |

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

### Results (Kaggle runs, 7-8 Nov 2026, GPU T4 x2; calibration and seed results from the v0.3 reruns)
Sample: 25,380 training clips, 2,020 dev clips, 9,800 eval clips (2,000 bona fide, 600 per attack), each eval clip in
three conditions. Full tables: [`results/tables.md`](results/tables.md); all numbers: [`results/results.json`](results/results.json);
clip IDs: [`results/manifest.json`](results/manifest.json).

EER % [bootstrap 95% CI], lower is better:

| Features + classifier | Clean | Phone line | Phone + noise 15 dB |
|---|---|---|---|
| MFCC + logistic regression | 18.1 [17.3, 18.9] | 50.6 [49.6, 51.9] | 38.6 [37.7, 39.6] |
| MFCC + MLP | 16.3 [15.5, 17.1] | 33.8 [32.7, 34.9] | 50.0 [49.6, 50.4] |
| WavLM layer 3 + MLP | 1.9 [1.6, 2.4] | 12.0 [11.2, 12.8] | 35.2 [34.2, 36.3] |
| WavLM layer 6 + logistic regression | 2.7 [2.2, 3.2] | 3.3 [2.8, 3.6] | 9.8 [9.1, 10.4] |
| **WavLM layer 6 + MLP** | **1.8 [1.5, 2.2]** | **2.0 [1.7, 2.4]** | **9.3 [8.6, 9.9]** |
| WavLM layer 9 + MLP | 4.1 [3.7, 4.5] | 6.4 [5.8, 7.1] | 11.7 [10.9, 12.5] |
| WavLM layer 12 + logistic regression | 5.1 [4.7, 5.5] | 9.6 [8.9, 10.3] | 33.5 [32.5, 34.5] |

Three classifier seeds (42, 43, 44; same clips and features). The table above is seed 42. EER % mean (sd):

| Features + classifier | Clean | Phone line | Phone + noise 15 dB |
|---|---|---|---|
| MFCC + MLP | 15.8 (0.7) | 37.1 (3.6) | 49.5 (0.7) |
| WavLM layer 3 + MLP | 2.0 (0.6) | 9.4 (2.4) | 29.4 (5.6) |
| **WavLM layer 6 + MLP** | **1.7 (0.1)** | **2.2 (0.2)** | **8.0 (1.2)** |
| WavLM layer 9 + MLP | 4.0 (0.2) | 6.3 (0.8) | 9.8 (2.3) |
| WavLM layer 12 + MLP | 5.6 (0.2) | 6.8 (0.5) | 30.1 (8.0) |

Logistic regression is fitted deterministically, so its results are identical across seeds (sd 0). Three seeds give
only a rough spread; per-seed values are in `results/results.json` (`seed_summary`).

What the run shows:
- **Single MLP runs are unreliable under shift.** On clean audio the MLP seeds agree closely, but with noise the EER
  of layer 12 + MLP ranged from 20.9% to 35.9% across seeds and layer 3 + MLP from 24.1% to 35.2%. Layer 6 + MLP
  stayed between 7.1% and 9.3%; the seed-42 value reported in v0.2 (9.3%) was the highest of the three.
- **The layer matters more than the classifier under shift.** Layer 3 and layer 6 are close on clean audio
  (1.9% and 1.8% with the MLP), but on the phone line layer 3 rises to 12.0% while layer 6 stays at 2.0%.
  With noise, layers 3 and 12 reach 25-36% while layers 6 and 9 stay at 9-13%.
- **The phone line alone breaks the MFCC baseline** (EER 33.8-50.6%, close to chance), while the best WavLM
  features barely move. Adding noise is the hardest condition for every model.
- **Per attack (layer 6 + MLP):** with noise, A18 rises from 3.7% to 25.6%, A17 from 1.1% to 18.4% and A19
  from 4.7% to 14.4%; most other attacks stay below 10% (`results/fig_eer_per_attack.png`). A19 reuses a training
  algorithm (A06) and is still among the hardest.
- **Abstention helps only when the model is uncertain in the right places.** For layer 6 + MLP with noise, the error
  falls from 10.7% on all clips to 5.7% on the 80% it is most confident about and 3.3% at 50%. For layer 3 + logistic
  regression with noise, the error stays between 58% and 65% at every coverage: the model is confidently wrong.
- **Calibration (v0.3 rerun, same features and classifiers, so the EERs above are unchanged).** In v0.2, Platt
  scaling fitted on dev made clean ECE worse for layer 6 + MLP (0.004 before, 0.048 after), and the dev sample's class
  mix (about half spoof, against 80% in eval) looked like the cause. v0.3 maps scores to log-likelihood ratios on dev
  with equal class weights and then applies the eval sample's spoof share. This gave lower ECE than Platt in only 16 of
  30 model and condition pairs; for layer 6 + MLP on clean audio it gave 0.034, still above the uncalibrated 0.004. So
  the class mix explains only part of the problem.
- **The larger calibration problem is the channel.** Cllr measures the cost of the scores as evidence (in bits;
  0 is perfect, 1 equals always saying "I don't know") and minCllr is the same after the best possible recalibration.
  In 12 of 30 pairs Cllr was above 1, including every WavLM model with noise except layer 6, while minCllr stayed
  much lower: for layer 12 + logistic regression with noise, Cllr 3.25 against minCllr 0.50. The scores still rank
  clips reasonably, but a calibration fitted on clean dev audio is far too confident on the noisy phone line.

| Layer 6 model | Condition | Cllr | minCllr | ECE raw | ECE Platt (dev) | ECE LLR (v0.3) |
|---|---|---|---|---|---|---|
| MLP | Clean | 0.263 | 0.066 | 0.004 | 0.048 | 0.034 |
| MLP | Phone line | 0.118 | 0.081 | 0.018 | 0.015 | 0.010 |
| MLP | Phone + noise 15 dB | 0.685 | 0.340 | 0.082 | 0.077 | 0.066 |
| Logistic regression | Phone + noise 15 dB | 0.548 | 0.345 | 0.070 | 0.062 | 0.058 |

Full calibration table for all models: [`results/tables.md`](results/tables.md#calibration).

![EER by condition](results/fig_eer_by_condition.png)
![EER per attack](results/fig_eer_per_attack.png)

### Limitations of this baseline
- One English corpus. The licence and content audit for the cross-lingual stage is in [`docs/data_audit.md`](docs/data_audit.md): MLAAD is non-commercial (CC BY-NC 4.0) and has matching real speech (M-AILABS) in only 8 of its 51 languages.
- The phone line is simulated, not recorded over a real network.
- Three classifier seeds, but one clip sample and one feature extraction; the sd does not cover sampling variation.
- Calibration is fitted on clean dev audio only; it does not transfer to the noisy phone line (see Cllr above).
- Accuracy-type errors depend on the 80% spoof share of the eval sample; EER is the main measure.

## Hardware
Local machine: MacBook with Apple M4 and 16 GB unified memory. Feature extraction runs on a Kaggle GPU; local work is limited to tests and analysis.
