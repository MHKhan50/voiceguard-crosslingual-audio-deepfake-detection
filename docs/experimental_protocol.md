# Experimental protocol — draft v0.1

## Stage 0: literature and data audit
Build a matrix recording task, dataset, languages, generators, channels, split strategy, model, metrics, limitations, code/data availability, and overlap with this proposal. Inspect:
- Maltby et al. (2026), robustness in deepfake speech detection.
- MLAAD (IJCNN 2024).
- ASVspoof 2021 and ASVspoof 5.
Novelty is judged only after the matrix is complete.

## Stage 1: dataset feasibility
- MLAAD: maintainers describe 456,000 utterances, about 1,002.9 hours, and 54 languages. Hugging Face card states CC BY-NC 4.0. It is non-commercial research data. Audit how human source speech and synthetic language coverage are constructed; 54 languages does not mean balanced native-speaker coverage in all 54.
- ASVspoof 2021 DF: Zenodo archive is about 34.5 GB; not suitable for a full local download on the current machine.
- ASVspoof 5: Hugging Face packaging lists about 84 GB; do not download in full locally.
Start with streaming or a carefully sampled subset. Record revision, sample IDs, filters, class balance, and all sampling decisions. Do not redistribute source audio without permission.

## Stage 2: split design
Group by speaker and source recording. Hold out at least one generator family. Keep test corpus and transformations out of training/model selection. Check for near-duplicate clips. Publish split manifests and checksums, not copyrighted audio.

## Stage 3: baselines
- Acoustic-feature baseline (e.g. LFCC/log-mel + logistic regression).
- Frozen WavLM or wav2vec 2.0 representation + lightweight MLP.
- AASIST only if compute and implementation permit.
Report parameter count, model size, training time, latency, and memory.

## Stage 4: channel stress tests
Apply narrowband telephone filtering, resampling/codec changes, additive noise at specified SNRs, and reverberation. Preserve originals and record transformation parameters and random seeds.

## Stage 5: calibration and abstention
Fit calibration on validation data only. Select thresholds before final testing. Report risk-coverage curves and coverage at each operating point. Compare calibration in-domain and under shift.

## Stage 6: statistics
Use multiple fixed seeds; report means and confidence intervals; use paired comparisons where appropriate; include confusion matrices and subgroup counts. Do not over-interpret small subgroups.

## Stop/go
Stop if licensing or split provenance is unclear. If language or generator metadata are inadequate, narrow the claims instead of inventing labels.
