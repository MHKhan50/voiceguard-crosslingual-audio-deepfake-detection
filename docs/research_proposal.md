# Research proposal

## Title
**Beyond Clean Audio: Calibrated Cross-Lingual Detection of AI-Generated Speech under Telephony Degradation and Unseen Voice Generators**

## Abstract
The increasing availability of speech synthesis and voice cloning creates risks for telephone-based impersonation and social engineering. Detectors that perform well on conventional test splits may fail when the speech generator, language, recording channel, or codec differs from training conditions. This project proposes a resource-conscious study of robust speech deepfake detection under distribution shift. It will compare frozen self-supervised speech representations and lightweight neural classifiers with established anti-spoofing baselines. Evaluation will use speaker- and generator-disjoint splits, controlled telephone-channel transformations, cross-corpus testing, and calibration analysis. A selective prediction mechanism will allow abstention when evidence is insufficient. The intended outcome is a reproducible benchmark, a compact research prototype, and a carefully scoped public demonstration. Model output will not be treated as proof of fraud or authenticity.

## UK motivation
UK Finance's 2026 Annual Fraud Report reports approximately £1.28 billion lost to payment fraud in 2025. Telecommunications accounted for 17% of APP fraud cases and 28% of APP losses. These figures establish the broader importance of telecom-enabled fraud; they do not establish that voice deepfakes caused those losses.

## Research-gap hypothesis
Recent work already studies multilingual anti-spoofing and robustness failures. The narrower hypothesis to investigate is: a reproducible evaluation of compact speech detectors that jointly measures generator-disjoint generalisation, telephone-channel degradation, language/domain shift, and calibrated abstention under resource constraints. A literature matrix must verify whether this combination is sufficiently distinct.

## Research questions
RQ1: What is the generalisation gap between seen-generator and unseen-generator evaluation?
RQ2: How do narrowband filtering, codecs, noise, and reverberation change error rates?
RQ3: How stable are results across languages and speaker groups represented in selected corpora?
RQ4: Can calibration and selective abstention reduce high-confidence errors under shift?
RQ5: Which compact architecture offers the best accuracy, calibration, latency, and memory trade-off?

## Method
1. Audit licences, provenance, labels, speaker IDs, generator IDs, and recording conditions.
2. Construct leakage-resistant group splits.
3. Establish acoustic and pretrained-embedding baselines.
4. Apply documented telephone-channel transformations to held-out audio.
5. Evaluate generator-held-out and cross-corpus conditions.
6. Calibrate on validation data only; predeclare abstention thresholds.
7. Report uncertainty intervals across seeds and subgroup results where sample sizes permit.
8. Release code, split manifests, dataset documentation, and model card.

## Metrics
Equal Error Rate (EER), ROC-AUC, PR-AUC, false-positive rate at predeclared operating points, Expected Calibration Error, Brier score, risk-coverage curves, inference latency, peak memory, and model size. Accuracy alone is insufficient.

## Integrity
No claim of novelty, superiority or real-world fraud prevention is made before results exist. The demo is a research tool, not a forensic detector.
