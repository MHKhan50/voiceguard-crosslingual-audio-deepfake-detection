# VoiceGuard Research Blueprint v0.1.0

**Working title:** Beyond Clean Audio: Calibrated Cross-Lingual Detection of AI-Generated Speech under Telephony Degradation and Unseen Voice Generators.

**Status:** This is a research blueprint, not a trained detector, validated benchmark, or completed paper.

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

## Hardware
Plan for MacBook Air M4, 16 GB unified memory, and limited free disk. Stream or sample datasets; do not download full multi-tens-of-GB corpora without checking storage.
