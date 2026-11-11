# Related work and novelty check (Paper 1)

Checked on 11 November 2026. Every entry below was confirmed against the
publisher page, the ISCA/ACL archive or arXiv. Nothing here is claimed as
"first" unless the search found no prior work, and even then the wording in
the paper stays modest.

## 1. Verified references

| # | Reference | Venue (verified) | What it already shows |
|---|-----------|------------------|-----------------------|
| R1 | X. Wang et al., "ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech" | Computer Speech & Language 64, 101114, 2020. doi:10.1016/j.csl.2020.101114 | Our training and evaluation data (LA scenario). |
| R2 | S. Chen et al., "WavLM: Large-scale self-supervised pre-training for full stack speech processing" | IEEE JSTSP 16(6):1505-1518, 2022. doi:10.1109/JSTSP.2022.3188113 | Our frozen front end. |
| R3 | X. Wang and J. Yamagishi, "Investigating self-supervised front ends for speech spoofing countermeasures" | Odyssey 2022. doi:10.21437/Odyssey.2022-14 | SSL front ends (wav2vec 2.0, HuBERT) beat DSP features for spoofing CMs; fine-tuning helps. |
| R4 | H. Tak et al., "Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation" | Odyssey 2022. doi:10.21437/Odyssey.2022-16 | Fine-tuned XLS-R + AASIST + RawBoost; strong ASVspoof 2021 LA/DF results. |
| R5 | O. Pascu, A. Stan, D. Oneata, E. Oneata, H. Cucu, "Towards generalisable and calibrated audio deepfake detection with self-supervised representations" | Interspeech 2024 (arXiv 2309.05384) | Frozen SSL + logistic regression generalises and is better calibrated. Closest prior work to our setup. |
| R6 | Y. El Kheir et al., "Comprehensive layer-wise analysis of SSL models for audio deepfake detection" | Findings of NAACL 2025, pp. 4070-4082. doi:10.18653/v1/2025.findings-naacl.227 | Lower layers are the most discriminative; a few lower layers are enough. |
| R7 | N. M. Müller et al., "Does audio deepfake detection generalize?" | Interspeech 2022 (arXiv 2203.16263) | In-the-Wild dataset; ASVspoof-trained models collapse on real-world audio. |
| R8 | X. Liu et al., "ASVspoof 2021: Towards spoofed and deepfake speech detection in the wild" | IEEE/ACM TASLP 2023 (arXiv 2210.02437) | LA 2021 already includes telephony and codec channels. |
| R9 | X. Wang et al., "ASVspoof 5: Evaluation of spoofing, deepfake, and adversarial attack detection using crowdsourced speech" | IEEE TASLP, 2026 (arXiv 2601.03944) | 11 codec conditions incl. narrow-band; performance drops with neural codecs; includes a calibration study. |
| R10 | N. Brümmer and J. du Preez, "Application-independent evaluation of speaker detection" | Computer Speech & Language 20(2-3):230-275, 2006 | Cllr and minCllr. |
| R11 | Zhou and Wang, "When EER hides deployment failure" | arXiv 2606.21584 (preprint, June 2026) | Low in-domain EER can hide a large threshold-transfer failure. Cite as a preprint only. |
| R12 | Shi et al., ADD-C benchmark (codec and packet loss) | EUSIPCO 2025 (arXiv 2504.12423) | Codec/packet-loss robustness benchmark. |

Datasets and software referenced in the paper (licence checked):
ASVspoof 2019 (ODC-By). In-the-Wild: the Fraunhofer page states Apache 2.0,
while a Hugging Face mirror card says CC BY-SA 4.0; the paper will cite the
original page and not redistribute audio.

## 2. Honest novelty assessment

Already done by others (we cite, we do not claim):
- Layer-wise analysis of SSL models for deepfake detection (R6).
- Frozen SSL features + simple classifier being well calibrated (R5).
- Telephone and codec conditions in evaluation (R8, R9, R12).
- Low EER hiding threshold failures (R11).
- The group-norm padding effect in WavLM/wav2vec 2.0 Base is a known
  engineering issue (Hugging Face documentation and issue #21534).

What our paper can honestly offer:
1. A controlled, seed-repeated comparison of four WavLM layers under three
   simulated channels (narrow-band phone, phone + 15 dB noise, Opus 16 kbps),
   reporting both discrimination (EER, minCllr) and calibration (Cllr, ECE).
   Finding: layer 6 is the most stable; phone + noise, not the codec, causes
   the large drop.
2. Calibration transfer: a calibrator fitted on clean dev data is often
   miscalibrated under channel shift (Cllr > 1 in 12 of 40 model-condition
   pairs), even when minCllr stays good.
3. A measured effect of batch zero-padding on detector scores and verdicts,
   grouped by clip length (mean |dLLR| 2.98 for the shortest group), and the
   fix (equal-length batching). Framed as "documenting its impact on deepfake
   scores", not as a discovery.
4. A small, clearly informal case study of the public demo on real phone and
   WhatsApp recordings, showing that simulation does not explain the false
   alarm on a forwarded voice note.

Optional strengthening (one Kaggle GPU run): evaluate the exported layer-6
model on the public In-the-Wild set (R7) to add one real-world number.
