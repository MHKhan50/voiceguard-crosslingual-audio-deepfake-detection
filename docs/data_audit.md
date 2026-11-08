# Data audit for the cross-lingual stage (8 Nov 2026)

Purpose: decide which data can be used for the cross-lingual and real-channel experiments, under which licence, and
which pitfalls the experimental protocol must handle. Every fact below comes from the dataset card, paper or official
repository listed under Sources; points that could not be checked are marked **not verified**.

## Summary

| | MLAAD v9 | M-AILABS | RTCFake |
|---|---|---|---|
| What it gives | Spoofed speech only | Bona fide speech only | Bona fide and spoofed, offline and after real transmission |
| Size | 687.4 h, 140 TTS models | about 1,000 h | about 600 h, 317,076 utterances |
| Languages | 51 (spoof) | 8 (9 locale tags) | English, Chinese |
| Licence | CC BY-NC 4.0, non-commercial academic research only; gated on Hugging Face | Permissive BSD-style, commercial use allowed; Ukrainian audio "for machine learning purposes only" | Listed as "other" on Hugging Face; terms **not verified** |
| Official splits | None | None | Train / dev / eval with unseen generators, platforms and noise |
| Speakers | Reference speaker field from v8; bona fide speakers come from M-AILABS | Few speakers per language | 307 (train 105, dev 21, eval 181); disjointness **not verified** |

## 1. MLAAD (Multi-Language Audio Anti-Spoofing Dataset), version 9

- **Content:** synthetic speech only, made with 140 TTS models in 51 languages (687.4 hours), released 17 Jan 2026.
  For each language and model, 1,000 source utterances are taken from M-AILABS; for languages that M-AILABS does not
  cover, the English texts are machine-translated first.
- **Languages (51):** Amharic, Arabic, Bangla, Bulgarian, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian,
  Finnish, French, German, Greek, Hausa, Hebrew, Hindi, Hungarian, Igbo, Irish, Italian, Japanese, Javanese, Kannada,
  Korean, Latvian, Lithuanian, Malay, Malayalam, Maltese, Marathi, Persian, Polish, Portuguese, Romanian, Russian,
  Sinhala, Slovak, Slovenian, Spanish, Swahili, Swedish, Tamil, Thai, Turkish, Turkmen, Ukrainian, Urdu, Vietnamese,
  Yoruba.
- **Labels and metadata:** one `meta.csv` per language/model folder with `path`, `original_file`, `language`,
  `is_original_language`, `duration`, `training_data`, `model_name`, `architecture`, `transcript` and, from v8,
  `reference_speaker`. All MLAAD clips are spoofs; `original_file` points to the M-AILABS source clip.
- **Bona fide audio:** not included. The authors ask users to add the matching real clips from M-AILABS.
- **Licence:** from v8 on, CC BY-NC 4.0; the card says the dataset is published strictly for non-commercial academic
  research and that commercial use is not permitted. Access on Hugging Face requires accepting the gate (sharing
  contact details).
- **Splits:** the paper and card give no train/dev/eval protocol; VoiceGuard must define its own.

## 2. M-AILABS Speech Dataset

- **Content:** read speech, mostly LibriVox audiobooks, with transcripts; 999 h 32 min in total.
- **Languages and hours:** German 237 h, English UK 46 h, English US 102 h, Spanish 109 h, Italian 128 h,
  Ukrainian 87 h, Russian 47 h, French 191 h, Polish 54 h (rounded). Folders are split into female, male and mixed
  voices.
- **Speakers:** the official page gives no counts. One secondary source (Montreal Forced Aligner corpus pages) lists
  3 speakers for Russian and 6 for Ukrainian, so some languages have very few speakers.
- **Licence:** BSD-style; redistribution and use, including commercial use, are allowed if the copyright notice and
  disclaimer are kept. Ukrainian audio was provided by Nash Format or Gwara Media for machine learning purposes only,
  with details in each `info.txt`.
- **Download:** one `.tgz` per language from the host given on the official GitHub page (the GitHub repository itself
  holds no audio). Whether that host is reachable from a Kaggle notebook is **not verified**.

## 3. RTCFake (ACL Findings 2026)

- **Content:** about 600 hours of paired offline and online speech: each clip exists before and after being sent
  through a real communication platform. Real speech comes from LibriHeavy (English) and Chinese-LiPS (Chinese).
- **Generators:** 7 TTS (F5-TTS, OpenAudio-S1, VoxCPM, LLaSA, IndexTTS2, Doubao, SparkTTS) and 3 voice conversion
  systems (CosyVoice, SeedVC, ChatterboxVC).
- **Platforms:** Zoom and QQ in training; WeChat added in dev; DingTalk, Lark, VooV and Telegram only in eval
  (inferred from the paper's split table).
- **Counts:** 54,887 real and 262,189 fake utterances overall; the evaluation set also has office, café, echo and
  environmental noise.
- **Licence:** the Hugging Face card lists the licence as "other"; the actual terms could not be read from here
  (**not verified**). The English/Chinese share, native sampling rate and speaker disjointness between splits are not
  stated in the paper.

## Consequences for VoiceGuard

1. **Cross-lingual test only where both classes exist.** MLAAD has spoofs in 51 languages but M-AILABS has real
   speech in only 8 (English, French, German, Italian, Polish, Russian, Spanish, Ukrainian). In the other 43 languages
   every clip would be fake, so a detector could score well just by recognising the language. The cross-lingual EER
   must therefore be computed only on these 8 languages, or with another licensed source of real speech per language.
   Urdu, for example, has MLAAD spoofs but no matching bona fide set in either dataset.
2. **Hold out by generator and by language.** Proposed protocol: train on a subset of languages and TTS models, test on
   unseen languages and unseen models (using `language`, `model_name` and `architecture` from `meta.csv`), with the
   real clips drawn from the same M-AILABS languages.
3. **Speakers are a confound.** Real clips come from a few audiobook readers per language, and TTS outputs may copy a
   reference speaker. Speaker-disjoint splits may be impossible for some languages; results must report this.
4. **Non-commercial use.** Anything trained on MLAAD inherits CC BY-NC 4.0: the paper, code and a research demo are
   fine, but a model trained on MLAAD must not be used in a paid or company product (including a Masmatix service).
   The Hugging Face demo must be labelled as a non-commercial research demo.
5. **RTCFake is the closest match to the phone-line question.** Its real-platform transmission can test whether the
   simulated phone line (`channel.py`) predicts real degradation. It covers English and Chinese only, so it does not
   replace MLAAD for the cross-lingual question. Use it only after the licence terms on its card are read and accepted.

## Before any download

- [ ] Read and accept the MLAAD gate on Hugging Face with your own account.
- [ ] Open the RTCFake card on Hugging Face and record its licence terms here.
- [ ] Check that the M-AILABS download host can be reached from Kaggle (Internet on); otherwise find a mirror whose
      licence notice is kept.
- [ ] Keep each dataset's licence file next to any derived features.

## Sources

- MLAAD dataset card: https://huggingface.co/datasets/mueller91/MLAAD
- Müller et al., MLAAD paper, v9: https://arxiv.org/abs/2401.09512v9
- M-AILABS official page and licence: https://github.com/imdatceleste/m-ailabs-dataset
- Montreal Forced Aligner corpus pages (speaker counts): https://mfa-models.readthedocs.io/en/latest/corpus/Russian/M-AILABS%20Russian.html
- RTCFake paper (Xue et al., Findings of ACL 2026, pp. 5763-5775): https://aclanthology.org/2026.findings-acl.285/
- RTCFake dataset card: https://huggingface.co/datasets/JunXueTech/RTCFake
