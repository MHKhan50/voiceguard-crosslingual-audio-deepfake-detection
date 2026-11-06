# Dataset, licensing, and compute plan

## MLAAD
Dataset: https://huggingface.co/datasets/mueller91/MLAAD
Paper: https://arxiv.org/abs/2401.09512
Maintainer overview: https://www.resemble.ai/resources/mlaad-the-multi-language-audio-anti-spoofing-dataset

The Hugging Face card identifies CC BY-NC 4.0. This is non-commercial research data. Review dataset and model terms before publishing weights or offering a commercial service. Its 54-language figure must not be interpreted as equal real-human speech coverage for all languages.

## ASVspoof 2021 DF
Official challenge: https://www.asvspoof.org/index2021.html
Zenodo: https://zenodo.org/records/4835108
The DF evaluation archive is approximately 34.5 GB. Avoid full download with current limited free disk.

## ASVspoof 5
Hugging Face: https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof5
Listed size is approximately 84 GB. Do not download the full dataset locally.

## Hardware-aware plan
Local machine: MacBook with Apple M4 and 16 GB unified memory, limited free disk. Heavy feature extraction runs on a Kaggle GPU.
- Begin with a streamed/subsampled corpus.
- Limit duration and batch size.
- Cache only selected files.
- Start with frozen embeddings and a small classifier.
- Do not begin by fine-tuning a multi-billion-parameter model.
- Keep raw audio out of Git; exclude virtual environments and model caches.
- Check free disk before each data acquisition.

## Release rights
A Hugging Face Space is suitable for a research demonstration, but dataset licence restrictions can affect data and derived weights. Do not assume non-commercial data permits commercial deployment. Model card must disclose data, licence, intended use, limitations, and evaluation domains.
