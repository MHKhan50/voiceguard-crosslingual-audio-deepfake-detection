"""Feature extraction on GPU (Kaggle). Needs torch, torchaudio, transformers, soundfile."""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModel

from .channel import apply_condition

MAX_SEC = 4.0
SR = 16000
LAYERS = (3, 6, 9, 12)


def load_clip(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        raise ValueError("expected 16 kHz audio: " + path)
    return x[: int(MAX_SEC * SR)]


class _Clips(torch.utils.data.Dataset):
    def __init__(self, paths: List[str], condition: str):
        self.paths, self.condition = paths, condition

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = apply_condition(load_clip(self.paths[i]), self.condition, seed=i)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def _collate(batch):
    n = max(len(b) for b in batch)
    wav = torch.zeros(len(batch), n)
    mask = torch.zeros(len(batch), n, dtype=torch.long)
    for i, b in enumerate(batch):
        wav[i, : len(b)] = b
        mask[i, : len(b)] = 1
    return wav, mask


class Extractor:
    def __init__(self, model_name: str = "microsoft/wavlm-base-plus", device: str = "cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        if device == "cuda":
            self.model.half()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=SR, n_mfcc=40, melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 64}
        ).to(device)

    @torch.no_grad()
    def run(self, paths: List[str], condition: str, batch_size: int = 48,
            workers: int = 4) -> Dict[str, np.ndarray]:
        dl = torch.utils.data.DataLoader(_Clips(paths, condition), batch_size=batch_size,
                                         num_workers=workers, collate_fn=_collate)
        feats = {"wavlm_l%d" % l: [] for l in LAYERS}
        feats["mfcc"] = []
        t0, done = time.time(), 0
        for wav, mask in dl:
            wav, mask = wav.to(self.device), mask.to(self.device)
            # normalise each clip over its real samples (as the WavLM feature extractor does)
            m = mask.float()
            mean = (wav * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
            var = (((wav - mean) * m) ** 2).sum(1, keepdim=True) / m.sum(1, keepdim=True)
            norm = (wav - mean) / torch.sqrt(var + 1e-7) * m
            inp = norm.half() if self.device == "cuda" else norm
            out = self.model(inp, attention_mask=mask, output_hidden_states=True)
            frames = out.hidden_states[0].shape[1]
            lens = (mask.sum(1).float() / 320).floor().clamp(1, frames).long()
            fmask = (torch.arange(frames, device=self.device)[None, :] < lens[:, None]).float()
            for l in LAYERS:
                h = out.hidden_states[l].float()
                pooled = (h * fmask[..., None]).sum(1) / fmask.sum(1, keepdim=True)
                pooled = torch.nan_to_num(pooled)
                feats["wavlm_l%d" % l].append(pooled.cpu().numpy().astype(np.float16))
            c = self.mfcc(wav)  # (B, 40, T)
            cl = (mask.sum(1).float() / 160).floor().clamp(1, c.shape[-1]).long()
            cm = (torch.arange(c.shape[-1], device=self.device)[None, :] < cl[:, None]).float()[:, None, :]
            mu = (c * cm).sum(-1) / cm.sum(-1)
            sd = torch.sqrt((((c - mu[..., None]) * cm) ** 2).sum(-1) / cm.sum(-1) + 1e-6)
            feats["mfcc"].append(torch.cat([mu, sd], 1).cpu().numpy().astype(np.float32))
            done += len(wav)
            if done % (batch_size * 40) < batch_size:
                rate = done / (time.time() - t0)
                print("  %s %d/%d clips, %.0f clips/s, ETA %.1f min" % (
                    condition, done, len(paths), rate, (len(paths) - done) / rate / 60), flush=True)
        return {k: np.concatenate(v) for k, v in feats.items()}
