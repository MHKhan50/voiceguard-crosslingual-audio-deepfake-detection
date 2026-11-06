"""Read ASVspoof 2019 LA countermeasure protocol files."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Utterance:
    speaker: str
    utt_id: str
    attack: str  # "bonafide" for real speech, otherwise A01..A19
    label: int  # 1 = spoof, 0 = bona fide


def parse_line(line: str) -> Optional[Utterance]:
    parts = line.split()
    if len(parts) != 5:
        return None
    speaker, utt_id, _env, attack, key = parts
    if key not in ("bonafide", "spoof"):
        return None
    is_spoof = key == "spoof"
    return Utterance(speaker, utt_id, attack if is_spoof else "bonafide", int(is_spoof))


def read_protocol(path: str) -> List[Utterance]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            u = parse_line(line)
            if u is not None:
                out.append(u)
    return out


def stratified_sample(items: List[Utterance], per_group: Dict[str, int], default: int,
                      seed: int = 42) -> List[Utterance]:
    """Take up to N items per attack (or bona fide) group, reproducibly."""
    groups: Dict[str, List[Utterance]] = {}
    for u in items:
        groups.setdefault(u.attack, []).append(u)
    rng = random.Random(seed)
    out: List[Utterance] = []
    for name in sorted(groups):
        g = sorted(groups[name], key=lambda u: u.utt_id)
        n = per_group.get(name, default)
        out.extend(g if len(g) <= n else rng.sample(g, n))
    return sorted(out, key=lambda u: u.utt_id)


def find_file(root: str, name: str) -> str:
    for dirpath, _dirs, files in os.walk(root):
        if name in files:
            return os.path.join(dirpath, name)
    raise FileNotFoundError(name + " not found under " + root)


def find_dir(root: str, name: str) -> str:
    for dirpath, dirs, _files in os.walk(root):
        if name in dirs and os.path.isdir(os.path.join(dirpath, name, "flac")):
            return os.path.join(dirpath, name, "flac")
    raise FileNotFoundError(name + "/flac not found under " + root)
