"""Publish the demo to a Hugging Face Space (Gradio, free CPU).

Run from the repository root after logging in to Hugging Face (a token with write access):

    python demo/publish_space.py

The Space gets the demo files, the four VoiceGuard modules the demo imports and the model files in demo/model/
(the model is kept out of git).
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
SPACE_NAME = "voiceguard-fake-voice-check"
MODULES = ["__init__.py", "channel.py", "metrics.py", "pipeline.py"]
MODEL_FILES = ["voiceguard_wavlm_l6_mlp.joblib", "model_card.json"]


def main() -> None:
    missing = [f for f in MODEL_FILES if not (DEMO / "model" / f).exists()]
    if missing:
        sys.exit("Missing in demo/model/: " + ", ".join(missing))
    api = HfApi()
    repo_id = "%s/%s" % (api.whoami()["name"], SPACE_NAME)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in ["app.py", "core.py", "requirements.txt", "README.md"]:
            shutil.copy(DEMO / name, stage / name)
        (stage / "voiceguard").mkdir()
        for name in MODULES:
            shutil.copy(ROOT / "src" / "voiceguard" / name, stage / "voiceguard" / name)
        (stage / "model").mkdir()
        for name in MODEL_FILES:
            shutil.copy(DEMO / "model" / name, stage / "model" / name)
        api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(stage),
                          commit_message="Publish VoiceGuard demo")
    print("published https://huggingface.co/spaces/" + repo_id)


if __name__ == "__main__":
    main()
