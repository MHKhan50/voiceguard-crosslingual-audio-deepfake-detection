"""Build the browser demo and publish it as a static Hugging Face Space (runs on Kaggle).

Inputs (Kaggle "Add Input"): the ASVspoof 2019 LA dataset and the output of the VoiceGuard notebook (it holds
results/model/voiceguard_wavlm_l6_mlp.joblib and results/manifest.json). Environment variable HF_TOKEN: a Hugging
Face token with write access. Accelerator: GPU (used for the reference features).

    python demo/build_space.py

Steps:
1. Rebuild the exported detector as one PyTorch network: waveform -> WavLM Base+ up to layer 6 -> mean pooling ->
   scaler -> MLP -> calibrated log-likelihood ratio (LLR).
2. Export it to ONNX (float32) and make an int8 copy (dynamic quantisation of the matrix products).
3. Check on a stratified sample of ASVspoof 2019 LA eval clips that the ONNX models reproduce the PyTorch network
   and how the single-clip demo scores compare with the batched scores of the experiments.
4. Publish index.html, app.js, the chosen ONNX model, parity.json and the Space card. Nothing is published if a
   check fails.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voiceguard import metrics as M  # noqa: E402
from voiceguard.protocol import find_dir, find_file, read_protocol, stratified_sample  # noqa: E402

DATA = Path(os.environ.get("VOICEGUARD_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("VOICEGUARD_WORK", "/kaggle/working/space_build"))
SPACE_NAME = "voiceguard-fake-voice-check"
MODEL_FILE = "voiceguard_wavlm_l6_mlp.joblib"
FAKE_AT, REAL_AT = 0.95, 0.05  # verdict thresholds on the probability at an even prior (same as app.js)


def sklearn_head(bundle: dict) -> dict:
    """Weights of scaler, MLP and calibrator as plain arrays: llr = a * logit(clip(MLP(x))) + b."""
    scaler, mlp = bundle["model"].steps[0][1], bundle["model"].steps[1][1]
    if mlp.activation != "relu" or len(mlp.coefs_) != 2 or mlp.out_activation_ != "logistic":
        raise ValueError("expected a one-hidden-layer ReLU MLP with a logistic output")
    lr = bundle["calibrator"].lr
    return {"mean": scaler.mean_, "scale": scaler.scale_, "w1": mlp.coefs_[0], "b1": mlp.intercepts_[0],
            "w2": mlp.coefs_[1], "b2": mlp.intercepts_[1], "a": float(lr.coef_[0, 0]),
            "b": float(lr.intercept_[0])}


def head_llr(head: dict, feats: np.ndarray) -> np.ndarray:
    """NumPy version of the head; equals calibrator.llr(model.predict_proba(feats)[:, 1]).

    Like scikit-learn on float32 features, the MLP and the probability clipping run in float32 (this matters only
    for saturated scores, where 1 - p is close to the float32 resolution).
    """
    f32 = {k: np.asarray(head[k], dtype=np.float32) for k in ("mean", "scale", "w1", "b1", "w2", "b2")}
    x = (np.asarray(feats, dtype=np.float32) - f32["mean"]) / f32["scale"]
    z = np.maximum(x @ f32["w1"] + f32["b1"], np.float32(0))
    p = (1 / (1 + np.exp(-(z @ f32["w2"] + f32["b2"])[:, 0]))).astype(np.float32)
    p = np.clip(p, np.float32(1e-6), np.float32(1 - 1e-6))
    logit = np.log(p / (1 - p)).astype(np.float64)
    return head["a"] * logit + head["b"]


def verdicts(llr: np.ndarray) -> np.ndarray:
    p = 1.0 / (1.0 + np.exp(-llr))
    return np.where(p >= FAKE_AT, 2, np.where(p <= REAL_AT, 0, 1))


def build_network(head: dict, layer: int, model_name: str):
    import torch
    from transformers import AutoModel

    class VoiceGuardNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wavlm = AutoModel.from_pretrained(model_name).float().eval()
            self.wavlm.encoder.layers = torch.nn.ModuleList(list(self.wavlm.encoder.layers)[:layer])
            for k in ("mean", "scale", "w1", "b1", "w2", "b2"):
                self.register_buffer(k, torch.tensor(np.asarray(head[k]), dtype=torch.float32))
            self.a, self.b = head["a"], head["b"]

        def forward(self, waveform):  # (1, samples), 16 kHz, values in [-1, 1]
            mean = waveform.mean(dim=1, keepdim=True)
            var = ((waveform - mean) ** 2).mean(dim=1, keepdim=True)
            x = (waveform - mean) / torch.sqrt(var + 1e-7)
            feats = self.wavlm(x).last_hidden_state.mean(dim=1)
            z = torch.relu((feats - self.mean) / self.scale @ self.w1 + self.b1)
            p = torch.clamp(torch.sigmoid((z @ self.w2 + self.b2)[:, 0]), 1e-6, 1 - 1e-6)
            return self.a * torch.log(p / (1 - p)) + self.b, feats

    return VoiceGuardNet().eval()


def load_clip(path: str) -> np.ndarray:
    """First 4 s of a 16 kHz clip, as in src/voiceguard/extract.py (kept here to avoid importing torchaudio)."""
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        raise ValueError("expected 16 kHz audio: " + path)
    return x[:64000]


def main() -> None:
    import joblib
    import onnxruntime as ort
    import torch
    from huggingface_hub import HfApi
    from onnxruntime.quantization import QuantType, quantize_dynamic

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN (a Hugging Face token with write access).")
    found = sorted(DATA.rglob(MODEL_FILE))
    if not found:
        sys.exit("%s not found under %s: add the VoiceGuard notebook output as input." % (MODEL_FILE, DATA))
    bundle = joblib.load(found[0])
    card = bundle["card"]
    recipe = card["feature_recipe"]
    manifest = json.loads((found[0].parent.parent / "manifest.json").read_text())
    print("model:", found[0], flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    report = {"model": card["name"], "eval_clean_eer_reported": card["eval_clean_eer_check"]["reported"]}

    # Check clips: a stratified sample of the eval clips used in the experiments (clean condition).
    proto = {u.utt_id: u for u in read_protocol(find_file(str(DATA), "ASVspoof2019.LA.cm.eval.trl.txt"))}
    pool = [proto[i] for i in manifest["eval"]]
    check = stratified_sample(pool, {"bonafide": 400}, 40, seed=0)
    flac = find_dir(str(DATA), "ASVspoof2019_LA_eval")
    paths = [os.path.join(flac, u.utt_id + ".flac") for u in check]
    y = np.array([u.label for u in check])
    clips = [load_clip(p) for p in paths]
    report["check_clips"] = {"n": len(check), "bonafide": int((y == 0).sum()), "spoof": int(y.sum()),
                             "seconds_min": round(min(len(c) for c in clips) / 16000, 2),
                             "seconds_max": round(max(len(c) for c in clips) / 16000, 2)}
    print("check clips:", report["check_clips"], flush=True)

    # 1. Head: NumPy weights must reproduce scikit-learn exactly.
    head = sklearn_head(bundle)
    probe = np.random.default_rng(0).standard_normal((64, len(head["mean"]))).astype(np.float32)
    sk = bundle["calibrator"].llr(bundle["model"].predict_proba(probe)[:, 1])
    gap = float(np.max(np.abs(head_llr(head, probe) - sk)))
    report["head_vs_sklearn_max_abs_llr"] = gap
    if gap > 1e-3:
        sys.exit("head weights do not reproduce scikit-learn (max |dLLR| %.2e)" % gap)

    # 2. Reference scores of the experiments: the saved eval features of the run (batched, float16 on GPU).
    layer = recipe["hidden_state_index"]
    feat_files = sorted(found[0].parent.parent.parent.rglob("eval%d_clean.npz" % len(manifest["eval"])))
    if not feat_files:
        sys.exit("eval%d_clean.npz not found next to the model: add the VoiceGuard notebook output as input."
                 % len(manifest["eval"]))
    row = {u: i for i, u in enumerate(manifest["eval"])}
    ref_feats = np.load(feat_files[0])["wavlm_l%d" % layer][[row[u.utt_id] for u in check]].astype(np.float32)
    llr_batched = head_llr(head, ref_feats)

    # 3. The demo network (single clip, float32): truncation must equal hidden state 6 of the full model.
    from transformers import AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_network(head, layer, recipe["wavlm_model"]).to(device)
    full = AutoModel.from_pretrained(recipe["wavlm_model"]).float().eval().to(device)
    with torch.no_grad():
        x = torch.from_numpy(clips[0])[None].to(device)
        xn = (x - x.mean(1, keepdim=True)) / torch.sqrt(((x - x.mean(1, keepdim=True)) ** 2).mean(1, keepdim=True) + 1e-7)
        a = full(xn, output_hidden_states=True).hidden_states[layer].mean(1)
        b = net(x)[1]
    trunc_gap = float((a - b).abs().max())
    report["truncated_vs_full_max_abs_feature"] = trunc_gap
    del full
    if trunc_gap > 1e-3:
        sys.exit("truncated WavLM does not match hidden state %d (gap %.2e)" % (layer, trunc_gap))
    with torch.no_grad():
        llr_torch = np.array([float(net(torch.from_numpy(c)[None].to(device))[0]) for c in clips])

    # 4. ONNX export (float32) and int8 copy.
    net = net.cpu()
    onnx32, onnx8 = WORK / "voiceguard_fp32.onnx", WORK / "voiceguard_int8.onnx"
    kwargs = dict(input_names=["waveform"], output_names=["llr", "features"],
                  dynamic_axes={"waveform": {1: "samples"}}, opset_version=17)
    with torch.no_grad():
        try:
            torch.onnx.export(net, torch.from_numpy(clips[0])[None], str(onnx32), dynamo=False, **kwargs)
        except TypeError:  # older PyTorch without the dynamo argument
            torch.onnx.export(net, torch.from_numpy(clips[0])[None], str(onnx32), **kwargs)
    quantize_dynamic(str(onnx32), str(onnx8), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])

    def run_onnx(path: Path) -> np.ndarray:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return np.array([float(sess.run(["llr"], {"waveform": c[None].astype(np.float32)})[0][0]) for c in clips])

    llr32, llr8 = run_onnx(onnx32), run_onnx(onnx8)

    def compare(name: str, llr: np.ndarray, base: np.ndarray) -> dict:
        r = {"max_abs_llr": float(np.max(np.abs(llr - base))), "mean_abs_llr": float(np.mean(np.abs(llr - base))),
             "verdict_agreement": float(np.mean(verdicts(llr) == verdicts(base)))}
        print("%-36s %s" % (name, r), flush=True)
        return r

    report["onnx_fp32_vs_torch"] = compare("ONNX float32 vs PyTorch (single clip)", llr32, llr_torch)
    report["onnx_int8_vs_onnx_fp32"] = compare("ONNX int8 vs ONNX float32", llr8, llr32)
    report["demo_fp32_vs_batched_experiment"] = compare("demo float32 vs batched experiment", llr32, llr_batched)
    report["eer_on_check_clips"] = {k: M.eer(y, v) for k, v in
                                    (("batched_experiment", llr_batched), ("pytorch_single_clip", llr_torch),
                                     ("onnx_fp32", llr32), ("onnx_int8", llr8))}
    report["size_mb"] = {"fp32": round(onnx32.stat().st_size / 1e6, 1), "int8": round(onnx8.stat().st_size / 1e6, 1)}
    print("EER on check clips:", report["eer_on_check_clips"], "| sizes MB:", report["size_mb"], flush=True)

    # Where do the single-clip demo scores and the batched experiment scores differ? In the experiments, shorter
    # clips were zero-padded to the longest clip of their batch.
    secs = np.array([len(c) / 16000 for c in clips])
    gap = np.abs(llr32 - llr_batched)
    report["demo_vs_batched_by_length"] = []
    for label, lo, hi in (("under 2 s", 0, 2), ("2-3 s", 2, 3), ("3-4 s", 3, 3.99), ("4 s (cut, so not padded)", 3.99, 9)):
        m = (secs >= lo) & (secs < hi)
        if m.any():
            report["demo_vs_batched_by_length"].append({
                "seconds": label, "n": int(m.sum()),
                "mean_abs_llr": float(gap[m].mean()),
                "verdict_agreement": float(np.mean(verdicts(llr32[m]) == verdicts(llr_batched[m])))})
    print("demo vs batched by clip length:", report["demo_vs_batched_by_length"], flush=True)

    eer = report["eer_on_check_clips"]
    fp32_ok = (report["onnx_fp32_vs_torch"]["verdict_agreement"] >= 0.995
               and report["onnx_fp32_vs_torch"]["mean_abs_llr"] <= 0.01
               and abs(eer["onnx_fp32"] - eer["pytorch_single_clip"]) <= 0.002)
    report["onnx_fp32_check"] = ("passed" if fp32_ok else "failed") + (
        ": verdict agreement >= 99.5%, mean |dLLR| <= 0.01 and EER within 0.2 points of PyTorch")
    (WORK / "parity.json").write_text(json.dumps(report, indent=1))
    if not fp32_ok:
        sys.exit("ONNX float32 does not reproduce PyTorch; nothing published (see parity.json).")
    use_int8 = (report["onnx_int8_vs_onnx_fp32"]["verdict_agreement"] >= 0.98
                and abs(eer["onnx_int8"] - eer["onnx_fp32"]) <= 0.01)
    chosen = onnx8 if use_int8 else onnx32
    report["published_model"] = "int8" if use_int8 else "fp32"
    report["rule"] = "int8 if verdict agreement with float32 >= 98% and EER within 1 point on the check clips"
    (WORK / "parity.json").write_text(json.dumps(report, indent=1))
    print("publishing the %s model" % report["published_model"], flush=True)
    if os.environ.get("VOICEGUARD_DRY_RUN"):
        print("dry run: not publishing")
        return

    # 5. Publish the static Space.
    api = HfApi(token=token)
    repo_id = "%s/%s" % (api.whoami()["name"], SPACE_NAME)
    fill = {"{model_mb}": "%.0f" % (chosen.stat().st_size / 1e6), "{precision}": report["published_model"],
            "{eer_demo}": "%.1f" % (100 * eer["onnx_%s" % report["published_model"]]),
            "{eer_batched}": "%.1f" % (100 * eer["batched_experiment"]),
            "{n_check}": str(len(check)),
            "{agree}": "%.1f" % (100 * report["demo_fp32_vs_batched_experiment"]["verdict_agreement"])}
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in ("index.html", "app.js", "README.md"):
            text = (ROOT / "demo" / name).read_text()
            for k, v in fill.items():
                text = text.replace(k, v)
            (stage / name).write_text(text)
        (stage / "voiceguard.onnx").write_bytes(chosen.read_bytes())
        (stage / "parity.json").write_text(json.dumps(report, indent=1))
        (stage / "model_card.json").write_text(json.dumps(card, indent=1))
        api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(stage),
                          commit_message="Publish VoiceGuard browser demo")
    print("\n=== Done ===\nDemo: https://huggingface.co/spaces/" + repo_id)


if __name__ == "__main__":
    main()
