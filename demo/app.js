// VoiceGuard browser demo: the clip is decoded, resampled and scored on this device; nothing is uploaded.
const MODEL_URL = "voiceguard.onnx";
const SR = 16000;
const MAX_SAMPLES = 4 * SR; // the detector looks at the first 4 seconds, as in training
const MIN_SAMPLES = 1 * SR;
const FAKE_AT = 0.95; // probability thresholds at an even prior (same as demo/build_space.py)
const REAL_AT = 0.05;

const $ = (id) => document.getElementById(id);
const state = { session: null, recorder: null, chunks: [] };

function setStatus(text, kind) {
  const el = $("status");
  el.textContent = text;
  el.className = "status " + (kind || "");
}

function enable(on) {
  $("file").disabled = !on;
  $("record").disabled = !on;
}

async function loadModel() {
  setStatus("Loading the detector (about {model_mb} MB, only the first time)…", "busy");
  ort.env.wasm.numThreads = 1;
  state.session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
  setStatus("Detector ready. Upload a voice clip or record one.", "ok");
  enable(true);
}

// Decode any format the browser supports and resample to 16 kHz mono.
async function decode(arrayBuffer) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
  try {
    const buf = await ctx.decodeAudioData(arrayBuffer);
    const n = buf.length;
    const mono = new Float32Array(n);
    for (let c = 0; c < buf.numberOfChannels; c++) {
      const ch = buf.getChannelData(c);
      for (let i = 0; i < n; i++) mono[i] += ch[i] / buf.numberOfChannels;
    }
    return { samples: mono, seconds: n / buf.sampleRate, rate: buf.sampleRate };
  } finally {
    ctx.close?.();
  }
}

function verdict(p) {
  if (p >= FAKE_AT) return ["Likely AI-generated", "high"];
  if (p <= REAL_AT) return ["Likely real", "low"];
  return ["Uncertain", "mid"];
}

async function analyse(arrayBuffer, label) {
  enable(false);
  $("result").innerHTML = "";
  try {
    setStatus("Reading the audio…", "busy");
    const { samples, seconds, rate } = await decode(arrayBuffer);
    if (rate !== SR) throw new Error("This browser could not resample the audio to 16 kHz.");
    if (samples.length < MIN_SAMPLES) throw new Error("The clip is shorter than 1 second. Please use 1 to 10 seconds of speech.");
    const x = samples.slice(0, MAX_SAMPLES);
    let energy = 0;
    for (let i = 0; i < x.length; i++) energy += x[i] * x[i];
    if (Math.sqrt(energy / x.length) < 1e-4) throw new Error("The clip is silent or almost silent.");

    setStatus("Analysing…", "busy");
    const t0 = performance.now();
    const input = new ort.Tensor("float32", x, [1, x.length]);
    const out = await state.session.run({ waveform: input });
    const llr = out.llr.data[0];
    const ms = Math.round(performance.now() - t0);
    const p = 1 / (1 + Math.exp(-llr));
    const [text, kind] = verdict(p);
    $("result").innerHTML =
      `<div class="verdict ${kind}">${text}</div>` +
      `<table><tbody>` +
      `<tr><td>Probability of an AI voice</td><td class="num">${(100 * p).toFixed(1)}%</td></tr>` +
      `<tr><td>Log-likelihood ratio</td><td class="num">${llr >= 0 ? "+" : ""}${llr.toFixed(2)}</td></tr>` +
      `<tr><td>Audio analysed</td><td class="num">${(x.length / SR).toFixed(1)} of ${seconds.toFixed(1)} s</td></tr>` +
      `</tbody></table>` +
      `<p class="caption">${label}. The probability assumes real and AI voices are equally likely before listening. ` +
      `A positive log-likelihood ratio points to an AI voice, a negative one to a real voice.</p>`;
    setStatus(`Done in ${ms} ms, on this device.`, "ok");
  } catch (e) {
    setStatus(e.message || String(e), "error");
  } finally {
    enable(true);
  }
}

function showPlayer(blob) {
  const player = $("player");
  player.src = URL.createObjectURL(blob);
  player.hidden = false;
}

$("file").addEventListener("change", async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  showPlayer(f);
  await analyse(await f.arrayBuffer(), "Uploaded file: " + f.name.replace(/[<>&"]/g, ""));
  ev.target.value = "";
});

$("record").addEventListener("click", async () => {
  if (state.recorder && state.recorder.state === "recording") {
    state.recorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.chunks = [];
    state.recorder = new MediaRecorder(stream);
    state.recorder.ondataavailable = (e) => e.data.size && state.chunks.push(e.data);
    state.recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      $("record").textContent = "Record";
      const blob = new Blob(state.chunks, { type: state.recorder.mimeType || "audio/webm" });
      showPlayer(blob);
      await analyse(await blob.arrayBuffer(), "Recorded clip");
    };
    state.recorder.start();
    $("record").textContent = "Stop recording";
    setStatus("Recording… speak for 2 to 10 seconds, then press Stop.", "busy");
  } catch (e) {
    setStatus("Microphone not available: " + (e.message || e), "error");
  }
});

loadModel().catch((e) => setStatus("Could not load the detector: " + (e.message || e), "error"));
