# vlm-dimension-agent

Estimate the physical **width / height / depth** of an object from **3–5 photographs**
using open-weight vision-language models — through **visual reasoning only**. No classical
CV, camera intrinsics, depth maps, point clouds, stereo, or 3D reconstruction.

This is a proof-of-concept built to answer one question:

> *Given these images, how much can an open-weight VLM actually infer about the physical
> dimensions of an object, and how reliable is that estimate?*

It is deliberately optimized for **honesty and useful uncertainty**, not impressive-looking
precise numbers.

## How it works — a 4-agent pipeline

Each agent has a **distinct** job (it is not the same prompt four times):

```
3–5 images
   ├─▶ Observer   what can we observe? (category, geometry, scale refs) — NO guessing sizes
   ├─▶ Estimator  estimate W/H/D, labeling each as observed / inferred / assumed
   ├─▶ Critic     try to DISPROVE the estimate; propose corrected ranges
   └─▶ Judge      weigh the evidence (not blind averaging) → final JSON + ranges + warnings
```

Every stage emits validated JSON, and every run is saved for inspection under `runs/`.

---

## 1. Installation

Python 3.13. Using [`uv`](https://docs.astral.sh/uv/) (matches the rest of this repo):

```bash
cd vlm-dimension-agent
uv sync                      # core deps (torch, transformers, pillow, pydantic, pyyaml)
```

Or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional extras (install only what a given model/backend needs):

```bash
# Local Apple Silicon runtime (4-bit quantized models):
pip install "mlx-vlm>=0.1.0"           # macOS arm64 only

# InternVL3 / MiniCPM-V remote code + preprocessing:
pip install einops timm sentencepiece

# NVIDIA 4-bit quantization for the transformers backend:
pip install bitsandbytes
```

---

## 2. GPU requirements & choosing a backend

There are **two interchangeable backends**, selectable via `--backend` or `config.yaml`:

| backend        | where it runs                    | notes |
|----------------|----------------------------------|-------|
| `transformers` | CUDA GPU (best), CPU, Apple MPS  | full precision / bf16; needs lots of VRAM for 8B+ |
| `mlx`          | Apple Silicon (M-series) only    | uses 4-bit community weights; **the way to run locally on a Mac** |

Approximate memory per model (bf16 on CUDA vs 4-bit via MLX), and whether it fits a **16 GB
Apple Silicon** machine:

| model            | HF repo                              | bf16 VRAM | 4-bit (MLX) | fits 16 GB Mac? |
|------------------|--------------------------------------|-----------|-------------|-----------------|
| `qwen2.5-vl-7b`  | Qwen/Qwen2.5-VL-7B-Instruct          | ~16 GB    | ~5 GB       | ✅ (MLX)         |
| `qwen3-vl-8b`    | Qwen/Qwen3-VL-8B-Instruct            | ~18 GB    | ~6 GB       | ✅ (MLX, tight)  |
| `internvl3-8b`   | OpenGVLab/InternVL3-8B               | ~17 GB    | ~5 GB       | ✅ (MLX)         |
| `minicpm-v-4.5`  | openbmb/MiniCPM-V-4_5                | ~17 GB    | ~6 GB       | ✅ (MLX, tight)  |
| `internvl3-14b`  | OpenGVLab/InternVL3-14B              | ~30 GB    | ~9 GB       | ⚠️ tight / no    |
| `qwen3-vl-30b`   | Qwen/Qwen3-VL-30B-A3B-Instruct       | ~62 GB    | ~18 GB      | ❌ needs GPU / ≥32 GB |
| `smolvlm-256m`   | HuggingFaceTB/SmolVLM-256M-Instruct  | <1 GB     | —           | ✅ (smoke test)  |

**On a 16 GB Mac:** use `--backend mlx` with a 7B/8B model. The larger models need a CUDA
GPU (or a bigger Mac). `qwen3-vl-30b` in bf16 needs an A100/H100-class GPU.

> Numbers are rough guidance for weights + a modest image/KV budget; real usage varies with
> image count/resolution and generation length.

### Optional alternative runtimes
For heavier local serving you can also run several of these under **vLLM** or
**llama.cpp/Ollama** (GGUF). This project does **not** depend on them; the two supported
backends above are enough for the POC. If you wire one up, add an adapter under `models/`
implementing `VisionModel.generate` and a REGISTRY entry.

---

## 3. Downloading & running each model

Weights download automatically from the Hugging Face Hub on first use (cached under
`~/.cache/huggingface`). For gated repos, set `HF_TOKEN` (see `.env.example`).

- **Transformers backend**: uses the `hf` repo id in `models/base.py::REGISTRY`.
- **MLX backend**: uses the `mlx` repo id (community 4-bit conversions). If a listed
  `mlx-community/...-4bit` repo doesn't exist yet, either pick another quantization on the
  Hub or convert it yourself with `mlx_vlm.convert`, then update the REGISTRY entry.

---

## 4. Run the POC

```bash
# Smoke test — runs on ANY hardware, validates the whole pipeline (accuracy is poor):
python main.py --model smolvlm-256m --backend transformers \
  --images dataset/object_001/*.jpg

# Real run on a Mac (Apple Silicon), 4-bit via MLX:
python main.py --model qwen3-vl-8b --backend mlx --images ./images/*.jpg

# Real run on a CUDA GPU:
python main.py --model internvl3-8b --backend transformers --images ./images/*

# Save the final JSON somewhere:
python main.py --model qwen3-vl-8b --images ./images/* --output result.json
```

CLI flags: `--model --images --backend --config --output --max-image-side`. Anything not
passed falls back to `config.yaml`, then `VLM_*` env vars.

## 5. Run the benchmark

```bash
python benchmark.py --model smolvlm-256m --dataset ./dataset
```

Reports per-dimension **MAPE**, overall MAPE, **range-hit rate** (how often the true value
lands inside the predicted `[min,max]`), and a confidence-vs-error summary. See
`dataset/README.md` for the dataset format. (The bundled `object_001` uses **placeholder**
ground truth — replace it with real measurements.)

---

## 6. Expected output

Console (stderr) shows staged progress and a final summary:

```
[Observer]
  ok
[Estimator]
  ok
[Critic]
  ok
[Judge]
  ok

Final result:
Object: wooden cabinet
Width:  120 cm  (range 115–125)
Height: 180 cm  (range 172–188)
Depth:  47 cm  (range 40–55)
Confidence: 0.71
  ! Depth has significantly higher uncertainty than width and height.
```

Every run also writes intermediate + final JSON so you can see *where* the reasoning went
wrong:

```
runs/2026-08-18_103000/
├── observer.json
├── estimator.json
├── critic.json
├── judge.json
└── final.json      # includes _meta (model, images, per-stage ok flags)
```

If a stage returns prose instead of JSON, the pipeline issues one correction prompt; if that
also fails it records the raw text under `_error`/`_raw_*` in that stage's file and keeps
going.

---

## 7. Known limitations

- **Reasoning only.** There is no metric ground truth from geometry — accuracy hinges on the
  presence of genuine, correctly-identified scale references. Without one, absolute size is
  largely an educated guess from typical dimensions.
- **Depth is the weakest dimension** (the far side is usually unseen); expect wider ranges
  and lower confidence there.
- **Quantization (4-bit / MLX) degrades quality** vs full precision.
- **The largest models won't run locally** on 16 GB — use a CUDA GPU or a bigger machine.
- **`smolvlm-256m` is a plumbing smoke test only** — its estimates are not meaningful.
- VLM output varies run to run; schemas are intentionally lenient and validation is for
  structure/diagnostics, not correctness.

The point of exposing all intermediate reasoning is to let us analyze failures and decide
whether classical computer vision / geometric reconstruction is actually necessary for the
packing-survey measurement step.

---

## Project layout

```
main.py          CLI: images -> pipeline -> JSON
benchmark.py     evaluate against dataset/ (MAPE, range-hit, confidence-vs-error)
pipeline.py      Observer -> Estimator -> Critic -> Judge orchestration + run artifacts
config.yaml      model/backend/device/dtype/quant/max_image_side/temperature/max_tokens
agents/          base + observer/estimator/critic/judge (distinct prompts)
models/          base (registry+factory) + mlx_backend + per-family transformers adapters
schemas/         Pydantic output schemas (lenient)
utils/           config, image, json_parser, logging
dataset/         benchmark objects (each: images + ground_truth.json)
runs/            per-run intermediate + final JSON (gitignored)
```
