"""
Edge Vision Model Benchmark
===========================

Benchmarks four lightweight vision models on a single sample image:
    1. MobileNetV4 (Conv-Small)   -> timm            (ImageNet top-3)
    2. YOLO11n                    -> ultralytics     (object detection)
    3. MobileCLIP-S0              -> open_clip_torch (zero-shot)
    4. FastViT-T8                 -> timm            (ImageNet top-3)

Install dependencies:

    pip install torch torchvision timm ultralytics open_clip_torch pillow numpy

Run:

    python small_models.py
"""

import io
import os
import time
import urllib.request

import numpy as np
import torch
from PIL import Image

# --------------------------------------------------------------------------- #
# 1. Device handling                                                          #
# --------------------------------------------------------------------------- #

def select_device():
    """Pick CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = select_device()


def synchronize():
    """Block until all queued device work is finished (needed for honest timing)."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


print(f"[setup] Using device: {DEVICE}")


# --------------------------------------------------------------------------- #
# 2. Sample image + ImageNet labels (download with graceful fallbacks)        #
# --------------------------------------------------------------------------- #

SAMPLE_PATH = "sample_image.jpg"
IMAGE_URLS = [
    # Laptop / electronics on a desk
    "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=640&q=80",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/"
    "Laptop-hard-drive-exposed.jpg/640px-Laptop-hard-drive-exposed.jpg",
]
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
LABELS_PATH = "imagenet_classes.txt"


def _fetch(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def load_sample_image():
    """Return a PIL RGB image, caching it to disk so YOLO can read a file path."""
    if os.path.exists(SAMPLE_PATH):
        return Image.open(SAMPLE_PATH).convert("RGB")

    for url in IMAGE_URLS:
        try:
            data = _fetch(url)
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.save(SAMPLE_PATH)
            print(f"[setup] Downloaded sample image from {url}")
            return img
        except Exception as e:  # noqa: BLE001
            print(f"[setup] Could not fetch {url}: {e}")

    # Last resort: synthetic image so the benchmark still runs.
    print("[setup] WARNING: using a synthetic image (predictions will be meaningless).")
    img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    img.save(SAMPLE_PATH)
    return img


def load_imagenet_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            return [ln.strip() for ln in f if ln.strip()]
    try:
        text = _fetch(LABELS_URL).decode("utf-8")
        with open(LABELS_PATH, "w") as f:
            f.write(text)
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    except Exception as e:  # noqa: BLE001
        print(f"[setup] Could not fetch ImageNet labels: {e}")
        return [f"class_{i}" for i in range(1000)]


IMAGE = load_sample_image()
IMAGENET_LABELS = load_imagenet_labels()


# --------------------------------------------------------------------------- #
# 3. Timing harness                                                           #
# --------------------------------------------------------------------------- #

WARMUP_ITERS = 10
TIMED_ITERS = 50


def benchmark(run_once, warmup=WARMUP_ITERS, iters=TIMED_ITERS):
    """Warm up, then time `run_once` and return (mean_ms, p95_ms)."""
    for _ in range(warmup):
        run_once()
    synchronize()

    times_ms = []
    for _ in range(iters):
        synchronize()
        t0 = time.perf_counter()
        run_once()
        synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return float(np.mean(times_ms)), float(np.percentile(times_ms, 95))


def count_params_m(module):
    return sum(p.numel() for p in module.parameters()) / 1e6


# --------------------------------------------------------------------------- #
# 4. Model benchmarks                                                         #
# --------------------------------------------------------------------------- #

def bench_timm(model_name, display_name):
    """MobileNetV4 / FastViT: ImageNet classification with top-3 output."""
    import timm

    model = timm.create_model(model_name, pretrained=True).eval().to(DEVICE)
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    x = transform(IMAGE).unsqueeze(0).to(DEVICE)

    def run_once():
        with torch.inference_mode():
            model(x)

    mean_ms, p95_ms = benchmark(run_once)

    with torch.inference_mode():
        probs = model(x).softmax(dim=-1)[0]
    top = torch.topk(probs, 3)
    preds = [
        f"{IMAGENET_LABELS[i]} ({p * 100:.1f}%)"
        for p, i in zip(top.values.tolist(), top.indices.tolist())
    ]

    print(f"\n[{display_name}] top-3:")
    for line in preds:
        print(f"    {line}")

    return {
        "name": display_name,
        "params_m": count_params_m(model),
        "mean_ms": mean_ms,
        "p95_ms": p95_ms,
        "top": preds[0],
    }


def bench_yolo():
    """YOLO11n: object detection with boxes + confidence."""
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    model.to(DEVICE)
    dev = DEVICE.type  # ultralytics accepts 'cuda' / 'mps' / 'cpu'

    def run_once():
        model.predict(SAMPLE_PATH, device=dev, imgsz=640, verbose=False)

    mean_ms, p95_ms = benchmark(run_once)

    results = model.predict(SAMPLE_PATH, device=dev, imgsz=640, verbose=False)
    boxes = results[0].boxes
    names = model.names

    detections = []
    print("\n[YOLO11n] detections:")
    if boxes is None or len(boxes) == 0:
        print("    (no objects detected)")
        top = "no detections"
    else:
        order = torch.argsort(boxes.conf, descending=True)
        for idx in order.tolist():
            cls = names[int(boxes.cls[idx])]
            conf = float(boxes.conf[idx])
            x1, y1, x2, y2 = [round(v, 1) for v in boxes.xyxy[idx].tolist()]
            detections.append((cls, conf, (x1, y1, x2, y2)))
            print(f"    {cls:<15} conf={conf * 100:5.1f}%  box=({x1}, {y1}, {x2}, {y2})")
        top_cls, top_conf, _ = detections[0]
        top = f"{top_cls} ({top_conf * 100:.1f}%)"

    return {
        "name": "YOLO11n",
        "params_m": count_params_m(model.model),
        "mean_ms": mean_ms,
        "p95_ms": p95_ms,
        "top": top,
    }


def bench_mobileclip():
    """MobileCLIP-S0: zero-shot classification over custom categories."""
    import open_clip

    categories = ["electronics", "furniture", "clothing", "kitchenware", "vehicle"]

    model, _, preprocess = open_clip.create_model_and_transforms(
        "MobileCLIP-S0", pretrained="datacompdr"
    )
    tokenizer = open_clip.get_tokenizer("MobileCLIP-S0")
    model.eval().to(DEVICE)

    prompts = [f"a photo of {c}" for c in categories]
    text_tokens = tokenizer(prompts).to(DEVICE)
    with torch.inference_mode():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    x = preprocess(IMAGE).unsqueeze(0).to(DEVICE)

    def run_once():
        with torch.inference_mode():
            model.encode_image(x)

    mean_ms, p95_ms = benchmark(run_once)

    with torch.inference_mode():
        image_features = model.encode_image(x)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]

    ranked = sorted(zip(categories, probs.tolist()), key=lambda t: t[1], reverse=True)
    print("\n[MobileCLIP-S0] zero-shot probabilities:")
    for cat, p in ranked:
        print(f"    {cat:<12} {p * 100:5.1f}%")

    best_cat, best_p = ranked[0]
    return {
        "name": "MobileCLIP-S0",
        "params_m": count_params_m(model),
        "mean_ms": mean_ms,
        "p95_ms": p95_ms,
        "top": f"{best_cat} ({best_p * 100:.1f}%)",
    }


# --------------------------------------------------------------------------- #
# 5. Run everything + summary table                                           #
# --------------------------------------------------------------------------- #

def main():
    benches = [
        ("MobileNetV4-ConvS", lambda: bench_timm("mobilenetv4_conv_small.e2400_r224_in1k",
                                                 "MobileNetV4-ConvS")),
        ("YOLO11n", bench_yolo),
        ("MobileCLIP-S0", bench_mobileclip),
        ("FastViT-T8", lambda: bench_timm("fastvit_t8.apple_in1k", "FastViT-T8")),
    ]

    results = []
    for name, fn in benches:
        try:
            results.append(fn())
        except Exception as e:  # noqa: BLE001
            print(f"\n[{name}] FAILED: {type(e).__name__}: {e}")
            results.append({
                "name": name, "params_m": float("nan"),
                "mean_ms": float("nan"), "p95_ms": float("nan"),
                "top": f"ERROR: {type(e).__name__}",
            })

    # Summary table
    print("\n" + "=" * 92)
    print(f"BENCHMARK SUMMARY  (device={DEVICE}, warmup={WARMUP_ITERS}, timed={TIMED_ITERS})")
    print("=" * 92)
    header = f"{'Model':<20}{'Params (M)':>12}{'Mean (ms)':>12}{'P95 (ms)':>12}  {'Top Output':<32}"
    print(header)
    print("-" * 92)
    for r in results:
        top = r["top"] if len(r["top"]) <= 32 else r["top"][:29] + "..."
        print(f"{r['name']:<20}{r['params_m']:>12.2f}{r['mean_ms']:>12.2f}"
              f"{r['p95_ms']:>12.2f}  {top:<32}")
    print("=" * 92)


if __name__ == "__main__":
    main()
