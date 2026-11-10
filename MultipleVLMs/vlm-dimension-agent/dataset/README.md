# Benchmark dataset

Each object is one folder containing 3–5 photos of the **same** physical object from
different viewpoints, plus a `ground_truth.json` with the real measured dimensions in cm.

```
dataset/
├── object_001/
│   ├── 01.jpg
│   ├── 02.jpg
│   ├── 03.jpg
│   └── ground_truth.json
├── object_002/
│   └── ...
```

`ground_truth.json`:

```json
{
  "width_cm": 120,
  "height_cm": 180,
  "depth_cm": 45
}
```

Run the benchmark over every `object_*` folder that has a `ground_truth.json`:

```bash
python benchmark.py --model smolvlm-256m --dataset ./dataset
```

## About `object_001`

`object_001` is a worked example seeded from existing project photos — a **Premier Soft
Face Tissues** box on a tiled floor (3 viewpoints). **Its `ground_truth.json` values are
PLACEHOLDERS** (`"_placeholder": true`). Measure the real box (or swap in your own object)
and replace the numbers before drawing any conclusions from the reported errors.

Tips for good test objects:
- include a genuine, in-frame scale reference when you can (outlet, tile, coin, hand)
- capture faces that reveal all three dimensions (don't shoot all photos head-on)
- measure carefully — the benchmark is only as honest as your ground truth
