# AgenticApproaches

Experiments with vision-language models (VLMs) for understanding objects in photos.

| Folder | What it is |
| --- | --- |
| `MultipleVLMs/vlm-dimension-agent` | Estimates an object's width, height and depth from 3 to 5 photos with four VLM "agents" (Observer, Estimator, Critic, Judge). No classical computer vision or depth maps. Has its own README. |
| `QuenVL2` | Quick tests of Qwen2-VL-2B, InternVL2.5-2B and LLaVA-OneVision, plus a small benchmark of edge vision models (MobileNetV4, YOLO11n, MobileCLIP, FastViT). |
| `smolVLM` | First try with SmolVLM. |

Model weights are not included; they download from Hugging Face on first run.

Status: the dimension agent's pipeline runs end to end, but it has not yet been scored against real measurements.
