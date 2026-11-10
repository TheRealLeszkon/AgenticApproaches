import time

import torch
from PIL import Image

from transformers import InternVLForConditionalGeneration, AutoProcessor

MODEL = "./models/InternVL2_5-2B-MPO-hf"
IMAGE = "Images/Night-Lords-Throne-of-Lies.jpg"

print("1. Loading model...")

model = InternVLForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype="auto",
    device_map="auto"
)

print("2. Model loaded!")

processor = AutoProcessor.from_pretrained(MODEL)

print("3. Processor loaded!")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is in the image?"},
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True
)

print("4. Processing image...")

image = Image.open(IMAGE).convert("RGB")

print("5. Image processed!")

inputs = processor(
    images=image,
    text=text,
    return_tensors="pt",
)

print("6. Inputs created!")

inputs = inputs.to(model.device, model.dtype)

print("7. Starting generation...")

start_time = time.time()

generated_ids = model.generate(
    **inputs,
    max_new_tokens=128
)

inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.2f} seconds")

generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(output[0])

print("8. Generation complete!")
