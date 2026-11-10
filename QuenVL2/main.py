import time

import torch

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL = "./models/Qwen2-VL-2B-Instruct"

print("1. Loading model...")

model = Qwen2VLForConditionalGeneration.from_pretrained(
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
            {
                "type": "image",
                "image": "Images/Night-Lords-Throne-of-Lies.jpg",
            },
            {
                "type": "text",
                "text": "What is in the image?"
            },
        ],
    }
]




text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("4. Processing image...")

image_inputs, video_inputs = process_vision_info(messages)

print("5. Image processed!")

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

print("6. Inputs created!")

inputs = inputs.to(model.device)

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