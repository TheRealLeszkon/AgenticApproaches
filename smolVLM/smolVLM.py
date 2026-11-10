from transformers import pipeline
import sys
pipe = pipeline(
    "image-text-to-text",
    model="HuggingFaceTB/SmolVLM-256M-Instruct"
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "test.jpeg"},
            {"type": "text", "text": sys.argv[1]}
        ]
    }
]

result = pipe(text=messages)

print(result)
