# extractor/qwen_vl_extractor.py

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import re
import json
from typing import Tuple, List

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

# Load model and processor once at module import
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)


def extract_from_image(image_path: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": (
                        "Analyze this Booking.com UI screenshot.\n"
                        "1. Extract all readable text exactly as shown.\n"
                        "2. Identify labels, values, filters, prices, times, and tags.\n"
                        "3. If you infer structured data (like cards or filters), also output it as JSON.\n"
                        "Return normal text first, then any JSON blocks."
                    ),
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
        )

    generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    extracted_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return extracted_text


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(8:10 AM, 12h 45m, 1 stop,?)+", "8:10 AM, 12h 45m, 1 stop", text)
    return text.strip()


def extract_json_blocks(text: str) -> List[dict]:
    blocks = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    parsed_blocks = []
    for b in blocks:
        try:
            parsed_blocks.append(json.loads(b))
        except json.JSONDecodeError:
            continue
    return parsed_blocks


def remove_json_blocks(text: str) -> str:
    return re.sub(r"\{.*?\}", "", text, flags=re.DOTALL)


def normalize_image_text(raw: str) -> str:
    """Turn Qwen raw response into retrieval-friendly text."""
    json_blocks = extract_json_blocks(raw)
    text_part = clean_text(remove_json_blocks(raw))

    normalized = [text_part]
    for obj in json_blocks:
        try:
            normalized.append(clean_text(json.dumps(obj, ensure_ascii=False)))
        except TypeError:
            continue
    return " ".join(normalized).strip()


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def extract_image_chunks(image_path: str) -> Tuple[list, str]:
    """Helper used by ingestion: returns normalized text chunks + modality."""
    raw = extract_from_image(image_path)
    normalized = normalize_image_text(raw)
    return chunk_text(normalized), "image"
