
import time
from typing import Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from .config import (
    AZURE_ENDPOINT,
    AZURE_DEPLOYMENT,
    AZURE_API_KEY,
    AZURE_API_VERSION,
    TEMPERATURE,
    DEBUG_MODE,
)
from .prompting import build_prompt


def call_llm(context_str: str, question: str) -> Dict[str, Any]:
    prompt = build_prompt(context_str, question)
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        deployment_name=AZURE_DEPLOYMENT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        temperature=TEMPERATURE,
    )

    t0 = time.time()
    raw = llm.invoke(prompt)
    gen_latency = time.time() - t0

    if DEBUG_MODE:
        print(f"[DEBUG] Generation latency: {gen_latency:.3f}s")

    text = getattr(raw, "content", raw)
    if not isinstance(text, str):
        text = str(text)

    if DEBUG_MODE:
        print("[DEBUG] Raw LLM output (truncated 1000 chars):")
        print(text[:1000])

    parser = JsonOutputParser()
    result = parser.parse(text)
    return result
