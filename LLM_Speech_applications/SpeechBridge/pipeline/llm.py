"""
Azure OpenAI LLM wrapper with conversation memory
"""

import os
from openai import AzureOpenAI


SYSTEM_PROMPT = """You are a helpful, friendly multilingual voice assistant. 
You support English and Indian languages including Tamil, Hindi, Telugu, Kannada, and Malayalam.

Key rules:
- Always respond in the SAME language the user spoke in. If they spoke Tamil, reply in Tamil. If Hindi, reply in Hindi.
- Keep responses concise (2-4 sentences) since they will be read aloud via Text-to-Speech.
- Avoid markdown formatting, bullet points, or symbols — plain natural speech only.
- Be conversational, warm, and helpful.
- If asked in a mix of English and another language (code-switching), match that style.
"""


class AzureOpenAILLM:
    """
    Generates responses using Azure OpenAI (GPT-4o) with conversation history.
    """

    def __init__(self):
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

        if not api_key or not endpoint:
            raise EnvironmentError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set."
            )

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    def generate(self, user_message: str, conversation_history: list, language_name: str = "English") -> str:
        """
        Generate a response given the user message and conversation history.

        Args:
            user_message: Latest user transcript
            conversation_history: List of previous {"role": ..., "content": ...} dicts
            language_name: Human-readable language name for context hint

        Returns:
            LLM response string
        """
        # Start with system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Build messages from conversation history (list of role/content dicts)
        for turn in conversation_history:
            raw = turn.get("content", "")
            clean = raw.split("] ", 1)[-1] if "] " in raw else raw
            clean = clean.replace("🤖 ", "").strip()
            role = "assistant" if turn["role"] == "assistant" else "user"
            if clean:
                messages.append({"role": role, "content": clean})

        # Add current user message with language hint
        messages.append({
            "role": "user",
            "content": f"[The user spoke in {language_name}]\n{user_message}"
        })

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )

        return response.choices[0].message.content.strip()
