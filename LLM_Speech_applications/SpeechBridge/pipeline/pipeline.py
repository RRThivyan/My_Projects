"""
VoiceAssistantPipeline — orchestrates STT → LLM → TTS
"""

from pipeline.stt import GoogleSTT
from pipeline.llm import AzureOpenAILLM
from pipeline.tts import GoogleTTS
from utils.language_utils import SUPPORTED_LANGUAGES


class VoiceAssistantPipeline:
    def __init__(self):
        self.stt = GoogleSTT()
        self.llm = AzureOpenAILLM()
        self.tts = GoogleTTS()

    def run(self, audio_path: str, language_code: str, conversation_history: list) -> dict:
        """
        Full pipeline: audio file → transcript → LLM response → audio output

        Returns:
            dict with keys: transcript, detected_language, llm_response,
                            audio_output, error
        """
        result = {
            "transcript": None,
            "detected_language": language_code,
            "llm_response": None,
            "audio_output": None,
            "error": None,
        }

        # ── Step 1: STT ──────────────────────────────────────────────────────
        try:
            transcript, detected_lang = self.stt.transcribe(audio_path, language_code)
            if not transcript or not transcript.strip():
                result["error"] = "Could not transcribe audio. Please speak clearly and try again."
                return result
            result["transcript"] = transcript.strip()
            result["detected_language"] = detected_lang or language_code
        except Exception as e:
            result["error"] = f"STT error: {str(e)}"
            return result

        # ── Step 2: LLM ──────────────────────────────────────────────────────
        try:
            lang_name = SUPPORTED_LANGUAGES.get(language_code, {}).get("name", "English")
            llm_response = self.llm.generate(
                user_message=result["transcript"],
                conversation_history=conversation_history,
                language_name=lang_name
            )
            result["llm_response"] = llm_response
        except Exception as e:
            result["error"] = f"LLM error: {str(e)}"
            return result

        # ── Step 3: TTS ──────────────────────────────────────────────────────
        try:
            audio_output_path = self.tts.synthesize(
                text=result["llm_response"],
                language_code=language_code
            )
            result["audio_output"] = audio_output_path
        except Exception as e:
            result["error"] = f"TTS error: {str(e)}"
            return result

        return result
