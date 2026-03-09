"""
Google Cloud Speech-to-Text (STT) wrapper
Supports multilingual recognition with automatic language detection fallback
"""

import os
import io
from google.cloud import speech
from pydub import AudioSegment


class GoogleSTT:
    """
    Transcribes audio using Google Cloud Speech-to-Text API.
    Handles audio format conversion and multilingual recognition.
    """

    def __init__(self):
        self.client = speech.SpeechClient()  # Uses GOOGLE_APPLICATION_CREDENTIALS env var

    def _convert_to_wav(self, audio_path: str) -> bytes:
        """Convert any audio format to 16kHz mono WAV PCM (required by Google STT)."""
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        return buffer.getvalue()

    def transcribe(self, audio_path: str, language_code: str) -> tuple[str, str]:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (any format pydub supports)
            language_code: BCP-47 language code e.g. 'ta-IN', 'en-IN'

        Returns:
            (transcript, detected_language_code)
        """
        audio_bytes = self._convert_to_wav(audio_path)

        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=language_code,
            # Alternative languages for fallback detection
            alternative_language_codes=self._get_alternative_languages(language_code),
            enable_automatic_punctuation=True,
            model="latest_long",
        )

        response = self.client.recognize(config=config, audio=audio)

        if not response.results:
            return "", language_code

        best_result = response.results[0]
        transcript = best_result.alternatives[0].transcript
        detected_lang = getattr(best_result, "language_code", language_code) or language_code

        return transcript, detected_lang

    def _get_alternative_languages(self, primary: str) -> list[str]:
        """Provide alternative language codes for mixed-language speech."""
        alternatives = {
            "ta-IN": ["en-IN"],
            "hi-IN": ["en-IN"],
            "te-IN": ["en-IN"],
            "kn-IN": ["en-IN"],
            "ml-IN": ["en-IN"],
            "en-IN": ["hi-IN", "ta-IN"],
        }
        return alternatives.get(primary, ["en-IN"])
