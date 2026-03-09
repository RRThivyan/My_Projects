"""
Google Cloud Text-to-Speech (TTS) wrapper
Supports high-quality WaveNet voices for English + Indian languages
"""

import os
import tempfile
from google.cloud import texttospeech

# Local temp folder — sits alongside assets/ in the project root
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# WaveNet / Neural2 voice map per language code
VOICE_MAP = {
    "en-IN": {"name": "en-IN-Wavenet-D", "gender": texttospeech.SsmlVoiceGender.MALE},
    "ta-IN": {"name": "ta-IN-Wavenet-D", "gender": texttospeech.SsmlVoiceGender.MALE},
    "hi-IN": {"name": "hi-IN-Wavenet-D", "gender": texttospeech.SsmlVoiceGender.MALE},
    "te-IN": {"name": "te-IN-Wavenet-D", "gender": texttospeech.SsmlVoiceGender.MALE},
    "kn-IN": {"name": "kn-IN-Wavenet-D", "gender": texttospeech.SsmlVoiceGender.MALE},
    "ml-IN": {"name": "ml-IN-Wavenet-D", "gender": texttospeech.SsmlVoiceGender.MALE},
}

# Fallback for unsupported voices (Standard tier)
FALLBACK_VOICE = {"name": None, "gender": texttospeech.SsmlVoiceGender.NEUTRAL}


class GoogleTTS:
    """
    Synthesizes speech using Google Cloud Text-to-Speech API.
    Returns path to a .mp3 file saved in the project's temp/ folder.
    """

    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()  # Uses GOOGLE_APPLICATION_CREDENTIALS

    def synthesize(self, text: str, language_code: str) -> str:
        """
        Convert text to speech audio file.

        Args:
            text: Text to synthesize
            language_code: BCP-47 code e.g. 'ta-IN', 'en-IN'

        Returns:
            Path to generated .mp3 audio file in project temp/ folder
        """
        voice_config = VOICE_MAP.get(language_code, FALLBACK_VOICE)

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_config["name"],
            ssml_gender=voice_config["gender"],
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.95,
            pitch=0.0,
            effects_profile_id=["headphone-class-device"],
        )

        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # Save to project temp/ folder
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=TEMP_DIR)
        tmp.write(response.audio_content)
        tmp.close()
        return tmp.name
