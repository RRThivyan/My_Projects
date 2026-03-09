# 🎙️ SpeechBridge — Multilingual Voice Assistant

> **Speak in English or any Indian language. Get an intelligent spoken response back.**

SpeechBridge is a production-style **Speech-to-Speech AI pipeline** that transcribes your voice, generates a smart response, and speaks it back — all in your language. Built with Google Cloud Speech APIs and Azure OpenAI GPT-4o.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-6.x-F97316?logo=gradio&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Speech%20APIs-4285F4?logo=google-cloud&logoColor=white)
![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-GPT--4o-0078D4?logo=microsoft-azure&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E)

---

## 📸 Demo

![SpeechBridge UI](assets/demo.png)

> 🔗 **Live Demo:** [Coming soon on HuggingFace Spaces](#)

---

## 🧠 Pipeline Architecture

```
🎙️ Voice Input
      │
      ▼
┌─────────────────┐
│  Google STT     │  ← Transcribes speech to text (6 languages)
└────────┬────────┘
         │  transcript
         ▼
┌─────────────────┐
│  Azure GPT-4o   │  ← Generates intelligent response in same language
└────────┬────────┘
         │  response text
         ▼
┌─────────────────┐
│  Google TTS     │  ← Converts response to natural-sounding speech
└────────┬────────┘
         │
         ▼
    🔊 Audio Output
```

Each stage is **fully modular** — swap any component independently (e.g. replace Azure GPT-4o with local Ollama, or Google TTS with F5-TTS).

---

## 🌐 Supported Languages

| Language   | Code    | Script     | STT | TTS |
|------------|---------|------------|-----|-----|
| English    | `en-IN` | Latin      | ✅  | ✅  |
| Tamil      | `ta-IN` | தமிழ்      | ✅  | ✅  |
| Hindi      | `hi-IN` | देवनागरी     | ✅  | ✅  |
| Telugu     | `te-IN` | తెలుగు      | ✅  | ✅  |
| Kannada    | `kn-IN` | ಕನ್ನಡ      | ✅  | ✅  |
| Malayalam  | `ml-IN` | മലയാളം  | ✅  | ✅  |

> Mixed-language (code-switching) speech is handled via Google STT's `alternative_language_codes`.

---

## ✨ Features

- 🎤 **3 input modes** — Live microphone, file upload, or bundled sample audios
- 🗣️ **Multilingual STT** — Google Cloud Speech-to-Text with language fallback detection
- 🤖 **GPT-4o responses** — Azure OpenAI with full multi-turn conversation memory
- 🔊 **Natural TTS** — Google WaveNet voices, one per Indian language
- 💬 **Persistent chat history** — full conversation displayed per session
- 🌍 **Code-switching support** — handles mixed language speech naturally
- ⚡ **Modular pipeline** — each of STT / LLM / TTS independently replaceable

---

## 🗂️ Project Structure

```
SpeechBridge/
├── app.py                    # Gradio UI — all 3 input modes, chat display
├── pipeline/
│   ├── pipeline.py           # Orchestrator: STT → LLM → TTS
│   ├── stt.py                # Google Cloud Speech-to-Text
│   ├── llm.py                # Azure OpenAI GPT-4o with conversation memory
│   └── tts.py                # Google Cloud Text-to-Speech (WaveNet)
├── utils/
│   └── language_utils.py     # Language codes, names, labels
├── assets/
│   ├── Voice1.wav            # Bundled sample audio 1
│   ├── Voice2.wav            # Bundled sample audio 2
│   └── Voice3.wav            # Bundled sample audio 3
├── temp/                     # Auto-created — TTS output files saved here
├── .env                      # Environment variable template
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/SpeechBridge.git
cd SpeechBridge
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> Also install **ffmpeg** (required by pydub for audio conversion):
> - Windows: [Download from ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
> - macOS: `brew install ffmpeg`
> - Linux: `sudo apt install ffmpeg`

### 4. Configure credentials
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env
GOOGLE_APPLICATION_CREDENTIALS=E:\path\to\your\service-account-key.json
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2025-01-01-preview
```

**Getting credentials:**
- **Google Cloud** → [Create a service account](https://console.cloud.google.com/iam-admin/serviceaccounts) with `Cloud Speech-to-Text User` and `Cloud Text-to-Speech User` roles → download JSON key
- **Azure OpenAI** → Get API key and endpoint from [Azure OpenAI Studio](https://oai.azure.com/)

> Make sure **Speech-to-Text** and **Text-to-Speech** APIs are enabled in your Google Cloud project.

### 5. Run the app
```bash
python app.py
```

Open **http://localhost:7860** in your browser.

To generate a **public shareable link**, set `share=True` in `app.py`:
```python
demo.launch(share=True, ...)
```

---

## 🔧 Swap Components

### Replace LLM → Local Ollama
```python
# pipeline/llm.py
import ollama

class OllamaLLM:
    def generate(self, user_message, conversation_history, language_name):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # ... add history ...
        messages.append({"role": "user", "content": user_message})
        response = ollama.chat(model="llama3", messages=messages)
        return response["message"]["content"]
```

### Replace TTS → F5-TTS (free, local)
Replace `GoogleTTS` in `pipeline/tts.py` with F5-TTS for fully offline, zero-cost voice synthesis — especially useful for Indian language voices.

### Add a new language
1. Add entry to `utils/language_utils.py`
2. Add voice config to `VOICE_MAP` in `pipeline/tts.py`

---

## 📊 Latency Benchmarks

| Stage               | Avg. Time         |
|---------------------|-------------------|
| Google STT          | ~0.8 – 1.5s       |
| Azure GPT-4o        | ~0.5 – 1.2s       |
| Google TTS          | ~0.3 – 0.7s       |
| **End-to-End**      | **~2 – 3.5s**     |

*Tested on consumer broadband. Varies with audio length and network conditions.*

---

## 🛣️ Roadmap

- [ ] Swap Google TTS → F5-TTS for fully local deployment
- [ ] Add Indic TTS for more natural Indian language voices
- [ ] Streaming TTS for lower perceived latency
- [ ] Docker deployment
- [ ] HuggingFace Spaces permanent demo

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

**RR Thivyan** · AI/ML Engineer  
[LinkedIn](https://www.linkedin.com/in/thivyan-rr) · [GitHub](https://github.com/RRThivyan/My_Projects)
