"""
SpeechBridge — Multilingual Voice Assistant
STT → LLM → TTS | English + Indian Languages
Google Cloud Speech APIs + Azure OpenAI GPT-4o
"""

import gradio as gr
import os
import shutil
import tempfile
from dotenv import load_dotenv
load_dotenv()

from pipeline.pipeline import VoiceAssistantPipeline
from utils.language_utils import SUPPORTED_LANGUAGES

# ── Init pipeline ──────────────────────────────────────────────────────────────
pipeline = VoiceAssistantPipeline()

# ── Sample audio files ─────────────────────────────────────────────────────────
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
SAMPLE_AUDIOS = {
    "🎵 Sample 1": os.path.join(ASSETS_DIR, "Voice1.wav"),
    "🎵 Sample 2": os.path.join(ASSETS_DIR, "Voice2.wav"),
    "🎵 Sample 3": os.path.join(ASSETS_DIR, "Voice3.wav"),
}
SAMPLE_LABELS = list(SAMPLE_AUDIOS.keys())


# ── Handlers ───────────────────────────────────────────────────────────────────
def process_audio(audio_path, language_code, conversation_history):
    if audio_path is None:
        return conversation_history, conversation_history, None, "⚠️ No audio provided."

    result = pipeline.run(audio_path, language_code, conversation_history)

    if result["error"]:
        return conversation_history, conversation_history, None, f"❌ {result['error']}"

    # Gradio 6.9 messages format: list of dicts with role + content
    conversation_history.append({"role": "user",      "content": f"🎙️ [{result['detected_language']}] {result['transcript']}"})
    conversation_history.append({"role": "assistant", "content": f"🤖 {result['llm_response']}"})

    return (
        conversation_history,
        conversation_history,
        result["audio_output"],
        f"✅ Transcribed: *{result['transcript']}*"
    )


def load_sample(sample_label):
    path = SAMPLE_AUDIOS.get(sample_label)
    if path and os.path.exists(path):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        shutil.copy(path, tmp.name)
        tmp.close()
        return tmp.name, f"📂 Loaded **{sample_label}** — click Send to process"
    return None, "⚠️ Sample file not found. Check assets/ folder."


def clear_chat():
    return [], [], None, ""


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="SpeechBridge — Multilingual Voice Assistant") as demo:

    conversation_state = gr.State([])

    gr.HTML("""
        <div style="text-align:center; padding:2rem 0 1rem;">
            <h1 style="font-size:2rem; font-weight:700; margin:0;">🎙️ SpeechBridge</h1>
            <p style="color:#6b7280; margin-top:0.4rem; font-size:0.95rem;">
                Multilingual Voice Assistant — English, Tamil, Hindi, Telugu, Kannada, Malayalam
            </p>
        </div>
    """)

    with gr.Row():
        language_dropdown = gr.Dropdown(
            choices=[(v["label"], k) for k, v in SUPPORTED_LANGUAGES.items()],
            value="en-IN",
            label="🌐 Language",
            scale=1
        )
        gr.HTML("""
            <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:8px 4px;">
                <span style="background:#ede9fe;color:#5b21b6;padding:3px 10px;border-radius:99px;font-size:0.78rem;">English</span>
                <span style="background:#fef3c7;color:#92400e;padding:3px 10px;border-radius:99px;font-size:0.78rem;">தமிழ்</span>
                <span style="background:#dcfce7;color:#166534;padding:3px 10px;border-radius:99px;font-size:0.78rem;">हिन्दी</span>
                <span style="background:#dbeafe;color:#1e40af;padding:3px 10px;border-radius:99px;font-size:0.78rem;">తెలుగు</span>
                <span style="background:#fce7f3;color:#9d174d;padding:3px 10px;border-radius:99px;font-size:0.78rem;">ಕನ್ನಡ</span>
                <span style="background:#e0f2fe;color:#075985;padding:3px 10px;border-radius:99px;font-size:0.78rem;">മലയാളം</span>
            </div>
        """, scale=2)

    with gr.Row():

        # ── Left: input panel ──────────────────────────────────────────────────
        with gr.Column(scale=1):
            with gr.Tabs():

                with gr.Tab("🎤 Microphone"):
                    mic_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Record your message"
                    )
                    mic_send_btn = gr.Button("▶  Send Recording", variant="primary", size="lg")

                with gr.Tab("📁 Upload Audio"):
                    upload_input = gr.Audio(
                        sources=["upload"],
                        type="filepath",
                        label="Upload a WAV / MP3 file"
                    )
                    upload_send_btn = gr.Button("▶  Send Uploaded Audio", variant="primary", size="lg")

                with gr.Tab("🎵 Sample Audios"):
                    gr.Markdown("**Load a bundled sample** then click Send.")
                    sample_preview = gr.Audio(
                        type="filepath",
                        label="▶ Preview / Selected Sample",
                        interactive=False,
                        autoplay=False
                    )
                    sample_status = gr.Markdown("")
                    with gr.Row():
                        btn1 = gr.Button(SAMPLE_LABELS[0], size="sm")
                        btn2 = gr.Button(SAMPLE_LABELS[1], size="sm")
                        btn3 = gr.Button(SAMPLE_LABELS[2], size="sm")
                    sample_send_btn = gr.Button("▶  Send Sample Audio", variant="primary", size="lg")

            status_label = gr.Markdown("")
            gr.HTML('<hr style="margin:10px 0; border-color:#e5e7eb;"/>')
            audio_output = gr.Audio(
                type="filepath",
                label="🔊 Assistant Response",
                autoplay=True
            )
            clear_btn = gr.Button("🗑  Clear Conversation", variant="secondary", size="sm")

        # ── Right: chat panel ──────────────────────────────────────────────────
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="💬 Conversation",
                height=520,
                avatar_images=(
                    "https://api.dicebear.com/7.x/icons/svg?seed=user&icon=person",
                    "https://api.dicebear.com/7.x/icons/svg?seed=bot&icon=robot"
                )
            )

    with gr.Accordion("📐 Pipeline Architecture", open=False):
        gr.HTML("""
        <div style="background:#f8fafc;border-radius:12px;padding:20px;font-size:0.85rem;line-height:2.2;">
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-family:monospace;">
                <span style="background:#7c3aed;color:white;padding:6px 14px;border-radius:8px;">🎙️ Mic / 📁 Upload / 🎵 Sample</span>
                <span>──▶</span>
                <span style="background:#2563eb;color:white;padding:6px 14px;border-radius:8px;">Google STT</span>
                <span>──▶</span>
                <span style="background:#0078D4;color:white;padding:6px 14px;border-radius:8px;">Azure GPT-4o</span>
                <span>──▶</span>
                <span style="background:#dc2626;color:white;padding:6px 14px;border-radius:8px;">Google TTS</span>
                <span>──▶</span>
                <span style="background:#7c3aed;color:white;padding:6px 14px;border-radius:8px;">🔊 Speaker</span>
            </div>
            <p style="color:#6b7280;margin-top:10px;font-family:sans-serif;font-size:0.8rem;">
                All three input modes share the same pipeline. Conversation memory is maintained per session.
            </p>
        </div>
        """)

    # ── Wire up events ─────────────────────────────────────────────────────────
    shared_outputs = [chatbot, conversation_state, audio_output, status_label]

    mic_send_btn.click(fn=process_audio, inputs=[mic_input, language_dropdown, conversation_state], outputs=shared_outputs)
    upload_send_btn.click(fn=process_audio, inputs=[upload_input, language_dropdown, conversation_state], outputs=shared_outputs)
    sample_send_btn.click(fn=process_audio, inputs=[sample_preview, language_dropdown, conversation_state], outputs=shared_outputs)

    btn1.click(fn=load_sample, inputs=gr.State(SAMPLE_LABELS[0]), outputs=[sample_preview, sample_status])
    btn2.click(fn=load_sample, inputs=gr.State(SAMPLE_LABELS[1]), outputs=[sample_preview, sample_status])
    btn3.click(fn=load_sample, inputs=gr.State(SAMPLE_LABELS[2]), outputs=[sample_preview, sample_status])

    clear_btn.click(fn=clear_chat, outputs=[chatbot, conversation_state, audio_output, status_label])


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="indigo",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("DM Sans"), "ui-sans-serif"]
        ),
        css="""
            .gradio-container { max-width: 960px !important; margin: auto; }
            footer { display: none !important; }
        """
    )
