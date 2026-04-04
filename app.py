import os
import gradio as gr
from dotenv import load_dotenv
import imageio_ffmpeg

# Load environment variables FIRST before any ML pipelines or web clients initialize
load_dotenv()

# Prevent downloading if model is already cached to avoid httpx concurrency crashes on startup
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Auto-inject ffmpeg so Gradio can process microphone audio on Windows
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

from pipeline.stt import transcribe
from pipeline.emotion import detect_emotion
from pipeline.memory import retrieve_relevant_memory
from pipeline.llm import generate_response
from pipeline.voice import speak

# We need a consistent astronaut ID for memory tracking
ASTRONAUT_ID = os.getenv("ASTRONAUT_ID", "ASTRO_001")

def process_interaction(audio_filepath: str, voice_name: str):
    """
    Main pipeline function to process the audio from the UI,
    run the AI pipeline, and return the outputs for the web interface.
    """
    if not os.environ.get("GROQ_API_KEY"):
         return "ERROR: Missing GROQ_API_KEY in .env file.", "N/A", "Please add your Groq API key to the .env file and restart.", None

    if not audio_filepath or not os.path.exists(audio_filepath):
         return "No audio provided.", "N/A", "Please record your voice first.", None
         
    # 1. Speech-to-Text
    text = transcribe(audio_filepath)
    
    # 2. Emotion Detection
    emotion = detect_emotion(text)
    
    # 3. Memory Retrieval
    memory = retrieve_relevant_memory(ASTRONAUT_ID, text)
    
    # 4. LLM Response
    response_text = generate_response(emotion, memory, text)
    
    # 5. Voice Generation via local NeuTTS-Air Voice Cloning
    output_audio_path = speak(ASTRONAUT_ID, response_text, voice_name)
    
    return text, emotion.capitalize(), response_text, output_audio_path

# --- UI Layout & Design ---
with gr.Blocks(
    title="Maitri AI - Mental Health support",
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="blue",
        neutral_hue="slate"
    )
) as app:
    
    gr.Markdown(
        """
        # 🌌 Maitri AI: AI Assistant for mental health support
        Welcome to Maitri. I am here to listen, understand, and support you during your mission.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Speak to Maitri")
            audio_input = gr.Audio(
                sources=["microphone", "upload"], 
                type="filepath",
                label="Record or Upload your message"
            )
            voice_dropdown = gr.Dropdown(
                choices=["Narendra Modi", "Andrew Tate", "Donald Trump", "Virat Kohli"],
                value="Narendra Modi",
                label="Select Maitri's Voice"
            )
            submit_btn = gr.Button("Send Message", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            gr.Markdown("### 2. Analysis")
            text_output = gr.Textbox(label="Transcribed Text", lines=3, interactive=False)
            emotion_output = gr.Textbox(label="Detected Emotion", lines=1, interactive=False)
            
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 3. Maitri's Response")
            llm_output = gr.Textbox(label="Response Text", lines=4, interactive=False)
            audio_output = gr.Audio(label="Response Audio", interactive=False)

    # Wire up the button to the processing function
    submit_btn.click(
        fn=process_interaction,
        inputs=[audio_input, voice_dropdown],
        outputs=[text_output, emotion_output, llm_output, audio_output]
    )

if __name__ == "__main__":
    app.launch(share=False)
