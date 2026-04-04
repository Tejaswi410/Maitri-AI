import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Transcribe the audio file and return the text using Groq's whisper
def transcribe(audio_path: str) -> str:
    with open(audio_path, "rb") as file:
        transcription = _client.audio.transcriptions.create(
            file=(audio_path, file.read()),
            model="whisper-large-v3",
            response_format="text",
        )
    return transcription.strip()
