import whisper

_model = whisper.load_model("base")  

# Transcribe the audio file and return the text
def transcribe(audio_path: str) -> str:
    result = _model.transcribe(audio_path)
    return result["text"].strip()
