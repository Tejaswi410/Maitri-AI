from TTS.api import TTS
import os

_tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    gpu=False
)

#for generating the voice of the astronaut
def speak(astronaut_id: str, text: str) -> str:
    speaker_wav = f"voice_cloning/voice_profiles/{astronaut_id}/Namo.wav"
    output_path = f"voice_cloning/outputs/{astronaut_id}_output.wav"

    if not os.path.exists(speaker_wav):
        raise FileNotFoundError(f"Voice file not found for {astronaut_id}")

    _tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language="en",
        temperature=0.6,
        file_path=output_path
    )

    return output_path
