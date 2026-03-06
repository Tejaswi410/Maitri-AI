from TTS.api import TTS
import os

tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    gpu=False
)

def generate_voice(astronaut_id, text):
    speaker_wav = f"voice_profiles/{astronaut_id}/sample_audio.wav"
    output_path = f"outputs/{astronaut_id}.wav"

    if not os.path.exists(speaker_wav):
        raise FileNotFoundError(f"Missing speaker file: {speaker_wav}")

    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language="en",
        file_path=output_path
    )

    return output_path

if __name__ == "__main__":
    astronaut_id = "ASTRO_002"
    text = "Chirag, You motherfucker bitch , when are you going to start studying, should I tell you dad?"

    output = generate_voice(astronaut_id, text)
    print("Voice generated at:", output)
