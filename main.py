# main.py

import os

from pipeline.stt import transcribe
from pipeline.emotion import detect_emotion
from pipeline.memory import retrieve_relevant_memory
from pipeline.llm import generate_response
from pipeline.voice import speak

def run_system(astronaut_id: str, audio_input: str):

    # 1. STT
    text = transcribe(audio_input)
    print("Transcribed:", text)

    # 2. Emotion
    emotion = detect_emotion(text)
    print("Emotion:", emotion)

    # 3. Memory Retrieval
    memory = retrieve_relevant_memory(astronaut_id, text)
    print("Memory:", memory)

    # 4. LLM Response
    response = generate_response(emotion, memory, text)
    print("Generated response:", response)

    # 5. Voice Output
    output_audio = speak(astronaut_id, response)
    print("Voice Output:", output_audio)


if __name__ == "__main__":
    astronaut_id = os.getenv("ASTRONAUT_ID", "ASTRO_001")
    audio_input = os.getenv("AUDIO_INPUT", "voice_cloning/input/input.wav")
    run_system(astronaut_id, audio_input)
