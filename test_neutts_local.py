import os
import sys

# Add the parent directory to the path so we can import the pipeline
sys.path.append(os.path.dirname(__file__))

from pipeline.voice import speak

def test_local_voice():
    print("Testing local NeuTTS voice cloning...")
    
    # Use the test astronaut ID and a short text
    astronaut_id = "ASTRO_001"
    text = "Hello world! This is a test of the local voice cloning system running entirely on my computer without API rate limits."
    voice_name = "Narendra Modi"
    
    output_audio_path = speak(astronaut_id, text, voice_name)
    
    if output_audio_path and os.path.exists(output_audio_path):
        print("\n\n✅ PERFECT! Local voice cloning successfully generated the audio file.")
        print(f"Checkout the file at: {output_audio_path}")
    else:
        print("\n\n❌ FAILED! Local voice cloning did not produce an audio file.")

if __name__ == "__main__":
    test_local_voice()
