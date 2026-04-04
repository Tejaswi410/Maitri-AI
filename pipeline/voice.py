import os
import sys
import torch
import numpy as np
import soundfile as sf
import re
from typing import Optional, Tuple

# NeuTTS local model initialization
# Ensure we can import from the local neutts-air directory
base_dir = os.path.dirname(os.path.dirname(__file__))
neutts_dir = os.path.join(base_dir, "voice_cloning", "model", "neutts-air - CPU (for AMD also)")
if neutts_dir not in sys.path:
    sys.path.append(neutts_dir)

# Set up espeak-ng paths before phonemizer is imported
def setup_espeak():
    possible_paths = [
        "C:\\Program Files\\eSpeak NG",
        "C:\\Program Files (x86)\\eSpeak NG",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            dll_path = os.path.join(path, "libespeak-ng.dll")
            if os.path.exists(dll_path):
                os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = dll_path
                os.environ['PATH'] = f"{path};{os.environ.get('PATH', '')}"
                return True
    return False

espeak_found = setup_espeak()
if not espeak_found:
    print("WARNING: espeak-ng not found. Local TTS may fail.")

# Import local NeuTTS after path setup
NeuTTSAir = None
try:
    from neuttsair.neutts import NeuTTSAir
    print("NeuTTSAir imported successfully.")
except ImportError as e:
    print(f"Error importing NeuTTSAir: {e}")

# Lazy model singleton
tts_model_instance = None

def get_tts_model():
    global tts_model_instance
    if tts_model_instance is None:
        print("Initializing local NeuTTS-Air model (first run)...")
        local_backbone_dir = os.path.join(neutts_dir, "Models", "neutts-air")

        def _resolve_hf_snapshot(root_path: str) -> str:
            try:
                for name in os.listdir(root_path):
                    if name.startswith("models--"):
                        snapshots_dir = os.path.join(root_path, name, "snapshots")
                        if os.path.isdir(snapshots_dir):
                            for snap in os.listdir(snapshots_dir):
                                snap_path = os.path.join(snapshots_dir, snap)
                                if os.path.exists(os.path.join(snap_path, "config.json")):
                                    return snap_path
            except Exception:
                pass
            return root_path

        backbone_arg = _resolve_hf_snapshot(local_backbone_dir) if os.path.isdir(local_backbone_dir) else "neutts-air-q4-gguf"
        print(f"Using backbone: {backbone_arg}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_model_instance = NeuTTSAir(
            backbone_repo=backbone_arg,
            backbone_device=device,
            codec_repo="neuphonic/neucodec",
            codec_device=device
        )
        print("NeuTTS-Air model initialized.")
    return tts_model_instance


def split_text_into_chunks(text: str, max_length: int = 300):
    """Split text into sentence-sized chunks suitable for TTS inference."""
    text = text.strip()
    if not text:
        return []

    sentence_pattern = r'([.!?]+)'
    parts = re.split(sentence_pattern, text)
    sentences = []
    i = 0
    while i < len(parts):
        if parts[i].strip():
            sentence = parts[i].strip()
            if i + 1 < len(parts) and parts[i + 1].strip():
                sentence += parts[i + 1]
                i += 2
            else:
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                i += 1
            sentences.append(sentence)
        else:
            i += 1

    if parts and parts[-1].strip():
        last_part = parts[-1].strip()
        if not any(last_part in s or s.startswith(last_part) for s in sentences):
            if not last_part.endswith(('.', '!', '?')):
                last_part += '.'
            sentences.append(last_part)

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(sentence) > max_length:
            comma_parts = re.split(r'(,)', sentence)
            i = 0
            while i < len(comma_parts):
                part = comma_parts[i].strip()
                comma = comma_parts[i + 1] if i + 1 < len(comma_parts) else ''
                if len(part) > max_length:
                    words = part.split()
                    temp_words = []
                    for word in words:
                        test_chunk = ' '.join(temp_words + [word])
                        if len(test_chunk) > max_length and temp_words:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = ""
                            chunks.append(' '.join(temp_words))
                            temp_words = [word]
                        else:
                            temp_words.append(word)
                    if temp_words:
                        part = ' '.join(temp_words) + comma
                        if current_chunk and len(current_chunk + ' ' + part) > max_length:
                            chunks.append(current_chunk.strip())
                            current_chunk = part
                        else:
                            current_chunk += (' ' if current_chunk else '') + part
                else:
                    part_with_comma = part + comma
                    if current_chunk and len(current_chunk + ' ' + part_with_comma) > max_length:
                        chunks.append(current_chunk.strip())
                        current_chunk = part_with_comma
                    else:
                        current_chunk += (' ' if current_chunk else '') + part_with_comma
                i += 2 if i + 1 < len(comma_parts) else 1
        else:
            if current_chunk and len(current_chunk + ' ' + sentence) > max_length:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (' ' if current_chunk else '') + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Deduplicate consecutive identical chunks
    final_chunks = []
    for chunk in chunks:
        if chunk.strip() and (not final_chunks or chunk.strip() != final_chunks[-1]):
            final_chunks.append(chunk.strip())

    return final_chunks


SAMPLE_RATE = 24000

PROFILES = {
    "Narendra Modi": {"audio": "ref_cropped.wav", "text": "Namaste mere pyare deshwasio"},
    "Andrew Tate": {"audio": "andrew_tate_cropped.wav", "text": "both President Putin"},
    "Donald Trump": {"audio": "donald_trump_cropped.wav", "text": "both President Putin"},
    "Virat Kohli": {"audio": "virat_kohli_cropped.wav", "text": "I never expected it to happen so"}
}


def speak(astronaut_id: str, text: str, voice_name: str = "Narendra Modi"):
    """
    Generates cloned voice audio using the local NeuTTS-Air model (CPU).

    Returns:
        A (sample_rate, numpy_array) tuple, which is what Gradio's gr.Audio expects.
        Returns None on failure.
    """
    print(f"Generating voice for: '{voice_name}' via local NeuTTS-Air")

    if NeuTTSAir is None:
        print("❌ NeuTTSAir was not imported — check espeak-ng and neuttsair package.")
        return None

    profile = PROFILES.get(voice_name, PROFILES["Narendra Modi"])
    ref_audio_path = os.path.join(base_dir, "voice_cloning", "voice_profiles", astronaut_id, profile["audio"])
    ref_text = profile["text"]

    if not os.path.exists(ref_audio_path):
        print(f"❌ Missing reference audio: {ref_audio_path}")
        return None

    output_path = os.path.join(base_dir, "voice_cloning", "outputs", f"{astronaut_id}_output.wav")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        tts = get_tts_model()

        # Step 1 – Encode voice reference
        print(f"Encoding voice reference for {voice_name}...")
        ref_codes = tts.encode_reference(ref_audio_path)

        # Step 2 – Split response text into manageable chunks
        chunks = split_text_into_chunks(text)
        if not chunks:
            print("❌ No text chunks to process.")
            return None

        print(f"Processing {len(chunks)} chunk(s)...")
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"  chunk {i+1}/{len(chunks)}: '{chunk[:60]}'...")
            try:
                chunk_wav = tts.infer(chunk, ref_codes, ref_text)
                if chunk_wav is not None:
                    processed_chunks.append(np.array(chunk_wav, dtype=np.float32))
            except Exception as chunk_err:
                print(f"  ❌ Chunk error: {chunk_err}")

        if not processed_chunks:
            print("❌ All chunks failed — no audio generated.")
            return None

        # Step 3 – Concatenate with small silence gaps
        silence = np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32)
        all_wav = processed_chunks[0]
        for chunk_wav in processed_chunks[1:]:
            all_wav = np.concatenate([all_wav, silence, chunk_wav])

        # Normalise to prevent clipping
        all_wav = all_wav.astype(np.float32)
        max_val = np.max(np.abs(all_wav))
        if max_val > 1.0:
            all_wav = all_wav / max_val

        # Step 4 – Save a backup .wav file to disk
        sf.write(output_path, all_wav, SAMPLE_RATE)
        print(f"✅ Audio saved to {output_path}")

        # Step 5 – Convert to int16 (Gradio gr.Audio requires int16, not float32)
        # float32 → int16: multiply by 32767 and cast
        all_wav_int16 = (all_wav * 32767).astype(np.int16)
        
        # Return (sample_rate, int16_array) — what Gradio gr.Audio expects
        return (SAMPLE_RATE, all_wav_int16)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ speak() failed: {e}")
        return None
