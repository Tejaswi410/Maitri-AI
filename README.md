# Maitri AI

Maitri AI is a voice-based mental health support assistant designed for astronauts in isolated environments.

It runs a full pipeline with a local Gradio web interface (`app.py`):
1. **Speech-to-Text**: Transcription from astronaut audio input using the Groq API (`whisper-large-v3`).
2. **Emotion Detection**: DistilRoBERTa model from HuggingFace to detect emotion from transcript text.
3. **Memory Retrieval**: Astronaut-specific personal memory retrieval via FAISS + embeddings (`sentence-transformers`).
4. **Compassionate Response**: Real-time response generation with the Groq API (`llama-3.1-8b-instant`).
5. **Voice Output**: Local voice cloning using the `NeuTTS-Air` CPU model.

## Project Structure

```text
.
|-- app.py
|-- pipeline/
|   |-- stt.py
|   |-- emotion.py
|   |-- memory.py
|   |-- llm.py
|   `-- voice.py
|-- memory/
|   |-- build_memory_index.py
|   `-- ASTRO_00X/
|       |-- memories.json
|       `-- faiss.index (generated)
|-- voice_cloning/
|   |-- input/
|   |-- outputs/
|   `-- voice_profiles/
|       `-- ASTRO_00X/NaMo.wav
|-- requirements.txt
`-- .env
```

## Requirements

- Python 3.10
- `eSpeak NG` installed and accessible in `C:\Program Files\eSpeak NG` (required on Windows for local TTS).
- `ffmpeg` combined with `imageio-ffmpeg` for Gradio audio processing.

## Setup (Local)

1. Create and activate a virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Ensure required assets exist:
- Create a `.env` file in the root directory and add your keys/settings:
  ```env
  GROQ_API_KEY=your_groq_api_key_here
  ASTRONAUT_ID=ASTRO_001
  ```
- Astronaut memory file at `memory/<ASTRONAUT_ID>/memories.json`
- Voice reference sample at `voice_cloning/voice_profiles/<ASTRONAUT_ID>/NaMo.wav`

4. (Optional) Prebuild FAISS index:

```powershell
python memory/build_memory_index.py
```
If the index is missing, it is built automatically at runtime when first invoked.

## Run

To launch the Gradio web interface, run:

```powershell
python app.py
```

The app will start on a local URL (e.g., `http://127.0.0.1:7860`). Open that in your browser to record audio and talk to Maitri AI. Output voice samples are saved to `voice_cloning/outputs/`.

## Docker

*Note: running local TTS (NeuTTS-Air) dynamically depends on local paths. Extra configuration for eSpeak NG within Linux containers is required for complete local execution.*

Build image:
```powershell
docker build -t maitri-ai .
```

Run container (mount local assets and pass `.env`):
```powershell
docker run --rm `
  --env-file .env `
  -v "${PWD}/memory:/app/memory" `
  -v "${PWD}/voice_cloning:/app/voice_cloning" `
  -p 7860:7860 `
  maitri-ai
```

## Memory Data Format

`memory/<ASTRONAUT_ID>/memories.json` should contain a JSON array of objects with a `text` field:

```json
[
  { "text": "Your family is proud of your space mission." },
  { "text": "Your sister misses your Sunday cooking sessions." }
]
```

## Adding a New Astronaut Profile

1. Create `memory/<NEW_ID>/memories.json`.
2. Add voice sample `voice_cloning/voice_profiles/<NEW_ID>/NaMo.wav`.
3. Set `ASTRONAUT_ID=<NEW_ID>` in your `.env` file.
4. Run `python app.py`.
5. The first run will process the new profile and create `memory/<NEW_ID>/faiss.index` automatically.
