FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    ASTRONAUT_ID=ASTRO_001 \
    AUDIO_INPUT=voice_cloning/input/input.wav

WORKDIR /app

# System packages needed by whisper/TTS/audio + potential llama-cpp build fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install openai-whisper==20250625

COPY . .

# Ensure runtime output/cache directories exist
RUN mkdir -p /app/voice_cloning/outputs /app/.cache/huggingface

CMD ["python", "main.py"]
