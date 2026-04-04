import os
import ast
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

_BLOCKED_LABELS = (
    "ai response",
    "detected emotion",
    "detected emotions",
    "relevant memory",
    "relievent memory",
    "user message",
    "voice sample",
    "subject",
    "assistant",
    "system",
    "user",
    "emotion=",
    "memory=",
)

def _normalize(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", normalized).strip()

def _has_user_overlap(sentence_norm: str, user_norm: str, ngram_size: int = 6) -> bool:
    words = sentence_norm.split()
    if len(words) < ngram_size:
        return sentence_norm in user_norm
    for i in range(len(words) - ngram_size + 1):
        phrase = " ".join(words[i : i + ngram_size])
        if phrase in user_norm:
            return True
    return False

def _clean_response(text: str, user_text: str) -> str:
    cleaned = text.strip()

    # If model returns a Python-like list, parse and merge only content lines.
    if cleaned.startswith("[") and cleaned.endswith("]"):
        try:
            items = ast.literal_eval(cleaned)
            if isinstance(items, list):
                cleaned = " ".join(str(item) for item in items)
        except (SyntaxError, ValueError):
            pass

    # Remove common template and role labels produced by small models.
    cleaned = re.sub(r"^(Example\s*\d+\s*:.*\n)+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"(?im)^\s*(Therapist responding|Assistant|AI Response|System|User|Voice Sample|Subject|Detected emotions?|Relevant memory|Relievent memory|User message)\s*:\s*",
        "",
        cleaned,
    )
    # Remove metadata chunks (label + attached content) even when inline.
    cleaned = re.sub(
        r"(?i)\b(ai response|detected emotions?|relevant memory|relievent memory|user message|emotion|memory)\s*[:=]\s*[^.!?\n]{0,240}(?:[.!?]|$)",
        " ",
        cleaned,
    )

    # Remove quoted fragments and list punctuation often spoken by TTS.
    cleaned = cleaned.replace("['", "").replace("']", "")
    cleaned = cleaned.replace("', '", " ")
    cleaned = cleaned.replace('", "', " ")
    cleaned = cleaned.replace("[", " ").replace("]", " ")

    # Drop any remaining full lines that still start with a role-like prefix.
    role_prefix = re.compile(
        r"^\s*(user|assistant|system|voice sample|subject)\s*:\s*",
        flags=re.IGNORECASE,
    )
    lines = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        if role_prefix.match(line):
            continue
        lines.append(line)
    cleaned = " ".join(lines)

    # Remove inline role labels that may remain after list flattening.
    cleaned = re.sub(
        r"(?i)\b(user|assistant|system|voice sample|subject|detected emotions?|relevant memory|relievent memory|user message|ai response)\s*[:=]\s*",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)^\s*(happy|sad|calm|anxious|curious)(\s+and\s+(happy|sad|calm|anxious|curious))*\s+",
        "",
        cleaned,
    )

    # Remove sentences that contain blocked labels or are copied from user text.
    user_norm = _normalize(user_text)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    filtered = []
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        s_norm = _normalize(s)
        # Remove short metadata leftovers.
        if len(s_norm.split()) <= 3 and any(w in s_norm for w in ("happy", "sad", "calm", "anxious", "curious")):
            continue
        if any(label in s_norm for label in _BLOCKED_LABELS):
            continue
        # Skip direct echoes of the user transcript.
        if len(s_norm) > 20 and _has_user_overlap(s_norm, user_norm):
            continue
        filtered.append(s)

    cleaned = " ".join(filtered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" '\".,;:-")
    return cleaned

def generate_response(emotion: str, memory: str, user_text: str) -> str:
    output = _client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Maitri, a compassionate mental health AI assistant. "
                    "Reply in EXACTLY 2-3 short sentences. Plain text only. No labels, no lists. "
                    "Be warm, direct, and empathetic. Never repeat the user's words back."
                ),
            },
            {
                "role": "system",
                "content": f"Emotion detected: {emotion}. Relevant context: {memory}",
            },
            {"role": "user", "content": user_text},
        ],
        max_tokens=120,
        temperature=0.45,
        top_p=0.9,
    )

    text = output.choices[0].message.content
    cleaned = _clean_response(text, user_text)
    if not cleaned:
        return "I am here with you. Please take one slow breath, and tell me what feels most difficult right now."
    return cleaned
