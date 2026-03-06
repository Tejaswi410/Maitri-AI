from transformers import pipeline

_emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

_emotion_map = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "anxious",
    "fear": "anxious",
    "neutral": "calm",
    "disgust": "anxious",
    "surprise": "calm"
}

#for detecting the emotion of the text
def detect_emotion(text: str) -> str:
    scores = _emotion_classifier(text)[0]
    scores = sorted(scores, key=lambda x: x["score"], reverse=True)

    label = scores[0]["label"]
    return _emotion_map.get(label, "calm")
