import faiss
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer

_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def _build_and_save_index(memory_path: str, index_path: str):
    with open(memory_path, "r", encoding="utf-8") as f:
        memories = json.load(f)

    texts = [m["text"] for m in memories]
    if not texts:
        raise ValueError(f"No memories found in {memory_path}")

    embeddings = _embedding_model.encode(texts).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    return index, memories

#for loading the memory of the astronaut
def load_astronaut_memory(astronaut_id: str):
    index_path = f"memory/{astronaut_id}/faiss.index"
    memory_path = f"memory/{astronaut_id}/memories.json"

    if not os.path.exists(memory_path):
        raise FileNotFoundError(f"Memory file not found: {memory_path}")

    if not os.path.exists(index_path):
        return _build_and_save_index(memory_path, index_path)

    try:
        index = faiss.read_index(index_path)
    except RuntimeError:
        return _build_and_save_index(memory_path, index_path)

    with open(memory_path, "r", encoding="utf-8") as f:
        memories = json.load(f)

    return index, memories

#for retrieving the relevant memory of the astronaut
def retrieve_relevant_memory(astronaut_id: str, user_text: str) -> str:
    index, memories = load_astronaut_memory(astronaut_id)

    query_embedding = _embedding_model.encode([user_text])
    D, I = index.search(np.array(query_embedding), 1)

    return memories[I[0][0]]["text"]
