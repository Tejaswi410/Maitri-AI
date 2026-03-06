import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(astronaut_id):
    base_dir = os.path.dirname(__file__)
    astronaut_dir = os.path.join(base_dir, astronaut_id)
    memory_path = os.path.join(astronaut_dir, "memories.json")
    index_path = os.path.join(astronaut_dir, "faiss.index")

    with open(memory_path, "r", encoding="utf-8") as f:
        memories = json.load(f)

    texts = [m["text"] for m in memories]
    embeddings = model.encode(texts).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))

    faiss.write_index(index, index_path)
    print(f"Index built for {astronaut_id}")

if __name__ == "__main__":
    build_index("ASTRO_001")
