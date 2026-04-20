import os
import numpy as np
import pandas as pd
import faiss
import kagglehub

from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -----------------------------
# 🚀 INIT FASTAPI
# -----------------------------
app = FastAPI(title="MAHI AI Backend")

# -----------------------------
# 🔹 LOAD MODELS (ON START)
# -----------------------------
print("🔄 Loading models...")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

print("✅ Models loaded")

# -----------------------------
# 🔹 LOAD DATASET
# -----------------------------
print("🔄 Loading dataset...")

path = kagglehub.dataset_download("a2m2a2n2/bhagwad-gita-dataset")
files = os.listdir(path)

df = pd.read_csv(os.path.join(path, files[0]))

# Flexible column mapping
df["text"] = df[df.columns[0]]
df["translation"] = df[df.columns[1]]

def clean_text(text):
    return str(text).lower().strip()

df["translation"] = df["translation"].apply(clean_text)
df = df[df["translation"] != ""].dropna().reset_index(drop=True)

print("✅ Dataset ready")

# -----------------------------
# 🔹 CREATE EMBEDDINGS
# -----------------------------
print("🔄 Creating embeddings...")

embeddings = embed_model.encode(
    df["translation"].tolist(),
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

# -----------------------------
# 🔹 FAISS INDEX
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("✅ FAISS ready")

# -----------------------------
# 🔹 EMOTION MAP
# -----------------------------
emotion_to_category = {
    "anger": "Self Control",
    "fear": "Bhakti Yoga",
    "joy": "Gratitude",
    "sadness": "Detachment",
    "love": "Devotion",
    "surprise": "Awareness"
}

# -----------------------------
# 🔹 REQUEST MODEL
# -----------------------------
class UserInput(BaseModel):
    text: str

# -----------------------------
# 🔹 FUNCTIONS
# -----------------------------
def detect_emotion(text):
    try:
        result = emotion_classifier(text)[0]
        return result["label"].lower()
    except:
        return "neutral"

def retrieve_shlokas(query, top_k=3):
    query_vec = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, top_k)

    results = []
    for i in indices[0]:
        results.append({
            "shloka": df.iloc[i]["text"],
            "meaning": df.iloc[i]["translation"]
        })

    return results

def generate_advice(emotion, category):
    return (
        f"You are experiencing {emotion}. "
        f"The Bhagavad Gita suggests following {category}. "
        f"Apply this wisdom in your actions and thoughts."
    )

# -----------------------------
# 🔹 API ROUTES
# -----------------------------
@app.get("/")
def home():
    return {"message": "MAHI AI is running 🚀"}

@app.post("/analyze")
def analyze(input: UserInput):

    user_input = input.text.strip()

    if not user_input:
        return {"status": "error", "message": "Please enter a valid problem"}

    emotion = detect_emotion(user_input)
    category = emotion_to_category.get(emotion, "General")

    shlokas = retrieve_shlokas(user_input, top_k=3)
    advice = generate_advice(emotion, category)

    return {
        "status": "success",
        "emotion": emotion,
        "category": category,
        "results": shlokas,
        "advice": advice
    }

# -----------------------------
# 🔹 RUN SERVER (LOCAL)
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
