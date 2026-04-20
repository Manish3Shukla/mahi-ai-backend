import os
import numpy as np
import pandas as pd
import faiss
import kagglehub

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# -----------------------------
# 🚀 INIT FASTAPI
# -----------------------------
app = FastAPI(title="MAHI Geeta AI")

# -----------------------------
# 🔹 LOAD SMALL MODEL (LOW MEMORY)
# -----------------------------
print("🔄 Loading embedding model...")

model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

print("✅ Model loaded")

# -----------------------------
# 🔹 LOAD DATASET
# -----------------------------
print("🔄 Loading dataset...")

path = kagglehub.dataset_download("a2m2a2n2/bhagwad-gita-dataset")
files = os.listdir(path)

df = pd.read_csv(os.path.join(path, files[0]))

# Flexible columns
df["text"] = df[df.columns[0]]
df["translation"] = df[df.columns[1]]

# Clean
df["translation"] = df["translation"].astype(str).str.lower().str.strip()
df = df[df["translation"] != ""].dropna().reset_index(drop=True)

# 🔥 IMPORTANT: reduce dataset for memory
df = df.sample(100)

print("✅ Dataset ready")

# -----------------------------
# 🔹 CREATE EMBEDDINGS
# -----------------------------
print("🔄 Creating embeddings...")

embeddings = model.encode(
    df["translation"].tolist()
)

embeddings = np.array(embeddings).astype("float32")

# Normalize
faiss.normalize_L2(embeddings)

# -----------------------------
# 🔹 FAISS INDEX
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("✅ FAISS ready")

# -----------------------------
# 🔹 REQUEST MODEL
# -----------------------------
class UserInput(BaseModel):
    text: str

# -----------------------------
# 🔹 SEARCH FUNCTION
# -----------------------------
def search_shloka(query):
    query_vec = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, 1)

    result = df.iloc[indices[0][0]]

    return {
        "shloka": result["text"],
        "meaning": result["translation"]
    }

# -----------------------------
# 🔹 ROUTES
# -----------------------------
@app.get("/")
def home():
    return {"message": "Geeta AI running 🚀"}

@app.post("/analyze")
def analyze(input: UserInput):

    text = input.text.strip()

    if not text:
        return {"status": "error", "message": "Empty input"}

    result = search_shloka(text)

    return {
        "status": "success",
        "input": text,
        "result": result
    }

# -----------------------------
# 🔹 LOCAL RUN
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
