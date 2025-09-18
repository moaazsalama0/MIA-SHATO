"""
Integrated RAG + Ollama API server (no local validation).
Send output to external validator service instead.

Run:
    uvicorn api_rag:app --reload --host 127.0.0.1 --port 9000
"""

import os, json, glob, requests
from typing import Dict, Any, List
from collections import Counter

import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------- Config ----------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_COLLECTION = "shato_examples"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4
OLLAMA_MODEL = "gemma3:270m"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://llm-service:11434")



# ---------------- Load dataset ----------------
def load_samples(data_dir: str) -> List[Dict[str, Any]]:
    samples = []
    for path in glob.glob(os.path.join(data_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    samples.extend(data)
            except Exception as e:
                print(f"Error reading {path}: {e}")
    return samples

print("Loading data...")
samples = load_samples(DATA_DIR)
print("Loaded samples:", len(samples))
print("Category counts:", Counter(s.get("category") for s in samples))

valid_samples = [s for s in samples if isinstance(s.get("expected_output"), dict)]
print("Valid-looking samples:", len(valid_samples))

# ---------------- Build Chroma index ----------------
print("Preparing embeddings and building Chroma index...")
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

client = chromadb.Client()
if CHROMA_COLLECTION in [c.name for c in client.list_collections()]:
    client.delete_collection(CHROMA_COLLECTION)
collection = client.create_collection(CHROMA_COLLECTION)

def doc_text(example):
    eo = example.get("expected_output", {})
    return json.dumps({
        "input": example.get("user_input"),
        "response": eo.get("response"),
        "command": eo.get("command"),
        "command_params": eo.get("command_params"),
    }, ensure_ascii=False)

docs = [doc_text(s) for s in valid_samples]
ids = [f"ex{i}" for i in range(len(docs))]
metadatas = [{"category": s.get("category")} for s in valid_samples]
embs = embed_model.encode(docs, show_progress_bar=True, convert_to_numpy=True)

collection.add(documents=docs, ids=ids, metadatas=metadatas, embeddings=embs)
print(f"Indexed {len(docs)} docs into Chroma collection '{CHROMA_COLLECTION}'")

# ---------------- FastAPI ----------------
app = FastAPI(title="SHATO RAG API", version="2.1 (no validation)")

class QueryRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: QueryRequest):
    prompt = req.message.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty message")

    # Retrieve top-k examples
    q_emb = embed_model.encode([prompt], convert_to_numpy=True)[0]
    res = collection.query(query_embeddings=[q_emb], n_results=TOP_K, include=["documents"])
    retrieved_docs = res["documents"][0] if res and "documents" in res else []
    context = "\n\n".join(retrieved_docs)

    # Build system + context prompt
    system_msg = (
        """You are SHATO's command brain. 
Always reply in pure JSON only.
Rules:
- Valid commands: ["move_to", "rotate", "start_patrol"]
- If unrelated: command=null, command_params=null, natural-language response
- JSON format: {"response": "string", "command": "...", "command_params": {...} or null}
"""
    )

    full_prompt = f"{system_msg}\nContext:\n{context}\n\nUser: {prompt}\nAssistant:"

    # Call Ollama
    try:
        payload = {"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False}
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        raw_output = data.get("response", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {e}")

    # Parse JSON safely
    parsed = None
    error_msg = None
    try:
        raw = raw_output.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json\n", "", 1).replace("json\r\n", "", 1)
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            parsed = json.loads(raw[start:end+1])
        else:
            error_msg = "Could not find JSON in LLM output"
    except Exception as e:
        error_msg = f"JSON parse error: {e}"

    llm_output = parsed if parsed else {"response": "", "command": None, "command_params": None}

    return {
        "input": prompt,
        "retrieved": retrieved_docs,
        "llm_output": llm_output,
        "model_raw": raw_output,
        "error": error_msg
    }

@app.get("/")
def root():
    return {"status": "ok", "samples_indexed": len(docs), "collection": CHROMA_COLLECTION}
