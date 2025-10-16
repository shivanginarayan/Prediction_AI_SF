import os, re, json, csv, hashlib
import requests
import chromadb
from pypdf import PdfReader

# ===== Optional: ML prediction hook =====
try:
    from models.predict import predict_from_features  # loads models/model.pkl
except Exception:
    predict_from_features = None

# =======================
# Config
# =======================
DOCS_DIR = "docs"
COLLECTION_NAME = "my_docs"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "mistral:latest"
OLLAMA_URL = "http://localhost:11434"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5

# Manifest to skip re-embedding unchanged files
DB_DIR = "./chroma_db"
MANIFEST = os.path.join(DB_DIR, "manifest.json")

# =======================
# Utilities
# =======================
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def load_manifest() -> dict:
    try:
        with open(MANIFEST, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_manifest(m: dict) -> None:
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(m, f)

def read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == ".pdf":
        reader = PdfReader(path)
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)
    if ext == ".csv":
        rows = []
        with open(path, newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                rows.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
                if i >= 20000:
                    break
        return "\n".join(rows)
    return ""

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = normalize_ws(text)
    chunks = []
    start = 0
    N = len(text)
    while start < N:
        end = min(N, start + chunk_size)
        chunks.append(text[start:end])
        if end == N:
            break
        start = max(0, end - overlap)
    return chunks

def ollama_embed(texts):
    vectors = []
    for t in texts:
        r = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": t},
            timeout=180,
        )
        r.raise_for_status()
        data = r.json()
        if "embedding" in data:
            vectors.append(data["embedding"])
        elif "data" in data:  # future-proofing
            vectors.extend([d["embedding"] for d in data["data"]])
        elif "error" in data:
            raise RuntimeError(f"Ollama embeddings error: {data['error']}")
        else:
            raise RuntimeError(f"Unexpected /api/embeddings response keys: {list(data.keys())}")
    return vectors

def ollama_chat(messages):
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": CHAT_MODEL, "messages": messages, "stream": False},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]

def ensure_collection():
    client = chromadb.PersistentClient(path=DB_DIR)  # Chroma 1.x
    return client.get_or_create_collection(name=COLLECTION_NAME)

def trim_context(blocks, max_chars=3500):
    out, total = [], 0
    for b in blocks:
        if total + len(b) > max_chars and out:
            break
        out.append(b)
        total += len(b)
    return out

# =======================
# Build / Update Index
# =======================
def build_index():
    col = ensure_collection()
    manifest = load_manifest()
    new_or_updated = []

    for root, _, files in os.walk(DOCS_DIR):
        for fn in files:
            if not fn.lower().endswith((".txt", ".md", ".pdf", ".csv")):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, DOCS_DIR).replace("\\", "/")
            try:
                h = file_hash(path)
            except FileNotFoundError:
                continue
            if manifest.get(rel, {}).get("hash") == h:
                continue

            text = read_file(path)
            if not text.strip():
                continue
            chunks = chunk_text(text)
            if not chunks:
                continue

            embeddings = []
            B = 32
            for i in range(0, len(chunks), B):
                embeddings.extend(ollama_embed(chunks[i:i+B]))

            ids = [f"{rel}::chunk-{i}" for i in range(len(chunks))]
            metadatas = [{"source": rel, "chunk": i} for i in range(len(chunks))]

            col.upsert(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)

            old_count = manifest.get(rel, {}).get("chunks", 0)
            if old_count > len(chunks):
                extra_ids = [f"{rel}::chunk-{i}" for i in range(len(chunks), old_count)]
                try:
                    col.delete(ids=extra_ids)
                except Exception:
                    pass

            manifest[rel] = {"hash": h, "chunks": len(chunks)}
            new_or_updated.append(rel)

    save_manifest(manifest)
    return new_or_updated

# =======================
# Query (RAG)
# =======================
def answer_question(question: str) -> str:
    col = ensure_collection()
    q_embed = ollama_embed([question])[0]
    res = col.query(
        query_embeddings=[q_embed],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    if not res.get("documents") or not res["documents"][0]:
        return "I couldn't find any relevant context in your docs."

    docs = res["documents"][0]
    metas = res["metadatas"][0]

    context_blocks = [f"[source: {m['source']} chunk {m['chunk']}]\n{d}" for d, m in zip(docs, metas)]
    context_blocks = trim_context(context_blocks, max_chars=3500)
    context = "\n\n---\n\n".join(context_blocks)

    system = (
        "You are a strict retrieval-augmented assistant. "
        "Only use the provided context. If the context is insufficient, say so. "
        "Keep answers concise. End with a short 'Sources:' list showing the [source: ...] tags you used."
    )
    user = (
        f"Question: {question}\n\n"
        f"Use ONLY this context to answer. If missing, say so.\n\n"
        f"Context:\n{context}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    answer = ollama_chat(messages)

    used_sources = []
    for line in context.splitlines():
        if line.startswith("[source: "):
            tag = line.split("]")[0] + "]"
            if tag not in used_sources:
                used_sources.append(tag)
    if used_sources and "Sources:" not in answer:
        answer += "\n\nSources:\n" + "\n".join(f"- {s}" for s in used_sources[:TOP_K])

    return answer

# =======================
# Prediction wiring
# =======================
def run_prediction_from_json(json_str: str) -> str:
    if predict_from_features is None:
        return ("Prediction hook not set. Create models/predict.py with "
                "`predict_from_features(features: dict) -> dict`, and make sure models/model.pkl exists.")
    try:
        features = json.loads(json_str)
        if not isinstance(features, dict):
            raise ValueError("JSON must be an object with feature: value pairs.")
    except Exception as e:
        return ('Invalid JSON after /predict.\n'
                'Example: /predict {"B365H":1.9,"B365D":3.2,"B365A":3.9,"AvgH":2.0,"AvgD":3.3,"AvgA":3.6,"PSH":1.95,"PSD":3.4,"PSA":3.7,"Avg>2.5":1.95,"Avg<2.5":1.85,"AHCh":-0.5}\n'
                f'Error: {e}')
    try:
        result = predict_from_features(features)
    except Exception as e:
        return f"Prediction error: {e}"
    return "Prediction:\n" + json.dumps(result, indent=2)

# =======================
# Interactive Chat
# =======================
def chat():
    print("Local RAG chat. Type /exit to quit.")
    print("Commands:")
    print("  • Ask normally for doc Q&A")
    print("  • /predict {JSON}  → run your ML model with features JSON")
    while True:
        try:
            q = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("/exit", "/quit"):
            break
        if q.startswith("/predict"):
            json_part = q[len("/predict"):].strip()
            print("\nPREDICT >", run_prediction_from_json(json_part))
            continue
        try:
            ans = answer_question(q)
            print("\nRAG >", ans)
        except Exception as e:
            print("ERROR:", e)

# =======================
# CLI
# =======================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="(Re)build index from docs/")
    parser.add_argument("--ask", type=str, help="Ask a question against the local index")
    parser.add_argument("--chat", action="store_true", help="Interactive local chat (RAG)")
    parser.add_argument("--predict", type=str, help='Run prediction with a JSON string, e.g. --predict "{\"B365H\":1.9, ...}"')
    args = parser.parse_args()

    if args.build:
        updated = build_index()
        print("Indexed/updated docs:", updated if updated else "(none)")

    if args.ask:
        print("\nQ:", args.ask)
        print("\nA:", answer_question(args.ask))

    if args.predict:
        print(run_prediction_from_json(args.predict))

    if args.chat:
        chat()
