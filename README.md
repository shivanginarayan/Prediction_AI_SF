# Offline Document Q&A + On-device KNN Predictions (Ollama + Chroma)
_Local, privacy-preserving chat over your own files + predictions from a trained ML model._  
**Project:** San Francisco State University (SFSU)

---

## What this repo contains
- **RAG chat (offline):** Ask questions about your local files; answers are grounded in your docs only.
- **On-device predictions:** Run a **KNN** model directly from chat using `/predict {JSON}`.
- **File types:** `.txt`, `.md`, `.pdf`, `.csv`
- **100% local:** Uses **Ollama** for LLM inference and **Chroma** for vector search. No cloud calls.

---

## Quickstart

### 0) Prereqs
- **Python** 3.10+  
- **Ollama** installed and running (Windows/macOS/Linux): <https://ollama.com/>

### 1) Install dependencies
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt
2) Pull local models (via Ollama)
bash
ollama pull mistral:latest
ollama pull nomic-embed-text
3) Index your docs
Place your files in ./docs (kept out of git), then:

bash

python rag.py --build
4) Chat locally
bash

python rag.py --chat
# Ask questions about your docs
Use the prediction model in chat
Train once (creates models/model.pkl)
bash

python KNN.py
Then call it from chat
Inside the chat prompt:

bash

/predict {"B365H":1.9,"B365D":3.2,"B365A":3.9,"AvgH":2.0,"AvgD":3.3,"AvgA":3.6,"PSH":1.95,"PSD":3.4,"PSA":3.7,"Avg>2.5":1.95,"Avg<2.5":1.85,"AHCh":-0.5}
JSON keys such as "Avg>2.5" / "Avg<2.5" must be quoted.

One-off prediction (CLI)
bash

python rag.py --predict "{\"B365H\":1.9,\"B365D\":3.2,\"B365A\":3.9,\"AvgH\":2.0,\"AvgD\":3.3,\"AvgA\":3.6,\"PSH\":1.95,\"PSD\":3.4,\"PSA\":3.7,\"Avg>2.5\":1.95,\"Avg<2.5\":1.85,\"AHCh\":-0.5}"
Files of interest
rag.py — local RAG app (Ollama + Chroma), interactive chat, /predict {JSON} support.

KNN.py — trains a KNN classifier and saves models/model.pkl.

models/predict.py — loads the saved model and returns predictions/probabilities.

Configuration (edit in rag.py)
CHAT_MODEL (default: mistral:latest)

EMBED_MODEL (default: nomic-embed-text)

Retrieval knobs:
CHUNK_SIZE = 900, CHUNK_OVERLAP = 150, TOP_K = 5

Data folders:
DOCS_DIR = "docs", DB_DIR = "./chroma_db"

Repo hygiene (already set)
docs/, chroma_db/, and models/model.pkl are gitignored (keep private data local).

Add safe sample files to docs/ if you want to demo.

Troubleshooting
Ollama port already in use

Message like bind 127.0.0.1:11434 ⇒ server is already running (OK). Don’t run ollama serve again.

Quick check: curl http://localhost:11434/api/tags

“No trained model found” when using /predict

Run python KNN.py to create models/model.pkl.

PDF text extraction is empty

The PDF might be scanned images. Add OCR (e.g., pytesseract + Pillow) as a future enhancement.

Embedding/connection errors

Make sure ollama pull nomic-embed-text completed.

Verify Ollama is running: ollama list (should show mistral:latest and nomic-embed-text).

Project structure
pgsql

.
├── rag.py
├── KNN.py
├── models/
│   ├── predict.py
│   └── model.pkl          # created by KNN.py (ignored by git)
├── docs/                  # your local files (ignored by git)
├── chroma_db/             # vector index (ignored by git)
├── requirements.txt
└── .gitignore
