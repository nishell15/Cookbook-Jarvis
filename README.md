# Build Your Own Jarvis — RAG Assistant (Cookbook)
This mini-cookbook spins up a personal AI assistant that:
- runs a **self-hosted LLM** via **Ollama** (default: `llama3`),
- ingests your local **./data** files into a **vector database** (Chroma by default, Pinecone optional),
- answers questions using **retrieval‑augmented generation (RAG)**,
- exposes a simple **Streamlit chat UI**.

> Designed to be finished in ~40 mins for an assignment/demo.

---

## 0) Prereqs
- **Python 3.10+**
- **Ollama** installed & running → https://ollama.com (then run `ollama pull llama3`)
- (Optional) **Pinecone** account + API key if you prefer cloud vector DB.

## 1) Setup
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Add your knowledge
Drop PDFs/TXTs/MD/DOCs into `./data`. (You can add more later.)

## 3) Configure (optional)
Create `.env` in the project root if needed:
```
# Use Pinecone instead of local Chroma (True/False). Defaults to False.
USE_PINECONE=False

# When USE_PINECONE=True
PINECONE_API_KEY=YOUR_KEY
PINECONE_ENV=YOUR_ENV   # e.g., us-east-1-aws
PINECONE_INDEX_NAME=jarvis-index

# LLM model to use with Ollama
OLLAMA_MODEL=llama3
SYSTEM_PROMPT=You are Jarvis, a concise, helpful assistant for my files. Cite file names when relevant.
```

## 4) Ingest
```bash
python ingest.py
```
This reads `./data`, builds embeddings, and populates Chroma or Pinecone.

## 5) Run the chat UI
```bash
streamlit run app.py
```
Open the local URL and start chatting.

## 6) Notes
- By default we use **`sentence-transformers/all-MiniLM-L6-v2`** (dim=384) for embeddings → fast and Pinecone-friendly.
- To switch LLMs, change `OLLAMA_MODEL` in `.env` and run: `ollama pull <model>`.
- Everything is kept small and clear for an assignment. Extend as you like: tools, agents, structured output, file upload, etc.

## 7) Project tree
```
jarvis_cookbook/
  ├─ app.py              # Streamlit chat UI
  ├─ chain.py            # RAG chain (LangChain)
  ├─ ingest.py           # Parse -> chunk -> embed -> upsert
  ├─ vectordb.py         # Chroma (local) or Pinecone (cloud)
  ├─ requirements.txt
  ├─ README.md
  └─ data/               # put your docs here
```
