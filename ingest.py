import os, glob
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from langchain_community.embeddings import SentenceTransformerEmbeddings
from vectordb import get_vector_store

load_dotenv()

DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def load_docs() -> List:
    paths = []
    paths.extend(glob.glob(os.path.join(DATA_DIR, "**/*.pdf"), recursive=True))
    paths.extend(glob.glob(os.path.join(DATA_DIR, "**/*.txt"), recursive=True))
    paths.extend(glob.glob(os.path.join(DATA_DIR, "**/*.md"), recursive=True))
    paths.extend(glob.glob(os.path.join(DATA_DIR, "**/*.docx"), recursive=True))

    docs = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(p)
            elif ext == ".txt":
                loader = TextLoader(p, encoding="utf-8")
            elif ext == ".md":
                loader = UnstructuredMarkdownLoader(p)
            elif ext == ".docx":
                loader = Docx2txtLoader(p)
            else:
                continue
            docs.extend(loader.load())
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    return docs

def main():
    print("[1/3] Loading documents from ./data ...")
    docs = load_docs()
    if not docs:
        print("No documents found. Add files to ./data and re-run.")
        return

    print(f"Loaded {len(docs)} raw docs. Splitting ...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("[2/3] Building embeddings (all-MiniLM-L6-v2) ...")
    embeddings = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


    print("[3/3] Upserting into vector store ...")
    store = get_vector_store(embeddings)
    store.add_documents(chunks)
    if hasattr(store, "persist"):
        store.persist()

    print("Done. You can now run:  streamlit run app.py")

if __name__ == "__main__":
    main()
