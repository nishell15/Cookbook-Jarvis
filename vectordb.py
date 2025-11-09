import os
from typing import Tuple, List
from dotenv import load_dotenv

load_dotenv()

USE_PINECONE = os.getenv("USE_PINECONE", "False").lower() == "true"

def get_vector_store(embeddings, index_name: str = "jarvis-index"):
    if USE_PINECONE:
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        spec = ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_ENV", "us-east-1"))
        # Create index if missing
        if index_name not in [i.name for i in pc.list_indexes()]:
            pc.create_index(
                name=index_name,
                dimension=embeddings.client.get_sentence_embedding_dimension(),
                metric="cosine",
                spec=spec,
            )
        index = pc.Index(index_name)
        # langchain vectorstore wrapper
        from langchain_community.vectorstores import Pinecone as LC_Pinecone
        return LC_Pinecone(index, embeddings.embed_query, "text", namespace="default")
    else:
        from langchain_community.vectorstores import Chroma
        persist_dir = os.path.join(os.getcwd(), ".chroma")
        return Chroma(collection_name=index_name, embedding_function=embeddings, persist_directory=persist_dir)

def as_retriever(store, k: int = 4):
    return store.as_retriever(search_kwargs={"k": k})
