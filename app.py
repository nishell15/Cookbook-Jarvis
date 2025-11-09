import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from vectordb import get_vector_store, as_retriever
from chain import build_rag_chain

load_dotenv()

st.set_page_config(page_title="Jarvis — RAG Assistant", page_icon="🤖")
st.title("🍳 Student Cookbook AI — RAG Demo")
st.caption("Ollama + LangChain + Chroma. Answers grounded in your PDF.")

# lazy init
@st.cache_resource(show_spinner=False)
def _bootstrap():
    embeddings = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    store = get_vector_store(embeddings)
    retriever = as_retriever(store, k=4)
    chain = build_rag_chain(retriever)
    return chain

chain = _bootstrap()

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

prompt = st.chat_input("Ask about your documents...")
if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = chain.invoke(prompt)
            except Exception as e:
                answer = f"Error: {e}\n\nMake sure Ollama is running and you've pulled the model (e.g., `ollama pull llama3`)."
        st.markdown(answer)
        st.session_state.history.append(("assistant", answer))
