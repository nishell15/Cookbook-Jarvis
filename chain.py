import os
from typing import Any, Dict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

load_dotenv()

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are Jarvis, a concise, helpful assistant for my files. Cite file names when relevant.")
MODEL = os.getenv("OLLAMA_MODEL", "llama3")

def build_rag_chain(retriever):
    llm = Ollama(model=MODEL, temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
    ])
    chain = (
        RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
