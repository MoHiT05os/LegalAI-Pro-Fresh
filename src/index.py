# src/index.py
from pathlib import Path
from typing import List
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_PERSIST_DIR = Path.cwd() / "chroma_db"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_chroma_index(
    docs: List[Document],
    collection_name: str = "legal",
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
):
    persist_directory = str(Path(persist_directory))
    embeddings = get_embeddings()

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    db.persist()
    print(f"Chroma index persisted to {persist_directory} (collection: {collection_name})")
    return db

def load_chroma(
    collection_name: str = "legal",
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
):
    persist_directory = str(Path(persist_directory))
    embeddings = get_embeddings()
    db = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    return db

if __name__ == "__main__":
    print("Index module. Use create_chroma_index(docs) from your scripts.")
