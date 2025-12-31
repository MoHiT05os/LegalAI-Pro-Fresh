# src/ask_cli.py

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

from src.index import load_chroma
from src.prompts import PROMPT_TEMPLATE

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA


def build_qa_chain():
    """
    Build a RetrievalQA chain over the 'legal' Chroma collection.
    """

    # 1. Load Chroma DB (must already be built and persisted)
    db = load_chroma(
        collection_name="legal",
        persist_directory=Path.cwd() / "chroma_db",
    )

    # 2. Create retriever
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # 3. LLM configuration
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )

    # 4. Prompt template (PROMPT_TEMPLATE must have {context} and {question})
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE,
    )

    # 5. RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        input_key="query",    # input key used when calling .invoke(...)
        output_key="result",  # key where final answer will be stored
    )

    return qa_chain


def ask(question: str):
    """
    Run a single legal question through the QA chain and print answer + sources.
    """
    qa = build_qa_chain()

    print(f"\n🤖 Legal Query: {question}\n")

    # Must match input_key above ("query")
    response = qa.invoke({"query": question})

    # Response keys: "result" (string answer), "source_documents" (list of docs)
    answer = response.get("result")
    source_docs = response.get("source_documents", [])

    print("\n=== ⚖ ANSWER ===\n")
    print(answer or "No answer returned.")

    print("\n=== 📚 SOURCES ===\n")
    if not source_docs:
        print("No source documents returned.\n")
        return

    for i, doc in enumerate(source_docs, start=1):
        meta = doc.metadata or {}
        snippet = (doc.page_content or "")[:350].replace("\n", " ")
        print(f"{i}. {meta.get('source', 'unknown')} (page {meta.get('page', '–')})")
        print(f'   "{snippet}..."\n')


if __name__ == "__main__":
    # Usage: python -m src.ask_cli "your question"
    if len(sys.argv) < 2:
        print('Usage: python -m src.ask_cli "your question"')
        sys.exit(1)

    user_question = " ".join(sys.argv[1:])
    ask(user_question)
