# src/ask_cli.py

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.index import load_chroma
from src.prompts import PROMPT_TEMPLATE

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA


def build_qa_chain():
    db = load_chroma(
        collection_name="legal",
        persist_directory=Path.cwd() / "chroma_db",
    )

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        input_key="query",
        output_key="result",
    )

    return qa_chain


def ask(question: str):
    qa = build_qa_chain()

    print(f"\n🤖 Legal Query: {question}\n")

    response = qa.invoke({"query": question})

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
    if len(sys.argv) < 2:
        print('Usage: python -m src.ask_cli "your question"')
        sys.exit(1)

    user_question = " ".join(sys.argv[1:])
    ask(user_question)
