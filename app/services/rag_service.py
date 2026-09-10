import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import EMBEDDING_MODEL, GROQ_MODEL


def build_grounded_refusal_message():
    return (
        "I can only answer questions using the uploaded PDFs. "
        "I couldn’t find relevant information in the uploaded documents for that question."
    )


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def uploaded_file_payloads(files):
    return tuple((file.name, file.getvalue()) for file in files)


@st.cache_resource(show_spinner="Processing your documents...")
def build_vectorstore(file_payloads):
    all_docs = []

    for filename, contents in file_payloads:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(contents)
                temp_path = temp_file.name

            documents = PyPDFLoader(temp_path).load()
            for document in documents:
                document.metadata["source"] = filename
                document.metadata["page"] = document.metadata.get("page", 0) + 1
            all_docs.extend(documents)
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    if not all_docs or not any(document.page_content.strip() for document in all_docs):
        raise ValueError("The uploaded PDFs do not contain readable text.")

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=900,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(all_docs)
    digest = hashlib.sha256(
        b"".join(
            filename.encode("utf-8") + b"\0" + contents
            for filename, contents in file_payloads
        )
    ).hexdigest()[:16]

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=f"documents_{digest}",
    )


def build_rag_chain(vectorstore, api_key):
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1800,
        groq_api_key=api_key,
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 24, "lambda_mult": 0.5},
    )
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Rewrite the latest user question as a standalone question using the conversation history. Do not answer the question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful document research assistant. Answer using only the supplied document context. "
                "First identify the relevant evidence, then explain the answer clearly and completely. "
                "If the context does not support an answer, say so instead of guessing. Never invent facts or citations. "
                "If no relevant information is present in the document context, answer exactly with: 'I can only answer questions using the uploaded PDFs. I couldn’t find relevant information in the uploaded documents for that question.' "
                "Give a detailed, well-structured answer with key points and examples when the context supports them. "
                "Cite source names and page numbers inline when possible.\n\nDocument context:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return create_retrieval_chain(
        history_aware_retriever,
        create_stuff_documents_chain(llm, answer_prompt),
    )


def stream_rag_chain(rag_chain, query, chat_history):
    context_documents = []
    answer_parts = []

    for chunk in rag_chain.stream({"input": query, "chat_history": chat_history}):
        if chunk.get("context"):
            context_documents.extend(chunk["context"])
        if chunk.get("answer"):
            answer_parts.append(chunk["answer"])

    if not context_documents:
        yield {"answer": build_grounded_refusal_message()}
        return

    if context_documents:
        yield {"context": context_documents}

    final_answer = "".join(answer_parts).strip()
    if final_answer:
        yield {"answer": final_answer}
    else:
        yield {"answer": build_grounded_refusal_message()}
