import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

APP_TITLE = "Multi-Document AI Assistant"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

st.set_page_config(page_title=APP_TITLE, page_icon="PDF", layout="wide")
st.markdown(
    """
    <style>
        :root {
            --ink: #1f2933;
            --muted: #697586;
            --line: #e5e7eb;
            --paper: #ffffff;
            --canvas: #f7f8fa;
            --accent: #176b87;
            --accent-soft: #e7f3f6;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--canvas);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 980px;
            padding: 2.5rem 1.5rem 7rem;
        }

        [data-testid="stSidebar"] {
            background: var(--paper);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
            color: var(--ink);
            font-size: 1rem;
            letter-spacing: 0;
        }

        .app-header {
            align-items: center;
            display: flex;
            justify-content: space-between;
            margin: 0 auto 2rem;
            max-width: 760px;
        }

        .app-brand {
            color: var(--ink);
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: 0;
        }

        .app-status {
            background: var(--accent-soft);
            border: 1px solid #cce6ec;
            border-radius: 999px;
            color: var(--accent);
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.35rem 0.7rem;
        }

        .empty-state {
            color: var(--muted);
            margin: 15vh auto 0;
            max-width: 560px;
            text-align: center;
        }

        .empty-state h1 {
            color: var(--ink);
            font-size: clamp(2rem, 5vw, 3rem);
            margin-bottom: 0.7rem;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 14px;
            margin: 0.8rem auto;
            max-width: 760px;
            padding: 0.85rem 1rem;
        }

        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background: var(--paper);
            border: 1px solid var(--line);
            margin-left: auto;
            margin-right: 0;
            max-width: 72%;
        }

        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            background: var(--paper);
            border: 1px solid var(--line);
            margin-left: 0;
            margin-right: auto;
            max-width: 82%;
        }

        div[data-testid="stChatInput"] {
            margin: 0 auto;
            max-width: 760px;
        }

        div[data-testid="stChatInput"] textarea {
            background: var(--paper);
            border: 1px solid #cfd6dd;
            border-radius: 14px;
            color: var(--ink);
            min-height: 52px;
        }

        div[data-testid="stChatInput"] textarea:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 1px var(--accent);
        }

        .stButton > button {
            border: 1px solid var(--line);
            border-radius: 8px;
        }

        @media (max-width: 640px) {
            .block-container {
                padding: 1.25rem 0.75rem 6rem;
            }

            .app-header {
                margin-bottom: 1rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-header"><div class="app-brand">Multi-Document AI</div>'
    '<div class="app-status">Groq RAG</div></div>',
    unsafe_allow_html=True,
)


def get_groq_api_key():
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
    if api_key:
        return api_key

    try:
        return st.secrets.get("GROQ_API_KEY") or st.secrets.get("groq_api_key")
    except (FileNotFoundError, KeyError):
        return None


def uploaded_file_payloads(files):
    return tuple((file.name, file.getvalue()) for file in files)


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


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
            (
                "system",
                "Rewrite the latest user question as a standalone question using the "
                "conversation history. Do not answer the question.",
            ),
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
                "You are a careful document research assistant. Answer using only the "
                "supplied document context. First identify the relevant evidence, then "
                "explain the answer clearly and completely. If the context does not "
                "support an answer, say so instead of guessing. Never invent facts or "
                "citations. Give a detailed, well-structured answer with key points "
                "and examples when the context supports them. Cite source names and "
                "page numbers inline when possible.\n\n"
                "Document context:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    answer_chain = create_stuff_documents_chain(llm, answer_prompt)
    return create_retrieval_chain(history_aware_retriever, answer_chain)


def build_chat_model(api_key):
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.4,
        max_tokens=1800,
        groq_api_key=api_key,
    )


def to_langchain_messages(messages):
    return [
        HumanMessage(content=message["content"])
        if message["role"] == "user"
        else AIMessage(content=message["content"])
        for message in messages
    ]


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not uploaded_files:
    st.markdown(
        '<div class="empty-state"><h1>What would you like to explore?</h1>'
        '<p>Ask a normal question or upload PDF documents for grounded answers.</p></div>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.success(f"{len(uploaded_files)} document(s) ready")

api_key = get_groq_api_key()
if not api_key:
    st.error("Set GROQ_API_KEY in .env or Streamlit secrets before asking a question.")
    st.stop()

rag_chain = None
chat_model = None
if uploaded_files:
    try:
        payloads = uploaded_file_payloads(uploaded_files)
        vectorstore = build_vectorstore(payloads)
        rag_chain = build_rag_chain(vectorstore, api_key)
    except Exception as error:
        st.error(f"Could not prepare the documents: {error}")
        st.stop()
else:
    chat_model = build_chat_model(api_key)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a question about your documents")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    source_documents = []
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_history = to_langchain_messages(st.session_state.messages[:-1])
                if rag_chain:
                    def rag_stream():
                        global source_documents
                        for chunk in rag_chain.stream(
                            {"input": query, "chat_history": chat_history}
                        ):
                            if chunk.get("context"):
                                source_documents = chunk["context"]
                            if chunk.get("answer"):
                                yield chunk["answer"]

                    answer = st.write_stream(rag_stream())
                else:
                    answer = st.write_stream(
                        chunk.content
                        for chunk in chat_model.stream(
                            [
                                SystemMessage(
                                    content="You are a friendly, helpful general-purpose assistant. "
                                    "Answer normal questions naturally and conversationally."
                                ),
                                *chat_history,
                                HumanMessage(content=query),
                            ]
                        )
                        if chunk.content
                    )

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
            if source_documents:
                with st.expander("Sources"):
                    shown_sources = set()
                    for document in source_documents:
                        source = document.metadata.get("source", "Unknown document")
                        page = document.metadata.get("page", "N/A")
                        source_key = (source, page)
                        if source_key in shown_sources:
                            continue
                        shown_sources.add(source_key)
                        st.markdown(f"- **{source}**, page {page}")
    except Exception as error:
        st.error(f"Groq request failed: {error}")
        st.stop()
