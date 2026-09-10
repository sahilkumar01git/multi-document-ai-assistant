import streamlit as st
from app.config import APP_TITLE, get_groq_api_key
from app.services.chat_service import (
    build_chat_model,
    stream_general_chat,
    to_langchain_messages,
)
from app.services.rag_service import (
    build_grounded_refusal_message,
    build_rag_chain,
    build_vectorstore,
    stream_rag_chain,
    uploaded_file_payloads,
)
from app.ui.sidebar import render_sidebar
from app.ui.styles import render_empty_state, render_header, render_styles


def render_sources(source_documents):
    if not source_documents:
        return

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


def render_conversation():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def run():
    st.set_page_config(page_title=APP_TITLE, page_icon="PDF", layout="wide")
    render_styles()
    st.title(APP_TITLE)
    render_header()

    uploaded_files = render_sidebar()
    if not uploaded_files:
        render_empty_state()

    try:
        secrets = st.secrets
    except (FileNotFoundError, KeyError):
        secrets = None
    api_key = get_groq_api_key(secrets)
    if not api_key:
        st.error("Set GROQ_API_KEY in .env or Streamlit secrets before asking a question.")
        st.stop()

    rag_chain = None
    chat_model = None
    if uploaded_files:
        try:
            vectorstore = build_vectorstore(uploaded_file_payloads(uploaded_files))
            rag_chain = build_rag_chain(vectorstore, api_key)
        except Exception as error:
            st.error(f"Could not prepare the documents: {error}")
            st.stop()
    else:
        chat_model = build_chat_model(api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    render_conversation()

    query = st.chat_input("Ask a question about your documents")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    source_documents = []
    try:
        chat_history = to_langchain_messages(st.session_state.messages[:-1])
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if rag_chain:
                    def answer_stream():
                        streamed_answer = ""
                        for chunk in stream_rag_chain(rag_chain, query, chat_history):
                            if chunk.get("context"):
                                source_documents.extend(chunk["context"])
                            if chunk.get("answer"):
                                streamed_answer = chunk["answer"]
                                yield chunk["answer"]

                        if not source_documents and not streamed_answer:
                            yield build_grounded_refusal_message()

                    answer = st.write_stream(answer_stream())
                else:
                    answer = st.write_stream(
                        stream_general_chat(chat_model, query, chat_history)
                    )
            st.session_state.messages.append({"role": "assistant", "content": answer})
            render_sources(source_documents)
    except Exception as error:
        st.error(f"Groq request failed: {error}")
        st.stop()
