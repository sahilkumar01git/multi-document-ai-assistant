import streamlit as st


def render_sidebar():
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

    if uploaded_files:
        st.sidebar.success(f"{len(uploaded_files)} document(s) ready")

    return uploaded_files
