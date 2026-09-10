import streamlit as st


def render_styles():
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

            [data-testid="stAppViewContainer"] { background: var(--canvas); }
            [data-testid="stHeader"] { background: transparent; }
            .block-container { max-width: 980px; padding: 2.5rem 1.5rem 7rem; }
            [data-testid="stSidebar"] { background: var(--paper); border-right: 1px solid var(--line); }
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
                color: var(--ink); font-size: 1rem; letter-spacing: 0;
            }
            .app-header {
                align-items: center; display: flex; justify-content: space-between;
                margin: 0 auto 2rem; max-width: 760px;
            }
            .app-brand { color: var(--ink); font-size: 1.25rem; font-weight: 700; }
            .app-status {
                background: var(--accent-soft); border: 1px solid #cce6ec;
                border-radius: 999px; color: var(--accent); font-size: 0.75rem;
                font-weight: 600; padding: 0.35rem 0.7rem;
            }
            .empty-state { color: var(--muted); margin: 15vh auto 0; max-width: 560px; text-align: center; }
            .empty-state h1 { color: var(--ink); font-size: clamp(2rem, 5vw, 3rem); margin-bottom: 0.7rem; }
            div[data-testid="stChatMessage"] {
                border-radius: 14px; margin: 0.8rem auto; max-width: 760px; padding: 0.85rem 1rem;
            }
            div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
                background: var(--paper); border: 1px solid var(--line);
                margin-left: auto; margin-right: 0; max-width: 72%;
            }
            div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
                background: var(--paper); border: 1px solid var(--line);
                margin-left: 0; margin-right: auto; max-width: 82%;
            }
            div[data-testid="stChatInput"] { margin: 0 auto; max-width: 760px; }
            div[data-testid="stChatInput"] textarea {
                background: var(--paper); border: 1px solid #cfd6dd;
                border-radius: 14px; color: var(--ink); min-height: 52px;
            }
            div[data-testid="stChatInput"] textarea:focus {
                border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent);
            }
            .stButton > button { border: 1px solid var(--line); border-radius: 8px; }
            @media (max-width: 640px) {
                .block-container { padding: 1.25rem 0.75rem 6rem; }
                .app-header { margin-bottom: 1rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        '<div class="app-header"><div class="app-brand">Multi-Document AI</div>'
        '<div class="app-status">Groq RAG</div></div>',
        unsafe_allow_html=True,
    )


def render_empty_state():
    st.markdown(
        '<div class="empty-state"><h1>What would you like to explore?</h1>'
        '<p>Ask a normal question or upload PDF documents for grounded answers.</p></div>',
        unsafe_allow_html=True,
    )
