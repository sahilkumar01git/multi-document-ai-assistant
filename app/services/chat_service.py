import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import GROQ_MODEL


@st.cache_resource
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


def stream_general_chat(chat_model, query, chat_history):
    return (
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
