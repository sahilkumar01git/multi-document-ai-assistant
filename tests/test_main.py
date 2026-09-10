from app.config import EMBEDDING_MODEL, GROQ_MODEL
from app.services.chat_service import to_langchain_messages
from app.services.rag_service import build_grounded_refusal_message


def test_uses_supported_models():
    assert EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert GROQ_MODEL == "openai/gpt-oss-120b"


def test_conversation_messages_are_converted():
    messages = to_langchain_messages(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    )
    assert [message.content for message in messages] == ["Hello", "Hi"]
    assert messages[0].type == "human"
    assert messages[1].type == "ai"


def test_no_pdf_context_uses_refusal_message():
    message = build_grounded_refusal_message()
    assert "uploaded PDFs" in message
    assert "couldn’t find" in message.lower()
