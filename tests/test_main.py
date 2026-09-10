from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_SOURCE = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")


def test_uses_supported_groq_model():
    assert 'GROQ_MODEL = "openai/gpt-oss-120b"' in MAIN_SOURCE


def test_uses_modern_retrieval_chain():
    assert "create_history_aware_retriever" in MAIN_SOURCE
    assert "create_retrieval_chain" in MAIN_SOURCE
    assert "ConversationalRetrievalChain" not in MAIN_SOURCE


def test_cleans_up_temporary_pdf_files():
    assert "Path(temp_path).unlink(missing_ok=True)" in MAIN_SOURCE


def test_grounding_and_sources_are_enabled():
    assert "Never invent facts" in MAIN_SOURCE
    assert 'chunk.get("context")' in MAIN_SOURCE


def test_retrieval_uses_diversity_and_normalized_embeddings():
    assert 'search_type="mmr"' in MAIN_SOURCE
    assert '"fetch_k": 24' in MAIN_SOURCE
    assert '"normalize_embeddings": True' in MAIN_SOURCE
