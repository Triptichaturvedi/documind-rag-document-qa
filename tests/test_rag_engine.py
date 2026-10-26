import gc
import tempfile

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import langchain_openai
from src.rag_engine import (
    ModelProviderError,
    answer_question,
    build_vector_store,
    load_uploaded_file,
    open_vector_store,
    split_documents,
)


class FakeVectorStore:
    def __init__(self, score=0.8):
        self.score = score
        self.document = Document(
            page_content="Project Falcon has code PF-204.",
            metadata={"source": "test.txt", "page_or_section": 0, "chunk_id": "CHUNK-00001"},
        )

    def similarity_search_with_relevance_scores(self, question, k):
        return [(self.document, self.score)]


def test_missing_openai_key_is_actionable():
    with pytest.raises(ModelProviderError, match="OPENAI_API_KEY"):
        answer_question("What is the code?", FakeVectorStore(), provider="OpenAI")


def test_configured_provider_is_used_and_sources_are_returned(monkeypatch):
    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return RunnableLambda(
            lambda _: AIMessage(
                content="Project Falcon has code PF-204. [Source: test.txt, 0, CHUNK-00001]"
            )
        )

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", fake_chat_openai)
    answer, sources = answer_question(
        "What is the code?",
        FakeVectorStore(),
        provider="OpenAI",
        model_name="gpt-4o-mini",
        openai_api_key="test-key",
    )

    assert captured["model"] == "gpt-4o-mini"
    assert captured["api_key"] == "test-key"
    assert "PF-204" in answer
    assert sources[0][0].metadata["source"] == "test.txt"
    assert sources[0][1] == 0.8


def test_irrelevant_question_refuses_without_calling_provider():
    answer, sources = answer_question(
        "What is the capital of France?",
        FakeVectorStore(score=0.19),
        provider="OpenAI",
        openai_api_key="test-key",
    )

    assert "couldn't find sufficiently relevant evidence" in answer
    assert sources == []


def test_chroma_persists_reopens_and_retrieves():
    with tempfile.TemporaryDirectory(
        prefix="documind_test_chroma_", ignore_cleanup_errors=True
    ) as directory:
        documents = load_uploaded_file(
            "test.txt",
            b"The project code is PF-204.",
        )
        chunks = split_documents(documents, chunk_size=200, chunk_overlap=20)
        store = build_vector_store(chunks, directory, "test_collection")
        reopened = open_vector_store(directory, "test_collection")

        assert store._collection.count() == 1
        assert reopened._collection.count() == 1
        results = reopened.similarity_search_with_relevance_scores(
            "What is the project code?", k=1
        )
        assert len(results) == 1
        assert "PF-204" in results[0][0].page_content
        assert results[0][0].metadata["source"] == "test.txt"
        del reopened
        del store
        gc.collect()
