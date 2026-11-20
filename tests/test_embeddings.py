from pathlib import Path

from store_agent.embeddings import DEFAULT_MODEL, DIM, FakeEmbedder, prefixes

ROOT = Path(__file__).resolve().parents[1]


def test_schema_vector_size_matches_embedder():
    assert f"vector({DIM})" in (ROOT / "db" / "schema.sql").read_text()
    assert len(FakeEmbedder().embed_query("feta")) == DIM


def test_default_model_gets_its_query_and_document_prefixes():
    query, doc = prefixes(DEFAULT_MODEL)
    assert query == "task: search result | query: "
    assert doc == "title: none | text: "


def test_models_without_prefixes_get_none():
    assert prefixes("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") == ("", "")
