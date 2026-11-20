import json

from store_agent import evals
from store_agent.embeddings import FakeEmbedder
from store_agent.seed import DEFAULT_DATA

PRODUCTS = [
    {"sku": "A", "name": "Rolled Oats", "category": "Pantry", "description": "oats for breakfast porridge"},
    {"sku": "B", "name": "Spaghetti", "category": "Pantry", "description": "pasta for dinner"},
    {"sku": "C", "name": "Dish Tablets", "category": "Household", "description": "dishwasher cleaning"},
]


def test_scores_hits_misses_and_must_not():
    cases = [
        {"query": "breakfast porridge", "expect": ["A"]},
        {"query": "dishwasher cleaning", "expect": ["A"]},  # wrong on purpose: C ranks first
        {"query": "pasta dinner", "expect": ["B"], "must_not": ["B"]},  # hit, but forbidden
    ]
    results = evals.run(cases, PRODUCTS, FakeEmbedder())
    assert [r.passed for r in results] == [True, True, False]  # top 3 of 3 always contains A
    assert results[0].reciprocal_rank == 1.0
    assert results[1].reciprocal_rank < 1.0
    assert results[2].bad == ["B"]
    s = evals.summary(results)
    assert s["cases"] == 3 and s["passed"] == 2


def test_cases_only_reference_real_products():
    skus = {p["sku"] for p in json.loads(DEFAULT_DATA.read_text())}
    for case in json.loads(evals.DEFAULT_CASES.read_text()):
        assert case["query"].strip()
        assert case["expect"], case["query"]
        assert set(case["expect"]) | set(case.get("must_not", [])) <= skus, case["query"]


def test_precision_counts_good_products_in_top_k():
    def result(top, expect):
        return evals.CaseResult(query="q", top=top, expect=expect, must_not=[])

    assert result(["A", "B", "C"], ["A", "B"]).precision == 1.0  # both good products, of 2 possible
    assert result(["A", "X", "Y"], ["A", "B", "C", "D"]).precision == 1 / 3
    assert result(["X", "Y", "A"], ["A"]).precision == 1.0  # only one good product exists
    assert result(["X", "Y", "Z"], ["A"]).precision == 0.0


def test_summary_reports_precision():
    results = evals.run([{"query": "breakfast porridge", "expect": ["A"]}], PRODUCTS, FakeEmbedder())
    assert evals.summary(results)["precision_at_3"] == 1.0


def test_embedding_text_includes_tags_when_present():
    from store_agent.seed import embedding_text

    p = {"name": "Red Lentils", "category": "Pantry", "description": "For soup."}
    assert embedding_text(p) == "Red Lentils. Pantry. For soup."
    assert embedding_text({**p, "tags": "vegan, plant protein"}) == "Red Lentils. Pantry. For soup. vegan, plant protein"


def test_case_diet_filters_candidates():
    products = [{**p, "diet": ["vegan"] if p["sku"] == "B" else []} for p in PRODUCTS]
    [r] = evals.run([{"query": "breakfast porridge", "expect": ["B"], "diet": ["vegan"]}], products, FakeEmbedder())
    assert r.top == ["B"]
