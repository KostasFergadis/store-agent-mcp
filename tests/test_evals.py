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
