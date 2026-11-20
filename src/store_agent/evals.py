"""Search evals: does semantic search return the products a person would expect?

Tests check that the code does what it should. Evals check that the *model*
does: each case is a query, the products that count as a good answer, and
optionally products that must not show up near the top (spaghetti is not
breakfast). The score makes a change of model or of embedded text a decision
based on numbers rather than on a few searches tried by hand.

Runs in memory against data/products.json with the same embedding text the
seed uses, so it needs no database and leaves the catalog alone. With 31
products, ranking by exact cosine similarity gives the same order the HNSW
index would.

    uv run store-agent-evals
    uv run store-agent-evals --model intfloat/multilingual-e5-large
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .embeddings import Embedder, FastEmbedder
from .seed import DEFAULT_DATA, embedding_text

DEFAULT_CASES = Path(__file__).resolve().parents[2] / "evals" / "search_cases.json"
TOP_K = 3


@dataclass
class CaseResult:
    query: str
    top: list[str]  # SKUs, best first
    expect: list[str]
    must_not: list[str]
    note: str = ""
    diet: list[str] = field(default_factory=list)

    @property
    def hit(self) -> bool:
        return any(sku in self.expect for sku in self.top[:TOP_K])

    @property
    def bad(self) -> list[str]:
        return [sku for sku in self.top[:TOP_K] if sku in self.must_not]

    @property
    def passed(self) -> bool:
        return self.hit and not self.bad

    @property
    def precision(self) -> float:
        """Share of the top 3 that are good, out of as many good products as exist (at most 3)."""
        good = sum(sku in self.expect for sku in self.top[:TOP_K])
        return good / min(TOP_K, len(self.expect))

    @property
    def reciprocal_rank(self) -> float:
        """1 if the first good product is ranked first, 1/2 if second, and so on."""
        for i, sku in enumerate(self.top, start=1):
            if sku in self.expect:
                return 1 / i
        return 0.0


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / ((math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0)


def run(cases: list[dict], products: list[dict], embedder: Embedder) -> list[CaseResult]:
    skus = [p["sku"] for p in products]
    doc_vecs = embedder.embed([embedding_text(p) for p in products])
    diets = [set(p.get("diet", [])) for p in products]
    results = []
    for case in cases:
        q = embedder.embed_query(case["query"])
        # A case's `diet` stands in for the filter the agent is told to set (see search_products).
        wanted = set(case.get("diet", []))
        pool = [(s, v) for s, v, d in zip(skus, doc_vecs, diets, strict=True) if wanted <= d]
        ranked = sorted(pool, key=lambda sv: _cosine(q, sv[1]), reverse=True)
        results.append(
            CaseResult(
                query=case["query"],
                top=[sku for sku, _ in ranked],
                expect=case["expect"],
                must_not=case.get("must_not", []),
                note=case.get("note", ""),
                diet=case.get("diet", []),
            )
        )
    return results


def summary(results: list[CaseResult]) -> dict:
    n = len(results)
    return {
        "cases": n,
        "passed": sum(r.passed for r in results),
        "pass_rate": round(sum(r.passed for r in results) / n, 3) if n else 0.0,
        "mrr": round(sum(r.reciprocal_rank for r in results) / n, 3) if n else 0.0,
        "precision_at_3": round(sum(r.precision for r in results) / n, 3) if n else 0.0,
    }


def report(results: list[CaseResult], names: dict[str, str], out=sys.stdout) -> None:
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        top = ", ".join(names.get(s, s) for s in r.top[:TOP_K])
        diet = f"  [diet: {', '.join(r.diet)}]" if r.diet else ""
        print(f"{mark}  {r.query!r}{diet}", file=out)
        print(f"      top {TOP_K}: {top}", file=out)
        if not r.hit:
            print(f"      expected one of: {', '.join(names.get(s, s) for s in r.expect)}", file=out)
        if r.bad:
            print(f"      should not be here: {', '.join(names.get(s, s) for s in r.bad)}", file=out)
    s = summary(results)
    print(
        f"\n{s['passed']}/{s['cases']} passed ({s['pass_rate']:.0%}), "
        f"MRR {s['mrr']:.2f}, precision@{TOP_K} {s['precision_at_3']:.2f}",
        file=out,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score semantic product search against expected results.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default=None, help="fastembed model name (default: the one the app uses)")
    parser.add_argument("--min-pass-rate", type=float, default=0.9, help="exit 1 below this (0 to never fail)")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text())
    products = json.loads(args.data.read_text())
    embedder = FastEmbedder(args.model) if args.model else FastEmbedder()
    results = run(cases, products, embedder)
    s = summary(results)
    if args.json:
        print(json.dumps(s))
    else:
        report(results, {p["sku"]: p["name"] for p in products})
    sys.exit(1 if s["pass_rate"] < args.min_pass_rate else 0)


if __name__ == "__main__":
    main()
