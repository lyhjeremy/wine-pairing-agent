"""The retrieval step, exposed as a real LangChain **tool**.

The agent's `retrieve` node doesn't just call a function — it invokes a named,
described `StructuredTool`. That's the same abstraction an LLM uses when it
decides to call a tool by name, and it makes the wine search a reusable,
introspectable component (it has a schema, a description, and shows up in traces
as a tool call) rather than a hard-coded step.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from .retriever import Retriever, WineHit

_RET: Retriever | None = None


def _ret() -> Retriever:
    global _RET
    if _RET is None:
        _RET = Retriever()
    return _RET


def search_wines(query: str, max_price: float | None = None, k: int = 6) -> list[WineHit]:
    """Semantic search over 130,000 real professional wine reviews.

    Args:
        query: a natural-language style / grape / flavour description of the wine
            you want (e.g. "bold tannic red with dark fruit for grilled steak").
        max_price: optional maximum price per bottle in USD.
        k: number of bottles to return.

    Returns:
        Matching real bottles, each with its score, price and tasting note.
    """
    return _ret().search(query, k=k, max_price=max_price)


# The tool object the agent invokes. `.invoke({"query": ..., "max_price": ...})`
# runs the search; its name/description/schema are introspectable and traceable.
wine_search_tool = StructuredTool.from_function(
    func=search_wines,
    name="search_wines",
    description=(
        "Search 130k professional wine reviews for real bottles matching a "
        "natural-language style query, with an optional max price. Returns bottles "
        "with score, price, and tasting note."
    ),
)
