# Wine pairing as an agent

*An agent that reasons about the dish, uses wine search as a tool, and checks its own pairing.*

## From "find a wine" to "pair a wine"

My [Wine Sommelier RAG](https://github.com/lyhjeremy/wine-sommelier-rag) handles "find me a bold red under $25." Pairing is harder, because given a dish you first have to decide what kind of wine to look for, and only then search. That step, from dish to pairing strategy to search query, is what an agent adds on top of retrieval.

This project makes retrieval a tool the agent calls and wraps it in a reason-and-check loop built with LangGraph.

## The graph

```text
parse -> strategize -> retrieve -> recommend -> critique --+-- sound --> END
              ^       (tool call)                          |
              +----------- issues, as feedback ------------+
```

`parse` turns "grilled ribeye with peppercorn sauce" into structured features (main ingredient, body, sauce, cuisine, flavors, cooking method). `strategize` reasons from classic pairing principles to a target wine style and a natural-language search query: acid cuts fat, tannin binds protein, match weight, sweetness tames heat, and what grows together goes together. Most of these will be familiar to anyone who has sat a WSET course (I hold Level 3); the last is older folk wisdom.

`retrieve` is the tool call, a semantic search over the local Wine Enthusiast index with the guest's budget as a filter. `recommend` picks one or two bottles from what came back and explains the pairing, citing each `[n]`. Then `critique` checks the result: were at least two candidates found, is any within budget, and is the pairing coherent for the dish? If any answer is no, it loops back to `strategize` with the feedback, so the next attempt tries a different style or a broader query, up to an attempt limit (two by default).

## Why the loop is there

Retrieval alone is brittle. If `strategize` picks too narrow a query, such as a grape that barely appears under the budget, the search comes back thin and the pairing suffers. The critique node catches that ("too few candidate wines were found, the search was too narrow") and feeds it back, so the agent broadens the search on its own and tries again.

Asked to pair a grilled ribeye with peppercorn cream sauce on a $45 budget, the agent reasoned to a full-bodied, tannic Cabernet, retrieved six Cabernets within budget (from Napa, Rutherford, Oakville, Diamond Mountain, Stellenbosch and Mendoza), and recommended two: the Vinifera 2006 Napa Valley Cabernet (95 points, $40) and the Textbook 2005 Mise en Place from Oakville (94 points, $44). It explained that the tannin scours the fat while the oak mirrors the char, and the critique node judged the pairing sound. The full run is in [`examples/sample_run.md`](examples/sample_run.md).

## Design notes

The agent uses the dense retriever from the first version of Wine Sommelier RAG: Chroma search with metadata filters, 40 candidates re-ordered by 0.8 × cosine similarity + 0.2 × rating, and the top six returned. The sommelier project has since added BM25 fusion and a cross-encoder; this agent has not picked them up.

`parse` and `strategize` both return JSON, so the dish features and the search query become state the graph can route on and log. Recommendations are limited to retrieved wines and cited, which means no bottle, score or price is invented. Generation runs on the Claude CLI (my Claude subscription, with no per-token cost) or the Anthropic API, and retrieval is local.

## Retrieval as a tool, and tracing

`retrieve` is a LangChain `StructuredTool` (`src/tools.py`) with a name, description and typed schema, invoked as `search_wines.invoke({...})`. That is the same abstraction an LLM uses to call a tool, so retrieval shows up in a trace as a tool call.

`--trace` times each node, prints a trace tree and saves the run as JSON, a self-contained record in the style of LangSmith with no account needed. In one traced ribeye run the local search took 0.3 seconds and the four LLM steps took 1.2 to 3.0 seconds each, about nine seconds in total. When critique loops back to re-strategize, the retry appears in the same trace, so self-correction can be seen and timed.

## Limitations and next steps

Pairing quality is not scored. Pairing has few single right answers, which explains the gap without filling it. The principles are also a compact list, well short of a sommelier's full judgment, and fortified wines, very hot food and dessert courses get thinner treatment.

`retrieve` filters on price only. A color or grape preference reaches the agent through the query and the critique, with no hard filter behind it.

A multi-course meal would need a planner that pairs each course and balances the sequence, which the same graph could grow into.

*Code: [github.com/lyhjeremy/wine-pairing-agent](https://github.com/lyhjeremy/wine-pairing-agent)*
