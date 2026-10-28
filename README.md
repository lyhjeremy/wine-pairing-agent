# Wine Pairing Agent

A LangGraph agent that takes a dish, reasons to a wine style, searches Wine Enthusiast reviews as a tool, recommends bottles with citations, and critiques its own pairing.

[Live site](https://lyhjeremy.github.io/wine-pairing-agent/) · [Write-up](https://lyhjeremy.github.io/wine-pairing-agent/overview/) · [Technical notes](writeup.md) · [Sample run](examples/sample_run.md)

Design and copy follow [these standards](https://github.com/lyhjeremy/lyhjeremy/blob/main/DESIGN_STANDARDS.md).

## What it does

For "grilled ribeye steak with a peppercorn cream sauce" with `--budget 45`, the agent reasoned to a full-bodied, tannic Cabernet, retrieved six Cabernets at or under $45, and recommended two:

- **Vinifera 2006 Cabernet Sauvignon, Napa Valley**, 95 points, $40 `[3]`: "The structure scours the cream and fat while the oak mirrors the grilled char."
- **Textbook 2005 Mise en Place Cabernet Sauvignon, Oakville**, 94 points, $44 `[5]`: "that grippy structure is exactly what a fatty ribeye needs"

The critique node judged the pairing sound. The full output, including all six candidates, is in [`examples/sample_run.md`](examples/sample_run.md).

## How it works

```text
parse -> strategize -> retrieve -> recommend -> critique --+-- sound --> END
              ^       (tool call)                          |
              +----------- issues, as feedback ------------+
```

| Node | What it does |
|---|---|
| `parse` | Turns the free-text meal into JSON features: main, body, sauce, cuisine, key flavors, cooking method |
| `strategize` | Reasons from the pairing principles in `src/pairing.py` to a target wine style and a search query |
| `retrieve` | Calls `search_wines`, a LangChain `StructuredTool`: dense search over a local Chroma index with the budget as a price filter, 40 candidates re-ordered by 0.8 × similarity + 0.2 × rating, top six kept |
| `recommend` | Claude picks one or two of the retrieved bottles, cites each as `[n]`, and explains the pairing |
| `critique` | Deterministic checks (at least two candidates, at least one within budget) plus an LLM coherence check; any issue routes back to `strategize` with the feedback |

Retries stop at `--max-attempts` (default 2). The graph is a LangGraph `StateGraph` in `src/graph.py`.

Stack: Python, LangGraph, LangChain tools, sentence-transformers, Chroma, Claude (CLI by default, Anthropic API optional).

## How to run

```bash
pip install -r requirements.txt

python fetch_data.py                 # download the ~130k-review dataset to data/
python -m src.ingest --limit 30000   # build the wine index the agent searches (or --all)

python -m src.cli "grilled ribeye with peppercorn sauce" --budget 40
python -m src.cli "spicy Thai green curry" --budget 25 --color white
python -m src.cli                    # interactive
```

Generation uses the Claude CLI and your Claude subscription by default. Set `ANTHROPIC_API_KEY` to use the API. Retrieval runs locally.

### Tracing

Add `--trace` to time each node, print a trace tree and save the run as JSON under `traces/`:

```bash
python -m src.cli "grilled ribeye with peppercorn sauce" --budget 45 --trace
```

```text
AGENT TRACE  (node · time · detail)
────────────────────────────────────────────────────
  parse         1.2s   ribeye · rich body
  strategize    2.3s   style → full-bodied tannic red
  retrieve      0.3s   6 candidate wines (tool: search_wines)
  recommend     3.0s   drafted pairing with citations
  critique      2.2s   sound
────────────────────────────────────────────────────
```

(Recorded output with its bullet and checkmark glyphs dropped.) The local search took 0.3 seconds; the four LLM steps took the rest. `python assets/make_trace_figure.py traces/last_run.json` draws a saved trace as a timeline. When critique sends the agent back, the second pass appears in the trace.

## Repo layout

| Path | What it is |
|---|---|
| `src/graph.py` | The LangGraph agent and its retry edge |
| `src/pairing.py` | The pairing principles the agent reasons with |
| `src/tools.py` | Retrieval wrapped as a LangChain `StructuredTool` |
| `src/trace.py` | Run tracer: per-node timing, tree view, JSON output |
| `src/retriever.py` | Dense search over the review index with price filters and rating-weighted re-ordering |
| `src/ingest.py` | Builds the index (embeddings to Chroma) |
| `src/embedder.py` | Local sentence-transformers embedder |
| `src/llm.py` | LLM wrapper: Claude CLI (default) or Anthropic API |
| `src/cli.py` | Collects the meal, runs the agent, prints the pairing and trace |
| `examples/sample_run.md` | The recorded ribeye run |

## Limitations

- Pairing quality has no scored evaluation. The repo shows that the agent works and checks itself; how good its pairings are hasn't been measured.
- The pairing principles are a short list. Fortified wines, very hot food and dessert courses get thinner treatment.
- Retrieval filters on price only. A color preference reaches the agent through the query and the critique.
- The default index holds the 30,000 highest-scored reviews of the 129,971 in the dataset.

The repo ships code only. The dataset and index are built locally and git-ignored.

## License

[MIT](LICENSE) © 2026 Jeremy Lee
