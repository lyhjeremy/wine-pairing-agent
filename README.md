# Wine Pairing Agent

A dish is not a search query: this LangGraph agent reasons from the dish to a wine style, searches Wine Enthusiast reviews as a tool, recommends bottles with citations, and critiques its own pairing.
On a frozen 10-dish set its first pick was an accepted grape for all 4 dishes it ran (searching with the dish text alone: 3 of those 4, 3 of 10), at $1.25 API-equivalent and 78 s per pairing, because its critique rejected all 12 drafts and spent every retry.

[Live site](https://lyhjeremy.github.io/wine-pairing-agent/) · [Write-up](https://lyhjeremy.github.io/wine-pairing-agent/overview/) · [Technical notes](writeup.md) · [Sample run](examples/sample_run.md)

Design and copy follow [these standards](https://github.com/lyhjeremy/lyhjeremy/blob/main/DESIGN_STANDARDS.md).

## Results

`eval/dishes.json` holds 10 dishes, each with a budget and the grapes classic pairing rules accept for it, committed before the first run. `python -m eval.run_eval` scores the agent and a baseline (the same search tool queried with the raw dish text, top hit taken) and records the cost and time of every Claude CLI call. Run on 2026-11-05 on an Apple M4:

| | Baseline (search on dish text) | Agent |
|---|---|---|
| Dishes run | 10 | 4 (the first four; each pairing costs about $1.25) |
| First pick an accepted grape | 3 of 10; 3 of the agent's 4 | 4 of 4 |
| Cost per pairing, API-equivalent | $0 (local) | $1.25 median, $1.23 to $1.31 |
| Latency per pairing | 0.3 s for the search in a traced run (not timed here) | 77.9 s median, 79.7 s p95 (n = 4) |

The cost is the `total_cost_usd` the Claude CLI reports (Opus 5.5, its default here); the CLI runs on a subscription, so nothing was billed. Each call read about 13,300 cached tokens and wrote 12,100 to 15,300 more, which is the CLI's own context; the agent's prompts and replies are a few hundred tokens.

The main failure is the critique. It found issues in all 12 rounds across the 4 runs, so every run made 10 LLM calls instead of 4 and stopped at the attempt limit. 7 of the 12 rounds opened by calling the wine too old ("The 2007 JD will be about 19 years old in 2026"): no vintage in the review file is newer than 2017, and the critique judges the notes as of today. The rest were quibbles with one line of the explanation, suggestions of a different grape, and one request for a style the index lacks. Hand-coded counts are in `eval/critique_issues.json`; every run, with its candidates, picks and critique, is in `eval/runs.json`.

The baseline fails where the dish words sound like a wine: "lamb" retrieves Lambrusco. The key is strict, and two baseline misses are arguable (a sweet Torcolato for chocolate, Riesling for roast chicken).

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
git clone https://github.com/lyhjeremy/wine-pairing-agent && cd wine-pairing-agent
python3 -m venv .venv && source .venv/bin/activate
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
| `eval/` | Frozen dish set, scoring script, per-run outputs, per-call cost and latency |

## Limitations

- The scored run is small (the agent on 4 dishes) and the answer key is mine, written from standard pairing rules. A grape match is a floor, not a judgment of the pairing; no sommelier or blind tasting checked the picks.
- The critique has no notion that the review index is a 2017 snapshot, so it rejects nearly every pick for its age.
- The pairing principles are a short list. Fortified wines, very hot food and dessert courses get thinner treatment.
- Retrieval filters on price only. A color preference reaches the agent through the query and the critique.
- The default index holds the 30,000 highest-scored reviews of the 129,971 in the dataset.

The repo ships code only. The dataset and index are built locally and git-ignored.

## License

[MIT](LICENSE) © 2026 Jeremy Lee
