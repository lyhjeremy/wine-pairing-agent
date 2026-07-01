# Wine Pairing Agent

Describe your meal — *"grilled ribeye with peppercorn sauce"*, *"spicy Thai green
curry"* — and a **LangGraph agent** reasons from sommelier pairing principles to
the right wine *style*, searches a database of **130,000 real wine reviews** for
matching bottles, recommends with citations, and **critiques its own pick** —
retrying with a different strategy if the pairing is weak or nothing fits your
budget.

> 🌐 **Overview:** https://lyhjeremy.github.io/wine-pairing-agent/

## The idea: RAG as a tool inside an agent
My [Wine Sommelier RAG](https://github.com/lyhjeremy/wine-sommelier-rag) answers
"find me a wine like X." This project puts that retrieval **inside an agent** that
first has to *figure out what style to even search for*. The agent doesn't just
retrieve — it reasons about the dish (body, fat, acid, spice), forms a pairing
strategy, uses retrieval as a **tool**, and then checks whether the result is
actually any good.

## The graph
```
parse ─▶ strategize ─▶ retrieve ─▶ recommend ─▶ critique ─┬─ clean ─▶ END
              ▲          (RAG tool)                       │
              └───────────────── issues & budget ─────────┘
```
| Node | What it does |
|---|---|
| **parse** | Turns the free-text meal into structured features (main, body, sauce, cuisine, flavours) |
| **strategize** | Reasons from pairing principles → a target wine style + a search query |
| **retrieve** | **Tool call:** semantic search over 130k real reviews (with a budget filter) |
| **recommend** | Picks 1–2 bottles and explains the pairing logic, cited `[n]` |
| **critique** | Checks candidate count, budget, and pairing coherence — loops back to `strategize` with feedback if weak |

If the first strategy retrieves too few wines or the critique judges the pairing
off, the agent **re-strategizes** (a different style or broader query) and tries
again, up to an attempt budget.

## Quick start
```bash
pip install -r requirements.txt

python fetch_data.py            # download the ~130k-review dataset -> data/
python -m src.ingest --limit 30000   # build the wine index the agent searches

python -m src.cli "grilled ribeye with peppercorn sauce" --budget 40
python -m src.cli "spicy Thai green curry" --budget 25 --color white
python -m src.cli                # interactive
```
Generation runs on the **Claude CLI** by default (your Claude subscription, no
per-token cost); set `ANTHROPIC_API_KEY` to use the API. Retrieval is local & free.

## Files
| Path | What it is |
|---|---|
| `src/graph.py` | The LangGraph agent: parse → strategize → retrieve → recommend → critique loop |
| `src/pairing.py` | The sommelier pairing principles the agent reasons with |
| `src/retriever.py` | The retrieval tool: semantic search over the review index |
| `src/ingest.py` | Build the wine index (embeddings → Chroma) |
| `src/embedder.py` | Local sentence-transformers embedder (free, offline) |
| `src/llm.py` | LLM wrapper — Claude CLI (default) or Anthropic API |
| `src/cli.py` | Collect the meal, run the agent, print the pairing + agent trace |

## Notes
Ships **code only** — the dataset and index are built locally and git-ignored.
Every run prints an **agent trace** so you can watch it reason, retrieve and
self-check. Recommendations come from professional reviews, for personal use.

## License
[MIT](LICENSE) © 2026 Jeremy Lee
