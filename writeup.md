<p align="center">
  <img src="assets/banner.png" alt="Wine Pairing Agent" width="100%">
</p>

# Wine Pairing as an Agent, Not Just a Search

*Giving retrieval a brain: an agent that reasons about the dish, uses wine search
as a tool, and second-guesses its own pairing.*

## From "find a wine" to "pair a wine"

My [Wine Sommelier RAG](https://github.com/lyhjeremy/wine-sommelier-rag) is great
at *"find me a bold red under $25."* But pairing is a harder problem: given a
**dish**, you first have to decide *what kind of wine you're even looking for*
before you can search for it. That reasoning step — dish → pairing strategy →
search query — is exactly what an **agent** adds on top of retrieval.

So this project treats retrieval as a **tool** the agent calls, and wraps it in a
reasoning-and-checking loop built with LangGraph.

## The graph

<p align="center">
  <img src="assets/architecture.png" alt="Wine Pairing Agent architecture" width="760">
</p>

- **parse** turns *"grilled ribeye with peppercorn sauce"* into structured
  features: main protein, body, sauce, cuisine, key flavours, cooking method.
- **strategize** reasons from classic pairing principles (acid cuts fat, tannin
  binds protein, match weight, sweetness tames heat, "what grows together goes
  together") to a **target wine style** and a natural-language **search query**.
- **retrieve** is the tool call: semantic search over 130,000 real Wine Enthusiast
  reviews, with the guest's budget pushed in as a filter.
- **recommend** picks 1–2 bottles from what was retrieved and explains the pairing
  logic, citing each `[n]`.
- **critique** checks the result: did we find enough candidates? is anything within
  budget? is the pairing actually coherent for the dish? If not, it loops back to
  **strategize** *with the feedback*, so the next attempt tries a different style
  or a broader query — bounded by an attempt budget.

## Why the loop matters

Retrieval alone is brittle: if `strategize` picks too niche a query (say, a grape
that barely appears under the budget), the search comes back thin and the pairing
suffers. The critique node catches that — "too few candidate wines were found, the
search was too narrow" — and feeds it back, so the agent *broadens* on its own
instead of confidently serving a bad match. That's the difference between a
pipeline and an agent: it can notice its first idea didn't work and change it.

Asked to pair a **grilled ribeye with peppercorn cream sauce ($45 budget)**, the
agent reasoned to a full-bodied tannic Cabernet, retrieved six real Napa/Oakville
Cabs within budget, and recommended two — explaining that the tannin scours the
fat while the oak mirrors the char — with the critique node confirming the pairing
was coherent. The full run is in [`examples/sample_run.md`](examples/sample_run.md).

## Design notes

- **The retriever is literally the same component** as the standalone Wine
  Sommelier RAG — this project demonstrates composing RAG *into* an agent rather
  than reimplementing it.
- **Structured hops.** `parse` and `strategize` return JSON, so the dish features
  and the search query are first-class state the graph can reason over and log.
- **Grounded output.** Recommendations are constrained to retrieved wines and
  cited, so — like the RAG project — no bottle, score or price is invented.
- **Generation** runs on the Claude CLI (my Claude subscription — no per-token
  cost) or the Anthropic API; retrieval is always local and free.

## Retrieval as a real tool — and watching the agent think

The `retrieve` step isn't a hard-coded function call; it's a LangChain
**`StructuredTool`** (`src/tools.py`) with a name, description and typed schema,
invoked as `search_wines.invoke({...})`. That's the same abstraction an LLM uses
to call a tool — so retrieval becomes a first-class, introspectable component that
appears in a trace as a *tool call* rather than an opaque step.

And because an agent's value is in *how* it reaches an answer, every run can be
traced: `--trace` times each node, prints a trace tree, and saves the run as JSON
— a self-contained, LangSmith-style record, no account needed. When critique loops
back to re-strategize, that retry shows up right there in the trace. It turns "the
agent self-corrects" from a claim into something you can see and time.

## Limitations & next steps

- The pairing principles are a compact canon, not an exhaustive sommelier's
  intuition; edge cases (fortified wines, extreme spice, dessert courses) are
  thinner.
- `retrieve` filters on price but not colour/variety directly — colour preference
  is currently honoured through the query and critique rather than a hard filter.
- A multi-course meal would benefit from a planner that pairs each course and
  balances the flight — a natural extension of the same graph.

*Code: [github.com/lyhjeremy/wine-pairing-agent](https://github.com/lyhjeremy/wine-pairing-agent)*
