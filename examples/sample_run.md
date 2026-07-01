# Example run

```
$ python -m src.cli "grilled ribeye steak with a peppercorn cream sauce" --budget 45
```

The agent parses the dish, reasons to a target style (full-bodied tannic Cabernet),
searches 130k real reviews for matching bottles within budget, recommends with
citations, and its critique node confirms the pairing is coherent:

```

================================================================
PAIRING
================================================================
For a rich, char-grilled ribeye with peppery cream sauce, you want bold tannin to cut the savory fat and dark fruit to echo the char. Two standouts:

**Vinifera 2006 Cabernet Sauvignon, Napa Valley — 95 pts, $40 [3]**
This full-bodied, 100% Cab wraps blackberry and black currant around silky tannins with graphite and vanilla-sandalwood oak — the structure scours the cream and fat while the oak mirrors the grilled char beautifully. My top pick.

**Textbook 2005 Mise en Place Cabernet Sauvignon, Oakville — 94 pts, $44 [5]**
Firmly tannic with a dry astringency and ripe blackberry-currant depth; that grippy structure is exactly what a fatty ribeye needs, and the wine's savory power stands up to the umami and black pepper without being overwhelmed.

Both deliver the classic steakhouse Cab-and-ribeye harmony — [3] leans elegant and polished, [5] leans bold and grippy if you like more tannic bite.

— Candidate wines (retrieved from 130k reviews) —
  [1] Stark-Condé 2011 Unfined and Unfiltered Cabernet Sauvignon (Stellenbosch) · 92 pts · $27
  [2] JD 2007 Cabernet Sauvignon (Diamond Mountain District) · 94 pts · $40
  [3] Vinifera 2006 Cabernet Sauvignon (Napa Valley) · 95 pts · $40
  [4] Foley Johnson 2013 Cabernet Sauvignon (Rutherford) · 92 pts · $45
  [5] Textbook 2005 Mise en Place Cabernet Sauvignon (Oakville) · 94 pts · $44
  [6] Las Vertientes 2014 Reserva Cabernet Sauvignon (Mendoza) · 92 pts · $40

================================================================
AGENT TRACE
================================================================
  • parse → grilled ribeye steak (rich body)
  • strategize → full-bodied, tannic dry red — Cabernet Sauvignon or a Bordeaux-style blend | query: "full-bodied Cabernet Sauvignon bold ripe black currant firm tannins oak vanilla savory pairs with grilled steak"
  • retrieve → 6 candidate wines from the review index
  • recommend → drafted pairing with citations
  • critique → sound
  ✔ pairing passed review
```
