"""Write eval/critic/labeling_sheet.csv: one row per first-round draft and criterion,
with the Opus critic's and the Sonnet judge's verdicts and empty columns for a
person to fill in. No human labels exist yet.

    python -m eval.make_labeling_sheet
"""
import csv
import json
from pathlib import Path

from src import rubric

OUT = Path(__file__).resolve().parent / "critic"


def main():
    runs = json.loads((OUT / "runs.json").read_text())
    judges = json.loads((OUT / "judges.json").read_text())
    with open(OUT / "labeling_sheet.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dish", "meal", "criterion", "definition", "recommendation", "candidates",
                    "opus_critic_pass", "sonnet_judge_pass", "sonnet_reason",
                    "human_pass (y/n)", "human_note"])
        for r in runs:
            d = r["rounds"][0]
            sec = judges.get("second", {}).get(r["id"], {}).get("criteria", {})
            for k, text in rubric.LLM_CRITERIA.items():
                w.writerow([r["id"], r["meal"], k, text, d["recommendation"], d["candidates_text"],
                            d["critique"]["criteria"][k]["pass"], sec.get(k, {}).get("pass", ""),
                            sec.get(k, {}).get("reason", ""), "", ""])
    print("wrote", OUT / "labeling_sheet.csv")


if __name__ == "__main__":
    main()
