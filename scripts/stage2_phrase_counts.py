# Count how Stage 2 used the phrase bank in an evaluation run, for D-04 and the
# phrase bank balance task. Reads a CSV written by scripts/evaluate_stages.py.
#
#   python scripts/stage2_phrase_counts.py results/evaluation_20260930_0220.csv
#
# Reports, for the Stage 2 rows with status ok:
#   - how many group choices put the written opposite of the positive phrase in
#     the negative prompt (78 of 400 on 30 September, before D-04; every choice
#     should be the written opposite once D-04 is merged).
#   - how often each phrase was chosen as the positive and as the negative.

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

BANK = Path("data/phrase_bank.json")
GROUPS = ("subject", "motion", "lighting", "atmosphere")


def load_bank():
    # Phrases in bank order per group, and each phrase's written opposite.
    with open(BANK, "r", encoding="utf-8") as file:
        bank = json.load(file)
    phrases, opposite = {}, {}
    for group, pairs in bank.items():
        phrases[group] = [p for pair in pairs for p in pair]
        for a, b in pairs:
            opposite[a], opposite[b] = b, a
    return phrases, opposite


def split_prompt(text):
    # Stage 2 joins one phrase per group, in GROUPS order, with ", ".
    # No phrase in the bank contains a comma, so this split is exact.
    return dict(zip(GROUPS, text.split(", ")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="results CSV from evaluate_stages.py")
    args = parser.parse_args()

    phrases, opposite = load_bank()
    with open(args.csv, newline="", encoding="utf-8") as file:
        rows = [r for r in csv.DictReader(file)
                if r["stage"] == "2" and r["status"] == "ok"]

    total = matched = 0
    by_group = Counter()
    positive_use = {g: Counter() for g in GROUPS}
    negative_use = {g: Counter() for g in GROUPS}

    for row in rows:
        positive = split_prompt(row["prompt"])
        negative = split_prompt(row["negative_prompt"])
        for group in GROUPS:
            chosen = positive[group]
            total += 1
            positive_use[group][chosen] += 1
            negative_use[group][negative[group]] += 1
            if negative[group] == opposite[chosen]:
                matched += 1
                by_group[group] += 1

    print(f"Stage 2 windows: {len(rows)}, group choices: {total}")
    print(f"Negative is the written opposite: {matched} / {total}")
    for group in GROUPS:
        print(f"  {group:<11} {by_group[group]} / {len(rows)}")

    print("\nPhrase use (positive, negative), in bank order")
    for group in GROUPS:
        print(f"\n{group}")
        for phrase in phrases[group]:
            print(f"  {positive_use[group][phrase]:>3}  "
                  f"{negative_use[group][phrase]:>3}  {phrase}")


if __name__ == "__main__":
    main()
