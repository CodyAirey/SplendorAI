#!/usr/bin/env python3
"""
join_ranks.py

Combine three rank CSVs into one CSV with a Rank column (as the first column).

Input format for each CSV (header exact match expected):
GemType,Points,Cost[Onyx],Cost[Diamond],Cost[Ruby],Cost[Sapphire],Cost[Emerald]

Default filenames:
- rank1_cards.csv
- rank2_cards.csv
- rank3_cards.csv

Usage:
  python join_ranks.py
  # or specify paths:
  python join_ranks.py --rank1 path/to/rank1.csv --rank2 path/to/rank2.csv --rank3 path/to/rank3.csv --output all_cards.csv
"""

import argparse
import csv
from pathlib import Path
import sys

EXPECTED_HEADERS = [
    "GemType",
    "Points",
    "Cost[Onyx]",
    "Cost[Diamond]",
    "Cost[Ruby]",
    "Cost[Sapphire]",
    "Cost[Emerald]",
]

def read_rank_file(path: Path, rank_value: int):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Validate header
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")
        if [h.strip() for h in reader.fieldnames] != EXPECTED_HEADERS:
            raise ValueError(
                f"{path} header mismatch.\n"
                f"Expected: {EXPECTED_HEADERS}\n"
                f"Found:    {[h.strip() for h in reader.fieldnames]}"
            )
        for row in reader:
            # Skip completely blank lines
            if not any((v or "").strip() for v in row.values()):
                continue
            # Attach rank
            out = {"Rank": rank_value}
            for h in EXPECTED_HEADERS:
                out[h] = row[h].strip()
            rows.append(out)
    return rows

def main():
    ap = argparse.ArgumentParser(description="Join rank1/2/3 card CSVs into a single CSV with Rank column.")
    ap.add_argument("--rank1", default="rank1_cards.csv", help="Path to rank 1 CSV")
    ap.add_argument("--rank2", default="rank2_cards.csv", help="Path to rank 2 CSV")
    ap.add_argument("--rank3", default="rank3_cards.csv", help="Path to rank 3 CSV")
    ap.add_argument("--output", default="all_cards.csv", help="Output CSV path")
    args = ap.parse_args()

    paths = [
        (Path(args.rank1), 1),
        (Path(args.rank2), 2),
        (Path(args.rank3), 3),
    ]

    for p, _ in paths:
        if not p.exists():
            print(f"Missing input file: {p}", file=sys.stderr)
            sys.exit(1)

    all_rows = []
    for p, rank in paths:
        all_rows.extend(read_rank_file(p, rank))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["Rank"] + EXPECTED_HEADERS
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {out_path}")

if __name__ == "__main__":
    main()
