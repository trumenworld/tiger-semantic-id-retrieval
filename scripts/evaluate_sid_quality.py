"""Evaluate usage and collision statistics for a Semantic-ID JSON table."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def entropy(probabilities: list[float]) -> float:
    import math

    return -sum(p * math.log(p) for p in probabilities if p > 0)


def gini(counts: list[int]) -> float:
    values = sorted(counts)
    total = sum(values)
    if not values or total == 0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(values))
    return (2 * weighted) / (len(values) * total) - (len(values) + 1) / len(values)


def load_table(path: Path) -> list[list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SID table must be a JSON object keyed by item ID")
    rows = [payload[key] for key in sorted(payload, key=lambda value: int(value))]
    if not rows or any(not isinstance(row, list) for row in rows):
        raise ValueError("SID table contains no valid token lists")
    length = len(rows[0])
    if any(len(row) != length for row in rows):
        raise ValueError("All SID rows must have the same length")
    return [[int(token) for token in row] for row in rows]


def evaluate(rows: list[list[int]], codebook_sizes: list[int]) -> dict:
    if len(rows[0]) != len(codebook_sizes):
        raise ValueError("one codebook size is required for each SID position")

    usage = []
    for position, size in enumerate(codebook_sizes):
        counts = Counter(row[position] for row in rows)
        histogram = [counts.get(code, 0) for code in range(size)]
        total = len(rows)
        probabilities = [count / total for count in histogram]
        usage.append(
            {
                "position": position + 1,
                "codebook_size": size,
                "unique_codes": sum(count > 0 for count in histogram),
                "usage_rate": sum(count > 0 for count in histogram) / size,
                "entropy": entropy(probabilities),
                "normalized_entropy": entropy(probabilities) / __import__("math").log(size),
                "gini": gini(histogram),
            }
        )

    buckets = Counter(tuple(row) for row in rows)
    collision_items = sum(count for count in buckets.values() if count > 1)
    return {
        "items": len(rows),
        "sid_length": len(rows[0]),
        "codebook_sizes": codebook_sizes,
        "unique_sid": len(buckets),
        "collision_groups": sum(count > 1 for count in buckets.values()),
        "collision_items": collision_items,
        "collision_item_rate": collision_items / len(rows),
        "max_collision_bucket": max(buckets.values()),
        "mean_items_per_sid": len(rows) / len(buckets),
        "positions": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--codebook-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = evaluate(load_table(args.index), args.codebook_sizes)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
