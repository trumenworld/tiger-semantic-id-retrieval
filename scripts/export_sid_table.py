"""Combine saved RQ and OPQ code arrays into decoder-ready SID tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq-ids", type=Path, required=True, help="N x 3 .npy array")
    parser.add_argument("--opq-ids", type=Path, required=True, help="N x 2 .npy array")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--global-offsets",
        action="store_true",
        help="Offset positions into disjoint decoder vocabulary ranges",
    )
    args = parser.parse_args()

    rq_ids = np.load(args.rq_ids)
    opq_ids = np.load(args.opq_ids)
    if rq_ids.ndim != 2 or rq_ids.shape[1] != 3:
        raise ValueError(f"expected RQ shape (N, 3), got {rq_ids.shape}")
    if opq_ids.ndim != 2 or opq_ids.shape != (rq_ids.shape[0], 2):
        raise ValueError(f"expected OPQ shape ({rq_ids.shape[0]}, 2), got {opq_ids.shape}")

    sid = np.concatenate([rq_ids, opq_ids], axis=1).astype(np.int64, copy=False)
    if args.global_offsets:
        sid = sid + np.asarray([0, 256, 512, 768, 896], dtype=np.int64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = {str(item_id): row.tolist() for item_id, row in enumerate(sid)}
    args.output.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(table)} item SIDs to {args.output}")


if __name__ == "__main__":
    main()
