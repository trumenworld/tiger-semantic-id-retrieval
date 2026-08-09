import argparse
import json
import time
from pathlib import Path

import torch

from modeling.models.prefix_logits import PrefixTableLogitsProcessor
from modeling.models.tiger import CorrectItemsLogitsProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='../data/Beauty/index_rqopq_local.json')
    parser.add_argument('--num-codebooks', type=int, default=5)
    parser.add_argument('--codebook-size', type=int, default=256)
    parser.add_argument('--num-beams', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--visited-count', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--sample-prefixes', type=int, default=0)
    return parser.parse_args()


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mapping_path = Path(args.index)
    with mapping_path.open('r', encoding='utf-8') as handle:
        mapping = json.load(handle)
    semantic_ids = torch.tensor(
        [mapping[str(i)] for i in range(len(mapping))],
        dtype=torch.long,
        device=device,
    )

    generator = torch.Generator(device=device).manual_seed(42)
    visited = torch.randint(
        0,
        semantic_ids.shape[0],
        (args.batch_size, args.visited_count),
        generator=generator,
        device=device,
    )
    if args.visited_count >= 3:
        visited[:, -2:] = 0  # Match the zero padding seen by the legacy code.

    old = CorrectItemsLogitsProcessor(
        args.num_codebooks,
        args.codebook_size,
        str(mapping_path),
        args.num_beams,
        visited,
    )
    new = PrefixTableLogitsProcessor(
        args.num_codebooks,
        args.codebook_size,
        str(mapping_path),
        args.num_beams,
        visited,
    )

    row_count = args.batch_size * args.num_beams
    vocab_size = args.num_codebooks * args.codebook_size + 2010
    beam_items = torch.randint(
        0,
        semantic_ids.shape[0],
        (row_count,),
        generator=generator,
        device=device,
    )
    beam_sids = semantic_ids[beam_items]
    decoder_start = torch.full(
        (row_count, 1),
        vocab_size - 3,
        dtype=torch.long,
        device=device,
    )

    old_seconds = 0.0
    new_seconds = 0.0
    for depth in range(args.num_codebooks):
        if depth == 0:
            input_ids = decoder_start
        else:
            offsets = torch.arange(depth, device=device) * args.codebook_size
            input_ids = torch.cat(
                [decoder_start, beam_sids[:, :depth] + offsets.unsqueeze(0)],
                dim=1,
            )

        base_scores = torch.randn(
            row_count,
            vocab_size,
            generator=generator,
            device=device,
        )
        for _ in range(args.repeats):
            synchronize(device)
            started = time.perf_counter()
            old_scores = old(input_ids, base_scores.clone())
            synchronize(device)
            old_seconds += time.perf_counter() - started

            synchronize(device)
            started = time.perf_counter()
            new_scores = new(input_ids, base_scores.clone())
            synchronize(device)
            new_seconds += time.perf_counter() - started

        old_mask = torch.isfinite(old_scores)
        new_mask = torch.isfinite(new_scores)
        mismatches = torch.logical_xor(old_mask, new_mask).sum().item()
        if mismatches:
            raise AssertionError(f'depth {depth}: {mismatches} allowed-token mismatches')
        if not torch.equal(old_scores[old_mask], new_scores[new_mask]):
            raise AssertionError(f'depth {depth}: allowed logits changed')
        print(f'depth {depth + 1}: exact allowed-token match')

    if args.sample_prefixes > 0:
        sample_count = min(args.sample_prefixes, semantic_ids.shape[0])
        sampled_items = torch.randperm(
            semantic_ids.shape[0], generator=generator, device=device
        )[:sample_count]
        sampled_sids = semantic_ids[sampled_items]
        for depth in range(1, args.num_codebooks):
            checked = 0
            for start in range(0, sample_count, row_count):
                prefix = sampled_sids[start:start + row_count, :depth]
                if prefix.shape[0] < row_count:
                    prefix = torch.cat([
                        prefix,
                        prefix[-1:].expand(row_count - prefix.shape[0], -1),
                    ])
                offsets = torch.arange(depth, device=device) * args.codebook_size
                input_ids = torch.cat([
                    torch.full(
                        (row_count, 1), vocab_size - 3,
                        dtype=torch.long, device=device,
                    ),
                    prefix + offsets.unsqueeze(0),
                ], dim=1)
                base_scores = torch.zeros(
                    row_count, vocab_size, device=device
                )
                old_mask = torch.isfinite(old(input_ids, base_scores.clone()))
                new_mask = torch.isfinite(new(input_ids, base_scores.clone()))
                mismatches = torch.logical_xor(old_mask, new_mask).sum().item()
                if mismatches:
                    raise AssertionError(
                        f'sampled depth {depth}: {mismatches} allowed-token mismatches'
                    )
                checked += min(row_count, sample_count - start)
            print(f'depth {depth + 1}: {checked} sampled prefixes match')

    print('device:', device)
    print('legacy seconds:', round(old_seconds, 6))
    print('prefix-table seconds:', round(new_seconds, 6))
    print('processor speedup:', round(old_seconds / new_seconds, 2), 'x')
    if device.type == 'cuda':
        print('peak allocated MiB:', round(torch.cuda.max_memory_allocated() / 2**20, 2))


if __name__ == '__main__':
    main()
