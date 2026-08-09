import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import LogitsProcessor


@dataclass(frozen=True)
class _PrefixTable:
    next_node: torch.Tensor
    subtree_item_count: torch.Tensor
    item_path_nodes: torch.Tensor


_TABLE_CACHE = {}


def _build_prefix_table(num_codebooks, codebook_size, index_path, device):
    path = Path(index_path).resolve()
    stat = path.stat()
    cache_key = (
        str(path), stat.st_mtime_ns, stat.st_size,
        num_codebooks, codebook_size, str(device),
    )
    cached = _TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with path.open('r', encoding='utf-8') as handle:
        mapping = json.load(handle)

    children = [{}]
    subtree_item_count = [0]
    item_path_nodes = []

    for item_id in range(len(mapping)):
        semantic_id = mapping[str(item_id)]
        assert len(semantic_id) == num_codebooks

        node = 0
        subtree_item_count[node] += 1
        path_nodes = []
        for token in semantic_id:
            token = int(token)
            assert 0 <= token < codebook_size
            child = children[node].get(token)
            if child is None:
                child = len(children)
                children[node][token] = child
                children.append({})
                subtree_item_count.append(0)
            node = child
            subtree_item_count[node] += 1
            path_nodes.append(node)
        item_path_nodes.append(path_nodes)

    next_node = torch.full(
        (len(children), codebook_size),
        -1,
        dtype=torch.int32,
    )
    for node, node_children in enumerate(children):
        if node_children:
            tokens = torch.tensor(list(node_children), dtype=torch.long)
            child_nodes = torch.tensor(list(node_children.values()), dtype=torch.int32)
            next_node[node, tokens] = child_nodes

    table = _PrefixTable(
        next_node=next_node.to(device),
        subtree_item_count=torch.tensor(
            subtree_item_count,
            dtype=torch.int32,
            device=device,
        ),
        item_path_nodes=torch.tensor(
            item_path_nodes,
            dtype=torch.int32,
            device=device,
        ),
    )
    _TABLE_CACHE.clear()
    _TABLE_CACHE[cache_key] = table
    return table


class PrefixTableLogitsProcessor(LogitsProcessor):
    """Constrain generation with a compact GPU prefix table.

    The legacy processor materializes every item SID for every beam at every
    generated token. This processor traverses one trie node per beam and checks
    only the 256 possible tokens in the active codebook. Per-user subtree counts
    preserve visited-item filtering, including duplicate SIDs shared by items.
    """

    def __init__(self, num_codebooks, codebook_size, index_path, num_beams, visited_items):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.num_beams = num_beams
        self.device = visited_items.device
        self.table = _build_prefix_table(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            index_path=index_path,
            device=self.device,
        )

        batch_size = visited_items.shape[0]
        self.remaining_item_count = self.table.subtree_item_count.unsqueeze(0).repeat(
            batch_size, 1
        )

        # The legacy scatter removes an item once even if it appears repeatedly.
        visited_mask = torch.zeros(
            batch_size,
            self.table.item_path_nodes.shape[0],
            dtype=torch.bool,
            device=self.device,
        )
        visited_mask.scatter_(1, visited_items, True)
        batch_ids, item_ids = visited_mask.nonzero(as_tuple=True)
        if item_ids.numel() > 0:
            path_nodes = self.table.item_path_nodes[item_ids].long()
            count = path_nodes.shape[1]
            repeated_batch_ids = batch_ids[:, None].expand(-1, count).reshape(-1)
            self.remaining_item_count.index_put_(
                (repeated_batch_ids, path_nodes.reshape(-1)),
                torch.full(
                    (path_nodes.numel(),),
                    -1,
                    dtype=self.remaining_item_count.dtype,
                    device=self.device,
                ),
                accumulate=True,
            )

    def _current_depth(self, input_ids):
        return (
            min(
                int(input_ids[:, -1].max().item() // self.codebook_size),
                self.num_codebooks - 1,
            ) + 1
        ) % self.num_codebooks

    def _lookup_nodes(self, input_ids, depth):
        beam_count = input_ids.shape[0]
        nodes = torch.zeros(beam_count, dtype=torch.long, device=input_ids.device)
        if depth == 0:
            return nodes

        offsets = torch.arange(depth, device=input_ids.device) * self.codebook_size
        raw_prefix = input_ids[:, -depth:] - offsets.unsqueeze(0)
        valid = torch.ones(beam_count, dtype=torch.bool, device=input_ids.device)

        for column in range(depth):
            token = raw_prefix[:, column]
            token_valid = (token >= 0) & (token < self.codebook_size)
            safe_nodes = nodes.clamp_min(0)
            safe_tokens = token.clamp(0, self.codebook_size - 1)
            next_nodes = self.table.next_node[safe_nodes, safe_tokens].long()
            valid &= token_valid & (next_nodes >= 0)
            nodes = torch.where(valid, next_nodes, torch.full_like(nodes, -1))
        return nodes

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        depth = self._current_depth(input_ids)
        current_nodes = self._lookup_nodes(input_ids, depth)
        batch_ids = torch.arange(
            input_ids.shape[0], device=input_ids.device
        ) // self.num_beams

        safe_nodes = current_nodes.clamp_min(0)
        child_nodes = self.table.next_node[safe_nodes].long()
        child_exists = child_nodes >= 0
        safe_children = child_nodes.clamp_min(0)
        allowed = child_exists & (
            self.remaining_item_count[batch_ids.unsqueeze(1), safe_children] > 0
        )
        allowed &= (current_nodes >= 0).unsqueeze(1)

        # The legacy implementation turns removed item rows into all-zero rows.
        # This can only add token 0 at depth 0; preserve it for exact comparison.
        if depth == 0 and allowed.shape[1] > 0:
            allowed[:, 0] = True

        start = depth * self.codebook_size
        end = (depth + 1) * self.codebook_size
        scores[:, :start] = -torch.inf
        scores[:, end:] = -torch.inf
        scores[:, start:end].masked_fill_(~allowed, -torch.inf)
        return scores
