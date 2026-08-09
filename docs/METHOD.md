# Method details

## End-to-end flow

```text
Amazon metadata + interactions
        |
        +--> content text: title + categories + description
        |        |
        |        +--> LLaMA-7B mean-pooled item embedding (4096-D)
        |                 |
        |                 +--> RQ-KMeans: 256 -> 256 -> 256
        |                          |
        |                          +--> residual after layer 3
        |                                   |
        |                                   +--> OPQ: 2 subspaces, 128 each
        |                                            |
        |                                            +--> five-token SID
        |
        +--> chronological user-item sequence
                 |
                 +--> leave-one-out train/validation/test samples
                          |
                          +--> TIGER encoder-decoder generates next SID
                                   |
                                   +--> NDCG / Recall and collision analysis
```

## Stage A: content representation

Each product receives a text representation built from its title, category
list, and description. The LLaMA-7B hidden states are masked and mean-pooled
to obtain a 4096-dimensional vector. The ASIN is a join key, not a semantic
feature. Review rating, review body, and user identity are also outside this
content representation.

## Stage B: Semantic ID construction

The baseline index is an RQ-VAE-derived table with four generated positions in
the public reproduction. The experimental index uses:

1. Residual K-Means layer 1 on the item embedding.
2. Residual K-Means layer 2 on the remaining residual.
3. Residual K-Means layer 3 on the remaining residual.
4. An orthogonal product-quantization stage on the final residual, split into
   two subspaces.

The resulting local vocabulary sizes are `256-256-256-128-128`. For the
decoder, position-local IDs are offset into disjoint ranges. The code exports
both local and global representations when the full notebook pipeline is run.

## Stage C: sequence construction

For every user, interactions are sorted by event time before this stage. With
`[i1, i2, ..., in]`, the implementation creates:

- training prefixes ending before the last two items;
- a validation prefix ending before the final item;
- a test prefix containing the full history and predicting the final item.

The extended training sampler creates multiple prefixes from one user, which
is why the number of training samples is larger than the number of users.

## Stage D: generation and evaluation

The TIGER model uses a T5-style encoder-decoder with 128 hidden units, four
encoder layers, four decoder layers, six heads, a 1024-unit feed-forward
block, ReLU activation, and 0.1 dropout. It generates one code position at a
time with beam size 100 and returns the top 20 sequences.

The prefix-table processor stores trie transitions on the target device and
masks the logits to valid child tokens. It preserves the original
visited-item filtering, including the fact that multiple items may share a
SID. Validation selects the best checkpoint by NDCG@10. Test is run once on
that checkpoint after training.
