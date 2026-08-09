# Experiment history and interpretation

## 1. Early RQ-VAE baseline

The first local implementation used an RQ-VAE encoder-decoder to quantize
Sentence-T5 item embeddings. A compact four-token decoder run produced a
complete train/evaluate pipeline, but the metric was below the paper report.
This was useful for validating data parsing, Semantic-ID generation, residual
quantization, beam decoding, and ranking metrics.

## 2. 200k-step scale-up study

The larger run changed several things together:

- LLaMA/Sentence-T5-side representation and embedding dimensionality;
- encoder-decoder capacity and batch size;
- learning rate and optimizer schedule;
- first-layer L2 normalization during quantization;
- a first-layer usage KL regularizer;
- training budget and checkpoint selection.

The long run performed worse than the compact baseline. It is not rigorous to
attribute the regression only to first-layer codebook collapse. The main
lesson is that the run was a confounded scale-up rather than a one-variable
ablation: normalization, usage regularization, learning-rate stability, and
model capacity all changed at once.

In the usage-regularized variant, the regularizer was applied to the batch
mean soft assignment for layer 1:

```text
L_usage = 0.01 * KL(mean(softmax(-distance / 0.1)) || Uniform(256))
```

It encouraged uniform marginal code usage; it did not directly pull every
codeword toward every item. However, on a small and semantically structured
catalogue, too much pressure toward uniform usage can fight the geometry of
the content embedding. This is why it is recorded as a diagnostic, not kept
in the final baseline.

## 3. Public TIGER implementation reproduction

The next stage reproduced the public `NonameUntitled/tiger` code path. The
downstream model and training/evaluation code were used as the comparison
anchor. The local run reached:

```text
NDCG@10   0.03034
Recall@10 0.05455
```

The public README reports `0.03191` and `0.05822` for its Beauty run. The
small gap is reported rather than hidden. The repository's implementation
also has approximately 5M parameters, while the paper describes a model of
approximately 13M parameters.

## 4. RQ-OPQ SID replacement

The RQ-OPQ experiment changed the item SID construction only:

```text
RQ-KMeans: 256-256-256
OPQ:       128-128
Final SID: 256-256-256-128-128
```

The downstream decoder architecture, learning rate, batch sizes, validation
criterion, validation frequency, early-stopping patience, beam settings, and
final-test procedure were held fixed relative to the public reproduction.

The exact-SID test result was:

```text
NDCG@10   0.0408
Recall@10 0.0749
```

These exceed the TIGER paper's reported Beauty values `0.0384` and `0.0648`
when evaluated as Semantic IDs. The five-token SID collision item rate was
15.83%, so this should be described as a strong Semantic-ID retrieval result,
not an unconditional product-level improvement.

## 5. Recommended next experiment

The clean next step is a collision-aware item-level evaluation or an AdaSID-
style adaptive disambiguation token. It must be trained and evaluated with
the same downstream decoder settings, and the report should contain both:

1. exact-SID metrics, for comparability with the current code path;
2. item-level metrics after resolving or marginalizing collided SID buckets.

No AdaSID result is claimed in this repository because that experiment was not
run.
