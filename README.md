# TIGER Semantic-ID Retrieval

Reproduction and controlled tokenizer study for generative recommendation on
Amazon Beauty. The project first reproduces the public
`NonameUntitled/tiger` implementation, then replaces its item Semantic ID
with an RQ-OPQ code while keeping the downstream generative recommender and
evaluation protocol fixed.

The project is intended as a research and engineering reference. It does not
redistribute Amazon data, LLaMA weights, or trained checkpoints.

## What is implemented

1. **TIGER baseline**
   - Amazon Beauty 5-core user-item sequences.
   - Leave-one-out train, validation, and test construction.
   - LLaMA-7B item content embeddings from `title`, `categories`, and
     `description`.
   - Semantic IDs loaded from an RQ-VAE table.
   - T5-style encoder-decoder trained to generate the next item SID.

2. **RQ-OPQ tokenizer variant**
   - Three residual K-Means codebooks with sizes `256-256-256`.
   - Two OPQ product-quantization codebooks with sizes `128-128`.
   - Five-token item IDs: `256-256-256-128-128`.
   - The same downstream model, decoder settings, data split, validation
     criterion, and final test procedure as the baseline.

3. **Constrained generation**
   - A GPU-resident prefix table replaces the legacy per-beam scan over every
     item. It masks invalid next tokens and preserves visited-item filtering.

4. **SID quality evaluation**
   - Code usage, entropy, Gini concentration, unique SID count, collision
     groups, and collision item rate.

## Results

The figures below are from the local Amazon Beauty experiments in this
project. `Recall` is the hit rate used by the original evaluation code.

| Setting | SID | NDCG@10 | Recall@10 | Important qualification |
| --- | --- | ---: | ---: | --- |
| TIGER paper report | paper SID | 0.0384 | 0.0648 | Reference value reported for Beauty |
| Public implementation reproduction | 4-token SID | 0.03034 | 0.05455 | Strict item-level test result |
| RQ-OPQ exact-SID run | 5-token SID | **0.0408** | **0.0749** | Exact-SID metric; 15.83% of items share a 5-token SID |

The RQ-OPQ line is therefore **not a strict item-level improvement claim**:
the metric is computed at the Semantic-ID level, and multiple products can
occupy one 5-token SID bucket. The collision-aware interpretation and the
full experiment history are documented in `docs/EXPERIMENTS.md`.

## Downstream model

The decoder is not changed between the baseline and RQ-OPQ comparison:

| Component | Value |
| --- | --- |
| Hidden size | 128 |
| Encoder layers | 4 |
| Decoder layers | 4 |
| Attention heads | 6 |
| Feed-forward size | 1024 |
| Activation | ReLU |
| Dropout | 0.1 |
| Train batch size | 256 |
| Validation/test batch size | 64 |
| Learning rate | 3e-4 |
| Beam size | 100 |
| Returned sequences | 20 |
| Validation metric | NDCG@10 |
| Validation frequency | Every 10 epochs |
| Early stopping | 40 validation epochs without improvement |
| Test | Once, using the best validation checkpoint |

The 5M versus approximately 13M parameter discrepancy is a property of the
public implementation's decoder configuration versus the paper description;
the RQ-OPQ comparison holds the public implementation fixed rather than
silently scaling the decoder.

## Repository layout

```text
configs/                       Reproducible JSON configurations
data/README.md                 Data acquisition and privacy notes
docs/EXPERIMENTS.md            Experiment history and interpretation
docs/METHOD.md                 End-to-end method details
notebooks/                     RQ-VAE, RQ-KMeans, and GPU OPQ notebooks
results/                       Small, human-readable metric summaries
scripts/evaluate_sid_quality.py
scripts/export_sid_table.py   SID table utilities
src/tiger/                     TIGER implementation and prefix decoder
tests/                         Lightweight tests for SID utilities
```

## Reproduce

### 1. Install dependencies

Use Python 3.10-3.12 and a CUDA-enabled PyTorch installation for the LLaMA
embedding and decoder experiments.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\\Scripts\\activate`.

### 2. Prepare data locally

Download Amazon Beauty 5-core interactions and the corresponding product
metadata from the official UCSD Amazon Review source. Follow
`data/README.md` to create the local files below:

```text
data/Beauty/inter.json
data/Beauty/index_rqvae.json
data/Beauty/index_rqopq_local.json
```

The index files map the zero-based item ID used by `inter.json` to a list of
local code IDs. They are experiment artifacts and are intentionally ignored
by Git.

### 3. Train the baseline or RQ-OPQ decoder

The code keeps the original working-directory convention: run from
`src/tiger` so the `modeling` package resolves directly.

```bash
cd src/tiger
python train_tiger.py --params ../../configs/tiger_beauty_rqvae.json
python train_tiger.py --params ../../configs/tiger_beauty_rqopq5.json
```

For faster and safer decoding with a valid SID prefix table:

```bash
python train_tiger_prefix.py --params ../../configs/tiger_beauty_rqopq5.json
```

The configs default to relative paths into the repository's `data/Beauty`
directory. Checkpoint paths are local and should be changed for each machine.

### 4. Evaluate a SID table

```bash
python scripts/evaluate_sid_quality.py \
  --index data/Beauty/index_rqopq_local.json \
  --codebook-sizes 256 256 256 128 128 \
  --output results/rqopq_quality.local.json
```

## Data and model boundaries

`asin` is used only to join metadata with interactions. It is not fed to the
content encoder. Ratings, review text, and user IDs are not content features
for SID construction. Reviews contribute the chronological user-item
sequence used by the downstream next-item task.

## Limitations

- The RQ-OPQ result is Semantic-ID level and must be reported together with
  the collision analysis.
- The public decoder is smaller than the approximately 13M parameters stated
  in the TIGER paper, so the reproduction is not a byte-for-byte replication.
- The early 200k-step scale-up run changed several variables at once
  (embedding model, normalization, usage regularization, learning rate,
  decoder size, and training budget). It is retained as a diagnostic study,
  not used as causal evidence for one individual factor.
- Amazon data and LLaMA weights remain the responsibility of the user to
  download under their respective terms.

## Acknowledgements

- The downstream TIGER implementation is based on
  [NonameUntitled/tiger](https://github.com/NonameUntitled/tiger).
- The Semantic-ID experiments were informed by the TIGER paper and the
  RQ-VAE/Recommender and OneSearch lines of work.

See `THIRD_PARTY_NOTICES.md` for license and attribution details.
