# Notebooks

The notebooks are convenience pipelines for the three SID construction
stages. They expect local data and are not required for running the decoder.

- `rqvae_pipeline.ipynb`: RQ-VAE-style SID construction.
- `rqkmeans_pipeline.ipynb`: residual RQ-KMeans SID construction.
The original GPU OPQ continuation notebook was intentionally kept out of this
public repository because it contains experiment-owner-specific Kaggle
dataset handles. The algorithmic configuration is documented in
`docs/EXPERIMENTS.md`; a local Python implementation of SID table export and
quality evaluation is in `scripts/`.
