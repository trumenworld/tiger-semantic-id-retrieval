# Data setup

This directory intentionally contains no Amazon data or pretrained weights.

Obtain the Amazon Beauty 5-core interactions and product metadata from the
official UCSD Amazon Review source:

- https://jmcauley.ucsd.edu/data/amazon/

Prepare a zero-based local item mapping and chronological interaction file:

```text
data/Beauty/inter.json
data/Beauty/index_rqvae.json
data/Beauty/index_rqopq_local.json
```

`inter.json` is a JSON object mapping a zero-based user ID to a chronological
list of zero-based item IDs. The index files are JSON objects mapping the
zero-based item ID to its SID token list.

The content embedding stage uses LLaMA-7B on `title`, `categories`, and
`description`. `asin` is used only to join metadata and interaction records;
ratings and review bodies are not content-encoder inputs.
