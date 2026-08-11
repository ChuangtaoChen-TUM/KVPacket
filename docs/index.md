# KV Packet documentation

KV Packet reuses independently precomputed context caches by surrounding each reusable
source with a small trainable header/trailer adapter. The library owns source
preprocessing, packet identity, cache assembly, adapter training, and serving; task
datasets and complete workflows live under `tasks/`.

## Start here

- [Getting started](getting-started.md): installation, dependency groups, and cache
  locations.
- [Architecture](concepts/architecture.md): ownership boundaries and the offline,
  online, and training paths.
- [Public API](reference/public-api.md): the concise package-root API and focused
  extension subpackages.

## Guides

- [Preprocessing reusable sources](guides/preprocessing.md)
- [Training packet wrappers](guides/training.md)
- [Serving and agent-framework integration](guides/serving.md)
- [Compressing packet caches](guides/compression.md)

## Reference

- [Wrapper registry and artifact identity](reference/wrapper-registry.md)

Runnable end-to-end examples are available in `tasks/multi_doc/`,
`tasks/agentic_flow/repository_qa/`, and `examples/`.
