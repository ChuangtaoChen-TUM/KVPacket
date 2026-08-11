# Offline KV-cache compression

KV Packet can apply token-pruning compression while it materializes reusable packet
caches. Compression is optional and does not change wrapper training: train or load a
wrapper normally, then select a compressor for offline preprocessing.

## Setup and use

Install the optional dependencies:

```bash
uv sync --extra compress
```

Pass a compressor to `PacketPreprocessor`:

```python
from kvpacket import KVPressCompressor, PacketPreprocessor

compressor = KVPressCompressor("knorm", compression_ratio=0.5)
preprocessor = PacketPreprocessor(
    backend=backend,
    wrapper_registry=wrapper_registry,
    source_encoder=source_encoder,
    cache_compressor=compressor,
)
packets = preprocessor.preprocess(assignments)
```

`compression_ratio` is the fraction of KV positions removed. A ratio of `0.5`
therefore retains approximately half of the wrapped source positions. Extra
provider-specific constructor arguments can be supplied through `options=`.

The supported KVPress method names are:

- `random`
- `knorm`
- `streaming_llm`
- `lagkv`
- `key_diff`
- `snapkv`
- `expected_attention`

These methods shorten the token axis uniformly in every layer. They may select
different source positions by layer or head, which does not prevent KV Packet from
moving the complete packet by one uniform position offset. Methods that change head
dimensions, produce different sequence lengths by layer, compress during decoding, or
require multiple reconstruction forwards are not part of the initial integration.

KV Packet supplies only the isolated wrapped source to KVPress. It does not append the
later user query during offline compression. Some methods, such as SnapKV, may use
query states from the end of that source internally; this is distinct from requiring an
external retrieval query.

## Physical and logical lengths

A compressed packet has two lengths:

- `StoredPacket.stored_len` / `KVCache.seq_len` is the number of physical KV slots
  retained after compression.
- `StoredPacket.wrapped_len` / `KVCache.doc_len` is the original wrapped-source span.

Serving concatenates physical tensors using `stored_len`, but advances logical
position IDs using `wrapped_len`. For example, a packet compressed from 1,000 to 500
KV slots still occupies 1,000 positions in the assembled prompt. Moving it from an
offline start position of zero to an online start position of 2,000 applies the same
`+2000` RoPE delta to every retained key; the compressor does not need to expose the
retained indices.

The full logical span is persisted independently from the shortened tensors. Packet
identity covers the compression provider, method, implementation version, ratio, and
options. Changing any of these settings makes an existing packet stale.

The initial integration supports global-attention models; sliding-window cache
visibility is rejected because it requires additional logical-position handling.
Compressed batched prefill currently requires unpadded inputs. `PacketPreprocessor`
materializes sources independently and therefore satisfies this restriction. Direct
backend callers should prefill padded sources one at a time when compression hooks are
active.
