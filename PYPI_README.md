# KV Packet

KV Packet provides recomputation-free, context-independent KV-cache reuse for large
language models. It surrounds reusable sources with small trainable header/trailer
adapters, prefills each packet independently, and assembles their caches at serving
time with positional realignment but no document-token recomputation.

## Installation

```bash
pip install kvpacket
```

Optional integrations are installed separately:

```bash
pip install "kvpacket[compress]"  # KVPress cache compression
pip install "kvpacket[serve]"     # OpenAI Responses-compatible HTTP/SSE transport
pip install "kvpacket[agent]"     # MCP support for agent examples
```

KV Packet requires Python 3.12 or newer. Install the PyTorch build appropriate for
your compute platform before loading a model.

## Using preprocessed packets

```python
from kvpacket import HFBackend, PacketSession, PacketStore

backend = HFBackend(model)
store = PacketStore.load("document-packets.pt", backend=backend)

session = PacketSession(backend=backend, store=store, tokenizer=tokenizer)
session.register_store_chunks()
session.push(prompt_containing_registered_content)
result = session.generate()
```

Packet construction and adapter training are offline operations. At serving time,
`PacketStore` loads validated precomputed caches, while `PacketSession` matches
registered content and forwards only unmatched online tokens.

## Library layers

- `PacketPreprocessor` builds reusable packet caches from original sources.
- `PacketTrainer` trains wrapper adapters from full-attention teacher samples.
- `PacketExecutor` exposes reentrant packet-aware prefill for existing serving stacks.
- `PacketSession` provides stateful matching, prefill, recording, and generation.
- `HFBackend` supplies the included Hugging Face compute backend.
- `KVPressCompressor` optionally compresses packet caches offline.

See the [documentation](https://github.com/ChuangtaoChen-TUM/KVPacket/blob/main/docs/index.md),
[training guide](https://github.com/ChuangtaoChen-TUM/KVPacket/blob/main/docs/guides/training.md),
and [serving guide](https://github.com/ChuangtaoChen-TUM/KVPacket/blob/main/docs/guides/serving.md)
for complete workflows and extension interfaces.

## Paper

KV Packet implements the method introduced in
[*KV Packet: Recomputation-Free Context-Independent KV Caching for
LLMs*](https://arxiv.org/abs/2604.13226).

The package is distributed under the
[MIT License](https://github.com/ChuangtaoChen-TUM/KVPacket/blob/main/LICENSE).
