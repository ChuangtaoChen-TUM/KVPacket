# Architecture

KV Packet separates reusable-context preparation from request-time cache assembly.
This keeps model-specific computation behind a backend and leaves applications free to
own their existing request and conversation state.

## Layers

```text
agent or application
        │ structured messages and tools
        ▼
provider adapter → model codec → completion engine
                                      │ token IDs
                                      ▼
                              session or state manager
                                      │ caller-owned cache state
                                      ▼
                              packet-aware executor
                               ┌──────┴──────┐
                               │ packet store│
                               └──────┬──────┘
                                      ▼
                               compute backend
```

Only the packet-aware executor, matching, packet store, and cache operations are core
to KV Packet. Provider adapters, codecs, the state manager, HTTP transport, and the
Hugging Face backend are supplied implementations that applications may replace.

## Offline packet construction

1. A source encoder resolves original content into token IDs or embeddings.
2. `PacketPreprocessor` fingerprints the original content and exact encoded tensor.
3. The selected `PacketWrapper` inserts learned header and trailer embeddings.
4. A packet-build backend prefills the wrapped sequence independently.
5. `PacketStore` persists the cache with model, source, wrapper, and compression
   identities.

Packets store both a physical KV length and the original logical span. Compression may
shorten the former without changing the positions occupied by following context.

## Online execution

`PacketExecutor` matches registered source token sequences, repositions their stored
caches, concatenates them, and forwards only unmatched online tokens. It receives and
returns `PacketCacheState`; it does not retain a conversation itself.

`PacketSession` is the stateful convenience layer. It buffers pushes, manages one
conversation cache, records full-attention teachers, and performs generation.
Applications with an existing state manager can use `PacketExecutor` directly.

## Training

Teacher samples are recorded with full attention and contain no wrapper state.
`PacketTrainer` later combines those samples with a `WrapperRegistry` and explicit
source-to-wrapper routing. The base model remains frozen; only wrapper header and
trailer parameters are optimized.
