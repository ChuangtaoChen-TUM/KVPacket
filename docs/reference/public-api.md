# Public API

The package root exposes the common end-to-end workflow. Specialized protocols,
records, and low-level results live in focused subpackages.

## Package root

| Area | Symbols |
|---|---|
| Backend | `HFBackend` |
| Sources and wrappers | `RawText`, `FileSource`, `TextSourceEncoder`, `PacketWrapper`, `WrapperRegistry` |
| Packet construction | `PacketPreprocessor`, `PacketStore`, `KVPressCompressor` |
| Reentrant execution | `PacketExecutor`, `PacketCacheState`, `MixedPrefillPolicy` |
| Stateful execution | `PacketSession`, `Mode`, `GenerationResult` |
| Training | `TrainingSample`, `PacketTrainer`, `create_wrapper` |

## Extension subpackages

- `kvpacket.backend`: compute, packet-build, generation, and training protocols plus
  compatibility records.
- `kvpacket.packet`: source declarations and encoders, wrapper entries, registry
  configuration, and fingerprint helpers.
- `kvpacket.kvcache`: backend-independent cache tensors and position operations.
- `kvpacket.runtime`: reentrant packet execution, matching results, sampling, and
  operation result records.
- `kvpacket.session`: stateful serving, teacher recording, and packet-store loading.
- `kvpacket.training`: wrapper initialization, attention masks, and training.
- `kvpacket.compression`: cache-compressor protocol and KVPress integration.
- `kvpacket.serve`: structured model requests, codecs, completion orchestration, and
  optional managed state. Provider-specific wire formats live in
  `kvpacket.serve.adapters`; HTTP transport lives in `kvpacket.serve.http`.

Internal persistence constants and matching algorithms are intentionally absent from
the package root.
