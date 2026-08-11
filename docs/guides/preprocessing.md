# Preprocessing reusable sources

Packet preprocessing is an offline operation. It binds original source content to a
named wrapper, builds an isolated cache, and records enough identity information to
reject stale or model-incompatible packets later.

```python
from kvpacket import PacketPreprocessor, RawText, TextSourceEncoder

assignments = {
    RawText("Reusable document text", source_id="document-1"): "document",
}
preprocessor = PacketPreprocessor(
    backend=backend,
    wrapper_registry=wrapper_registry,
    source_encoder=TextSourceEncoder(tokenizer),
)
preprocessor.preprocess(assignments)
preprocessor.store.save("document-packets.pt")
```

`RawText` and `FileSource` cover literal text and local files. A custom
`SourceEncoder` may return an `EncodedBody` containing either one input-ID sequence or
one embedding sequence. The framework—not the encoder—fingerprints the original bytes
and complete encoded tensor.

When tokenization has already happened, retain the original source declaration:

```python
preprocessor.preprocess_pretokenized(
    RawText(original_text, source_id="document-1"),
    token_ids,
    wrapper_id="document",
)
```

Serving loads the resulting store without loading training state:

```python
from kvpacket import PacketStore

store = PacketStore.load("document-packets.pt", backend=backend)
```

See [Wrapper registry and artifact identity](../reference/wrapper-registry.md) for
checkpoint fields and invalidation rules.
