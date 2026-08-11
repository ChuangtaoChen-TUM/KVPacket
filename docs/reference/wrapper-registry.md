# Wrapper Registry, Source, and Identity Contract

Status: **current** (2026-08-10).

This document defines the persistent identities and ownership boundaries used by KV Packet. The key separation is:

- a `WrapperRegistry` owns named adapters for one backend model;
- a `PacketPreprocessor` selects adapters and builds reusable packets offline;
- a `PacketSession` serves existing packets or records full-attention teacher data;
- a `TrainingSample` contains no wrapper state; and
- a `PacketTrainer` owns the source-to-wrapper routing for a training run.

## 1. Wrapper registry

A registry is a named set of `PacketWrapper` adapters that all target one exact backend compatibility descriptor. It is not a collection of packets or sources.

```python
registry = WrapperRegistry(
    compatibility=backend.compatibility,
    wrappers={
        "document": WrapperEntry(document_wrapper),
        "skill": WrapperEntry(skill_wrapper),
    },
)
```

The checkpoint format is `kvpacket.wrapper_registry/1` and contains:

```python
{
    "format": "kvpacket.wrapper_registry/1",
    "compatibility": {
        "backend": "hf",
        "model_id": "Qwen/Qwen3-8B",
        "model_config_fingerprint": "sha256:<hex>",
        "model_artifact_fingerprint": "sha256:<hex>",
        "hidden_size": 4096,
    },
    "wrappers": {
        "document": {
            "header": Tensor,
            "trailer": Tensor,
            "header_len": 8,
            "trailer_len": 8,
            "fingerprint": "sha256:<hex>",
            "training_config": {},
            "meta": {},
        },
    },
    "fingerprint": "sha256:<hex>",
    "meta": {},
}
```

Wrapper IDs match `^[a-z][a-z0-9_.-]{0,127}$`. A registry is non-empty, wrapper dimensions match the backend hidden size, and metadata is JSON-compatible.

A wrapper fingerprint hashes the complete header and trailer tensors. A registry fingerprint hashes the backend compatibility identity and the sorted mapping from wrapper IDs to wrapper fingerprints. The aggregate registry fingerprint validates a saved registry checkpoint; it is not part of an individual packet cache key. Retraining an unrelated wrapper therefore does not invalidate packets made with an unchanged wrapper.

The public preprocessing API always receives an explicit registry. There is no singleton compatibility constructor and no implicit wrapper owned by a session.

## 2. Sources and assignments

`Tokenizable` is a frozen, hashable declaration with these identity fields:

```python
kind: str
source_id: str
value: str
media_type: str | None
```

The preprocessing assignment is a normal mutable mapping:

```python
dict[Tokenizable, str]  # source -> wrapper_id
```

For raw text, content identity is the exact UTF-8 bytes. For local files, it is the exact file bytes rather than only the path. Changing source bytes invalidates the corresponding packet even when tokenization happens to remain unchanged.

Configuration files use `registry_path`:

```yaml
registry_path: ../checkpoints/agent-wrappers.pt
assignments:
  document:
    - type: raw-text
      doc:
        - id: ada-biography
          text: "Ada Lovelace was born ..."
    - type: text-manifest
      doc: [../manifests/biographies.jsonl]
  skill:
    - type: file
      doc:
        - id: powerpoint-skill
          path: ../skills/powerpoint/SKILL.md
    - type: file-manifest
      doc: [../manifests/skill-files.jsonl]
```

`raw-text`, `file`, `text-manifest`, and `file-manifest` are the supported configuration declaration types. Programmatic source encoders may define another non-empty kind. Paths are resolved relative to the config or manifest that declares them. Duplicate IDs, unknown wrapper IDs, malformed records, missing files, and empty manifests fail before model work.

```python
assignments = load_wrapper_assignments(path, registry)
registry, assignments = load_wrapper_registry_and_assignments(path)
```

## 3. Encoded source identity

A framework-side `SourceEncoder` produces `EncodedBody`. Exactly one of `input_ids` and `inputs_embeds` is present. The preprocessor derives two fingerprints with different responsibilities:

- `content_fingerprint` identifies the original declared source bytes;
- `encoded_input_fingerprint` identifies the complete tensor actually consumed by packet materialization.

For ordinary text input, the encoded-input fingerprint covers the complete input-ID tensor. For multimodal or embedding input, it covers the complete embedding tensor. It is deliberately not a sampled dimension, pooled statistic, or projection: exact cache reuse needs exact input identity.

When token IDs have already been computed, callers must still provide the original source declaration:

```python
preprocessor.preprocess_pretokenized(
    RawText(original_text, source_id="document-7"),
    token_ids,
    wrapper_id="document",
)
```

This keeps `content_fingerprint` consistent: it always represents original source content, never sometimes text and sometimes a list of token IDs. The framework hashes the supplied IDs as `encoded_input_fingerprint`; backends do not implement source decoding, schemas, or fingerprint policy.

## 4. Stored packet identity

Every `PacketIdentity` records:

```python
source_id: str
source_kind: str
source_content_fingerprint: str
backend_compatibility_fingerprint: str
wrapper_id: str
wrapper_fingerprint: str
encoded_input_fingerprint: str
```

The packet cache key includes the source content, exact backend, selected wrapper ID and tensors, and complete encoded input. It does not include the aggregate registry fingerprint. Packet stores without these required identities are rejected rather than guessed compatible.

## 5. Offline preprocessing and serving

```python
registry = WrapperRegistry.load(
    "agent-wrappers.pt",
    runtime=backend.compatibility,
    requires_grad=False,
)
preprocessor = PacketPreprocessor(
    backend=backend,
    wrapper_registry=registry,
    source_encoder=TextSourceEncoder(tokenizer),
)
preprocessor.preprocess(assignments)
preprocessor.store.save("agent-packets.pt")
```

Serving receives packets, not wrappers:

```python
store = PacketStore.load("agent-packets.pt", backend=backend)
session = PacketSession(backend=backend, store=store, tokenizer=tokenizer)
session.register_store_chunks()
session.push(prompt_containing_registered_content)
result = session.generate()
```

Serving registration never materializes a packet. A missing packet in `Mode.SERVE` is an error.

## 6. Wrapper-agnostic teacher samples

`Mode.RECORD` runs full attention and does not need a registry or wrapper:

```python
session = PacketSession(backend=backend, mode=Mode.RECORD, tokenizer=tokenizer)
session.register_chunk("document-7", text=original_text)
session.push(original_text)
session.push(query)
session.generate()
sample = session.export_sample()
```

`kvpacket.training_sample/1` stores only teacher information:

```python
token_ids: Tensor
generated_mask: Tensor
chunk_spans: dict[str, list[tuple[int, int]]]
logits: Tensor | None
```

`logits` is required for `objective="teacher_kl"`. For compact
`objective="teacher_ce"` samples it is `None`; CE labels are selected from
`token_ids` by `generated_mask`. The sample stores no wrapper snapshots, wrapper
fingerprints, registry fingerprints, or assignments. A teacher sample is generated
by the full-attention model before adapter selection and can be reused with different
routing experiments.

## 7. Trainer-owned routing

The trainer combines wrapper-agnostic samples with an explicit registry and routing policy:

```python
trainer = PacketTrainer(
    backend,
    registry,
    wrapper_assignments={
        "document-7": "document",
        "calculator-skill": "skill",
    },
    optimize=["document", "skill"],
    objective="teacher_kl",  # or "teacher_ce" for compact token-only samples
    loss_reduction="token_mean",
    optimizer_kwargs={"lr": 1e-3},
)
```

Homogeneous runs may use `default_wrapper_id="document"`. Explicit assignments override that default. Every chunk in a sample must resolve at training time. Trainer checkpoints save the registry, routing policy, optimizer state, and provenance.

This ownership model keeps backend capabilities focused: online backends implement model forwards and cache repositioning; packet-build and training backends add embedding, offline prefill, and differentiable training operations. The framework owns source resolution, fingerprinting, wrapper selection, persistence schemas, and routing.
