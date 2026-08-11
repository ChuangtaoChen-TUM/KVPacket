# Reference test assets

This directory contains historical fixtures and manual parity gates used to validate
KV Packet against known implementations and results. They are test assets rather than
runtime library data.

- `adapters/`: small converted adapter fixtures required by parity tests.
- `configs/`: exact historical reproduction configurations.
- `qwen3/`: GPU/manual parity gates for the pinned Qwen3 reference experiment.
- `rerotate_reference.py`: standalone legacy re-rotation math used by unit tests.

The normal `pytest` suite remains lightweight. The scripts under `qwen3/` require the
pinned model to be cached locally and must be invoked explicitly as documented there.
