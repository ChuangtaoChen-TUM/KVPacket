# Packaged wrapper

`qwen3_4b_repository_navigation.pt` is the final `PacketTrainer` checkpoint for
`Qwen/Qwen3-4B-Instruct-2507` and the checked-in repository-navigation skill. It uses an
8-token header and 8-token trailer and was trained for five epochs with token-ID teacher
cross entropy on 110 post-activation samples.

The checkpoint contains wrapper and optimizer state but no model weights, repository
contents, prompts, generated trajectories, or materialized KV cache. Load it only with a
compatible backend through `PacketTrainer.load`; `pipeline.py --phase materialize` is the
supported example.

SHA-256:
`060006f3c0a6f636c07054918e57c13c4f0b2a3e536ed318efc086bb8c7d3a3f`.
