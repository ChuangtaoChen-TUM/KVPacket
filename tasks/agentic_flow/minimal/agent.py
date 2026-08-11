"""A minimal, dependency-free agent loop driven by a ``PacketSession``.

For a task it assembles ``system + [activated skill bodies] + task`` as discrete
blocks and drives a session in one of three modes:

* **full_prefill** — no chunks; the whole context (all skill bodies) is prefilled
  fresh every task. Oracle for quality; the cost KV Packet removes.
* **naive_concat** — skills preprocessed with a zero-filler wrapper (raw caches).
* **kv_packet** — skills registered as chunks with a ``PacketWrapper``; each activated
  skill body is **auto-matched** in the pushed content and served from its packet.

The caller never says "this block is skill X" — the library matches the registered
skill bodies. RECORD mode instead logs a teacher trajectory (`TrainingSample`) with
the skill spans and generated-token mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from kvpacket.backend import HFBackend
from kvpacket.packet import PacketWrapper, RawText, TextSourceEncoder, WrapperEntry, WrapperRegistry
from kvpacket.preprocessing import PacketPreprocessor
from kvpacket.session import Mode, PacketSession, TrainingSample

from tasks.common.metrics import TransformerFlops, calculate_metrics, f1_states

from .skills import Skill
from .tasks import AgentTask

__all__ = ["AgentMode", "AgentResult", "MinimalAgent"]

SYSTEM_PROMPT = (
    "You are a careful assistant. Skill instructions are provided below. "
    "Apply the relevant skill and reply with only the final answer, nothing else."
)


class AgentMode(str, Enum):
    FULL_PREFILL = "full_prefill"
    NAIVE_CONCAT = "naive_concat"
    KV_PACKET = "kv_packet"


@dataclass
class AgentResult:
    task_id: str
    mode: str
    pred: str
    answer: str
    activated_skills: list[str]
    matched_skills: list[str]
    ttft: float
    flops: int
    num_generated: int
    success: bool
    f1: float
    meta: dict = field(default_factory=dict)


def _normalize(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line.strip().strip("\"'`.,: ").lower()


class MinimalAgent:
    def __init__(
        self,
        backend: HFBackend,
        tokenizer,
        skills: dict[str, Skill],
        wrapper: PacketWrapper | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.backend = backend
        self.tokenizer = tokenizer
        self.skills = skills
        self.wrapper = wrapper
        self.system_prompt = system_prompt
        raw_wrapper = PacketWrapper(0, 0, backend.hidden_size, requires_grad=False)
        self.raw_store = self._build_store(raw_wrapper)
        self.packet_store = self._build_store(wrapper) if wrapper is not None else self.raw_store
        self._flops = TransformerFlops(backend.config)

    def _build_store(self, wrapper: PacketWrapper):
        """Preprocess every skill once with the selected wrapper."""
        registry = WrapperRegistry(
            compatibility=self.backend.compatibility,
            wrappers={"skill": WrapperEntry(wrapper)},
        )
        preprocessor = PacketPreprocessor(
            backend=self.backend,
            wrapper_registry=registry,
            source_encoder=TextSourceEncoder(self.tokenizer),
        )
        preprocessor.preprocess(
            {RawText(skill.block, source_id=name): "skill" for name, skill in self.skills.items()}
        )
        return preprocessor.store

    # -- one task, one mode --------------------------------------------------
    def run(self, task: AgentTask, mode: AgentMode, max_new_tokens: int = 24) -> AgentResult:
        mode = AgentMode(mode)
        store = None
        if mode is AgentMode.KV_PACKET:
            store = self.packet_store
        elif mode is AgentMode.NAIVE_CONCAT:
            store = self.raw_store
        sess = PacketSession(
            backend=self.backend,
            store=store,
            mode=Mode.SERVE,
            match="token",
            tokenizer=self.tokenizer,
        )
        if store is not None:
            sess.register_store_chunks(task.skills)

        self._push_blocks(sess, task)
        res = sess.generate(max_new_tokens=max_new_tokens)

        pred, gold = _normalize(res.text), task.answer.lower()
        success = pred == gold or gold in pred.split()
        tp, fp, fn = f1_states(gold.split(), pred.split())
        f1 = calculate_metrics(tp, fp, fn)[2]

        flops = res.flops
        if mode is AgentMode.FULL_PREFILL:
            flops = self._skill_prefill_flops(task)

        return AgentResult(
            task_id=task.task_id,
            mode=mode.value,
            pred=res.text.strip(),
            answer=task.answer,
            activated_skills=list(task.skills),
            matched_skills=[m.chunk_id for m in res.cover],
            ttft=res.ttft,
            flops=int(flops),
            num_generated=len(res.token_ids),
            success=success,
            f1=f1,
            meta={"num_skills": task.num_skills},
        )

    # -- one task, RECORD (teacher trajectory) -------------------------------
    def record(self, task: AgentTask, max_new_tokens: int = 24) -> TrainingSample:
        sess = PacketSession(
            backend=self.backend,
            mode=Mode.RECORD,
            match="token",
            tokenizer=self.tokenizer,
        )
        for name in task.skills:
            sess.register_chunk(name, text=self.skills[name].block)
        self._push_blocks(sess, task)
        sess.generate(max_new_tokens=max_new_tokens)
        sample = sess.export_sample()
        assert sample is not None
        return sample

    # -- assembly ------------------------------------------------------------
    def _push_blocks(self, sess: PacketSession, task: AgentTask) -> None:
        sess.push(self.system_prompt + "\n\n")
        for name in task.skills:
            # Each skill.block ends in a blank line, so activated skills are back-to-back
            # packets (no interior unmatched run). The library matches them; the caller
            # never says which block is which skill.
            sess.push(self.skills[name].block)
        sess.push(f"Task: {task.prompt}\nAnswer:")

    def _skill_prefill_flops(self, task: AgentTask) -> int:
        skill_tokens = sum(
            len(self.tokenizer(self.skills[n].block, add_special_tokens=False)["input_ids"])
            for n in task.skills
        )
        sys_len = len(self.tokenizer(self.system_prompt, add_special_tokens=False)["input_ids"])
        return self._flops.total_flops(seq_len=skill_tokens, cache_len=sys_len)
