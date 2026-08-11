from __future__ import annotations

from tasks.multi_doc.data.base import RetrievalSample
from tasks.multi_doc.data.length_matching import length_match_samples


class _CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(chr(token_id) for token_id in token_ids)


def _sample(documents: list[str], index: int) -> RetrievalSample:
    return RetrievalSample(
        preamble="prefix",
        documents=documents,
        task_prompt="question",
        answer="answer",
        meta={"index": index},
    )


def _text_packets(sample: RetrievalSample) -> list[str]:
    assert all(isinstance(document, str) for document in sample.documents)
    return [document for document in sample.documents if isinstance(document, str)]


def test_length_match_is_deterministic_and_preserves_target_fields() -> None:
    tokenizer = _CharacterTokenizer()
    target = _sample(["answer passage", "support passage"], 7)
    fillers = [_sample(["unrelated alpha", "unrelated beta"], 8)]

    first = length_match_samples(
        [target],
        fillers,
        tokenizer=tokenizer,
        target_total_tokens=80,
        num_packets=8,
        seed=11,
    )[0]
    second = length_match_samples(
        [target],
        fillers,
        tokenizer=tokenizer,
        target_total_tokens=80,
        num_packets=8,
        seed=11,
    )[0]

    assert first.documents == second.documents
    assert len(first.documents) == 8
    first_packets = _text_packets(first)
    assert [len(tokenizer.encode(packet)) for packet in first_packets] == [10] * 8
    assert "".join(first_packets).startswith("answer passage\n\nsupport passage")
    assert first.preamble == target.preamble
    assert first.task_prompt == target.task_prompt
    assert first.answer == target.answer
    assert first.meta["index"] == 7
    assert first.meta["length_match"]["original_document_tokens"] == 31


def test_length_match_mixed_position_distributes_target_stream() -> None:
    tokenizer = _CharacterTokenizer()
    targets = [_sample([f"target-{index}"], index) for index in range(5)]
    fillers = [_sample(["abcdefghijklmnopqrstuvwxyz"], 8)]

    matched = length_match_samples(
        targets,
        fillers,
        tokenizer=tokenizer,
        target_total_tokens=40,
        num_packets=4,
        seed=11,
        filler_position="mixed",
    )

    for index, sample in enumerate(matched):
        packets = _text_packets(sample)
        stream = "".join(packets)
        target_text = f"target-{index}"
        assert stream.find(target_text) == sample.meta["length_match"][
            "prefix_filler_tokens"
        ]
        assert sample.meta["length_match"]["filler_position"] == "mixed"
        assert [len(tokenizer.encode(packet)) for packet in packets] == [10] * 4

    assert [
        sample.meta["length_match"]["prefix_filler_tokens"]
        for sample in matched
    ] == [0, 8, 16, 24, 32]
