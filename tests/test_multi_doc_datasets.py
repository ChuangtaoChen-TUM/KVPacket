"""Local, network-free checks for the ported reference dataset generators."""

from __future__ import annotations

from typing import Any, cast

import pytest

from tasks.multi_doc.data import DATASETS, get_ret_eval_generator
from tasks.multi_doc.data.base import RetrievalSample
from tasks.multi_doc.data import gsm8k_dependent as gsm8k_module
from tasks.multi_doc.data import hotpot_qa as hotpot_module
from tasks.multi_doc.data import multinews as multinews_module
from tasks.multi_doc.data import musique as musique_module
from tasks.multi_doc.data import niah as niah_module
from tasks.multi_doc.data._utils import split_sentences
from tasks.multi_doc.data.hotpot_qa import (
    hotpot_qa_answer_postprocess,
    hotpot_qa_generator,
)
from tasks.multi_doc.data.gsm8k_dependent import (
    gsm8k_dependent_generator,
    gsm8k_numeric_postprocess,
)
from tasks.multi_doc.data.musique import (
    musique_answer_postprocess,
    musique_generator,
)
from tasks.multi_doc.data.multinews import multinews_generator
from tasks.multi_doc.data.niah import niah_answer_postprocess, niah_generator
from tasks.multi_doc.data.registry import _group_document_packets


def _hotpot_rows() -> list[dict[str, Any]]:
    return [
        {
            "question": f"Question {index}?",
            "answer": f"Answer {index}",
            "level": ("easy", "medium", "hard")[index % 3],
            "context": {
                "title": [f"Title {index}A", f"Title {index}B"],
                "sentences": [
                    [f"Paragraph {index}A. ", "Second sentence."],
                    [f"Paragraph {index}B."],
                ],
            },
        }
        for index in range(9)
    ]


def _musique_rows() -> list[dict[str, Any]]:
    return [
        {
            "question": f"Multi-hop question {index}?",
            "answer": f"Result {index}",
            "paragraphs": [
                {
                    "title": f"Title {index}A",
                    "paragraph_text": f"Document {index}A.",
                },
                {
                    "title": f"Title {index}B",
                    "paragraph_text": f"Document {index}B.",
                },
            ],
        }
        for index in range(8)
    ]


def _niah_rows() -> list[dict[str, Any]]:
    return [
        {
            "context": f"Sentence {index}A. Sentence {index}B.",
            "question": f"Which values belong to {index}?",
            "answer": [f"value-{index}", f"code-{index}"],
        }
        for index in range(12)
    ]


def test_registry_contains_all_reference_dataset_generators():
    assert set(DATASETS) == {
        "synthetic",
        "biography",
        "gsm8k_dependent",
        "hotpot_qa",
        "musique",
        "multinews",
        "niah",
    }


def test_gsm8k_dependent_generator_masks_answer_and_splits_contiguously(
    monkeypatch,
):
    rows = [
        {
            "question": "Nia has 12 marbles and buys half as many. How many total?",
            "answer": (
                "She buys 12/2 = <<12/2=6>>6 marbles.\n"
                "The total is 12+6 = <<12+6=18>>18 marbles.\n"
                "#### 18"
            ),
        },
        {
            "question": "A box has 5 rows of 7 objects. How many objects?",
            "answer": "Multiply 5*7 = <<5*7=35>>35.\n#### 35",
        },
    ]
    monkeypatch.setattr(gsm8k_module, "_load", lambda *_args: rows)
    sample = next(
        gsm8k_dependent_generator(
            num_samples=1,
            num_documents=4,
            split="test",
            seed=0,
        )
    )
    assert len(sample.documents) == 4
    assert all(isinstance(document, str) for document in sample.documents)
    document_text = " ".join(document for document in sample.documents if isinstance(document, str))
    assert "[MASK]" in document_text
    assert sample.answer not in document_text
    assert sample.meta["structure"] == "contiguous_dependent_reasoning"
    assert "Later chunks may depend" in sample.preamble
    assert gsm8k_numeric_postprocess("Calculation... Short Answer: $18.00", "18") == ("18", "18")


def test_hotpot_generator_preserves_paragraphs_and_requested_difficulty(monkeypatch):
    rows = _hotpot_rows()
    monkeypatch.setattr(hotpot_module, "_load_hotpot", lambda *_args: rows)
    first = list(
        hotpot_qa_generator(
            num_samples=2,
            num_documents=0,
            seed=7,
            difficulty=["hard"],
        )
    )
    second = list(
        hotpot_qa_generator(
            num_samples=2,
            num_documents=0,
            seed=7,
            difficulty=["hard"],
        )
    )
    assert [sample.meta["index"] for sample in first] == [sample.meta["index"] for sample in second]
    assert all(sample.meta["difficulty"] == "hard" for sample in first)
    assert all(len(sample.documents) == 2 for sample in first)
    first_document = first[0].documents[0]
    assert isinstance(first_document, str)
    assert first_document.endswith("Second sentence.")
    assert first[0].task_prompt.endswith(
        "Your response should end with: 'Short Answer: <your final answer>'.\n"
    )
    with pytest.raises(ValueError, match="num_documents must be 0"):
        list(hotpot_qa_generator(1, num_documents=1))


def test_hotpot_answer_only_prompt_requests_no_explanation(monkeypatch):
    monkeypatch.setattr(hotpot_module, "_load_hotpot", lambda *_args: _hotpot_rows())
    sample = next(
        hotpot_qa_generator(
            num_samples=1,
            num_documents=0,
            seed=7,
            add_cot=False,
            answer_only=True,
        )
    )
    assert sample.task_prompt.endswith(
        "Return only: 'Short Answer: <your final answer>'. Do not include an explanation.\n"
    )


def test_musique_generator_keeps_few_shots_out_of_selected_samples(monkeypatch):
    rows = _musique_rows()
    monkeypatch.setattr(musique_module, "_load_musique", lambda *_args: rows)
    samples = list(
        musique_generator(
            num_samples=3,
            num_documents=0,
            num_shots=1,
            seed=3,
            add_cot=False,
        )
    )
    assert len(samples) == 3
    assert all(sample.preamble.startswith("Context:") for sample in samples)
    assert all(len(sample.documents) == 2 for sample in samples)
    assert all(sample.query.startswith("Multi-hop question") for sample in samples)
    assert "Answer:" in samples[0].preamble


def test_musique_direct_prompt_keeps_answer_marker_without_explicit_cot(monkeypatch):
    rows = _musique_rows()
    monkeypatch.setattr(musique_module, "_load_musique", lambda *_args: rows)

    sample = next(
        musique_generator(
            num_samples=1,
            num_documents=0,
            add_inst=True,
            add_cot=False,
            answer_only=True,
        )
    )

    assert "thinking step by step" not in sample.task_prompt
    assert sample.task_prompt.endswith(
        "Return only: 'Answer: <your final answer>'. Do not include an explanation.\n"
    )


def test_musique_views_pair_documents_and_resolve_final_hop(
    monkeypatch,
):
    paragraphs = [
        {
            "title": f"Title {index}",
            "paragraph_text": f"Document {index}.",
            "is_supporting": index in {2, 7},
        }
        for index in range(20)
    ]
    rows = [
        {
            "question": "Who is the spouse of the Green performer?",
            "answer": "Miquette Giraudy",
            "paragraphs": paragraphs,
            "question_decomposition": [
                {
                    "question": "Green >> performer",
                    "answer": "Steve Hillage",
                    "paragraph_support_idx": 7,
                },
                {
                    "question": "#1 >> spouse",
                    "answer": "Miquette Giraudy",
                    "paragraph_support_idx": 2,
                },
            ],
        }
    ]
    monkeypatch.setattr(musique_module, "_load_musique", lambda *_args: rows)

    multi_hop = next(
        musique_generator(
            num_samples=1,
            num_documents=0,
            support_preserving_num_documents=5,
            question_mode="multi_hop",
        )
    )
    direct = next(
        musique_generator(
            num_samples=1,
            num_documents=0,
            support_preserving_num_documents=5,
            question_mode="direct_final_hop",
        )
    )

    assert multi_hop.documents == direct.documents
    assert multi_hop.meta["document_indices"] == [0, 1, 2, 3, 7]
    assert set(multi_hop.meta["support_indices"]) <= set(multi_hop.meta["document_indices"])
    assert len(multi_hop.documents) == 5
    assert multi_hop.query == "Who is the spouse of the Green performer?"
    assert direct.query == ('What is the value of the relation "spouse" for "Steve Hillage"?')
    assert direct.answer == "Miquette Giraudy"
    assert direct.answer not in direct.query
    assert direct.meta["resolved_final_step"] == "Steve Hillage >> spouse"


def test_musique_all_documents_and_support_preserving_views_share_rows(
    monkeypatch,
):
    def row(index: int, count: int):
        return {
            "question": f"Question {index}?",
            "answer": f"Answer {index}",
            "paragraphs": [
                {
                    "paragraph_text": f"Document {index}-{paragraph}.",
                    "is_supporting": paragraph in {0, 1},
                }
                for paragraph in range(count)
            ],
            "question_decomposition": [
                {"question": "Entity >> relation", "answer": f"Answer {index}"}
            ],
        }

    rows = [row(0, 20), row(1, 18), row(2, 20), row(3, 19), row(4, 20)]
    monkeypatch.setattr(musique_module, "_load_musique", lambda *_args: rows)
    five_document_samples = list(
        musique_generator(
            num_samples=3,
            num_documents=0,
            seed=7,
            support_preserving_num_documents=5,
        )
    )
    all_document_samples = list(
        musique_generator(
            num_samples=3,
            num_documents=0,
            seed=7,
        )
    )

    assert [sample.meta["index"] for sample in five_document_samples] == [
        sample.meta["index"] for sample in all_document_samples
    ]
    assert all(len(sample.documents) == 5 for sample in five_document_samples)
    assert [len(sample.documents) for sample in all_document_samples] == [
        len(rows[sample.meta["index"]]["paragraphs"]) for sample in all_document_samples
    ]


@pytest.mark.parametrize(
    ("num_documents", "expected_group_sizes"),
    [(18, [4, 4, 4, 4, 2]), (19, [4, 4, 4, 4, 3]), (20, [4, 4, 4, 4, 4])],
)
def test_document_packet_grouping_preserves_exact_token_stream(
    num_documents,
    expected_group_sizes,
):
    class RoundTripTokenizer:
        def __call__(self, text, *, add_special_tokens=False):
            assert not add_special_tokens
            return {"input_ids": list(text.encode("utf-8"))}

        def decode(
            self,
            token_ids,
            *,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ):
            assert not skip_special_tokens
            assert not clean_up_tokenization_spaces
            return bytes(token_ids).decode("utf-8")

    sample = RetrievalSample(
        preamble="prefix",
        documents=[f"document-{index}" for index in range(num_documents)],
        task_prompt="question",
    )
    tokenizer = RoundTripTokenizer()
    original_ids = [
        token_id
        for document in sample.documents
        for token_id in tokenizer(document, add_special_tokens=False)["input_ids"]
    ]

    grouped = _group_document_packets(
        sample,
        tokenizer=cast(Any, tokenizer),
        max_documents_per_packet=4,
    )
    assert all(isinstance(document, list) for document in grouped.documents)
    grouped_ids = [token_id for document in grouped.documents for token_id in document]

    assert len(grouped.documents) == 5
    assert grouped.meta["source_document_count"] == num_documents
    assert grouped.meta["packet_group_sizes"] == expected_group_sizes
    assert grouped.meta["max_documents_per_packet"] == 4
    assert grouped.meta["packet_grouping_token_exact"] is True
    assert grouped_ids == original_ids


def test_multinews_generator_splits_sources_and_filters_long_rows(monkeypatch):
    rows = [
        {
            "document": "Article A|||||Article B",
            "summary": "A combined summary.",
        },
        {
            "document": "Only one article",
            "summary": "Not eligible.",
        },
        {
            "document": "X" * 40 + "|||||Y" * 40,
            "summary": "Too long.",
        },
        {
            "document": "Article C|||||Article D|||||Article E",
            "summary": "Another combined summary.",
        },
    ]
    monkeypatch.setattr(multinews_module, "_load_multinews", lambda *_args: rows)
    samples = list(
        multinews_generator(
            num_samples=2,
            num_documents=0,
            seed=3,
            max_total_chars=50,
        )
    )
    assert {tuple(sample.documents) for sample in samples} == {
        ("Article A", "Article B"),
        ("Article C", "Article D", "Article E"),
    }
    assert all("150-250-word news summary" in sample.task_prompt for sample in samples)
    assert all(sample.meta["num_documents"] >= 2 for sample in samples)
    with pytest.raises(ValueError, match="num_documents=0"):
        list(multinews_generator(1, num_documents=2))


def test_niah_fixed_partition_chunking_and_few_shot(monkeypatch):
    rows = _niah_rows()
    monkeypatch.setattr(niah_module, "_load_niah", lambda *_args: rows)
    monkeypatch.setattr(
        niah_module,
        "split_sentences",
        lambda text, max_chunk_size: [text[:max_chunk_size], text[max_chunk_size:]],
    )
    train = list(
        niah_generator(
            num_samples=3,
            num_documents=0,
            num_shots=1,
            split="train",
            seed=9,
            chunk_size=12,
        )
    )
    test = list(
        niah_generator(
            num_samples=3,
            num_documents=0,
            split="test",
            seed=9,
            chunk_size=12,
        )
    )
    assert {sample.meta["index"] for sample in train}.isdisjoint(
        {sample.meta["index"] for sample in test}
    )
    assert all(sample.preamble for sample in train)
    assert all(len(sample.documents) == 2 for sample in train + test)
    assert all("Short Answer:" in sample.task_prompt for sample in train + test)
    assert all(" " in sample.answer for sample in train + test)


def test_registry_wrapper_dispatches_without_downloading(monkeypatch):
    rows = _musique_rows()
    monkeypatch.setattr(musique_module, "_load_musique", lambda *_args: rows)
    samples = list(
        get_ret_eval_generator(
            name="musique",
            num_samples=2,
            num_documents=0,
            num_shots=0,
            subset="default",
            split="validation",
            seed=2,
        )
    )
    assert len(samples) == 2
    assert all(sample.documents for sample in samples)


def test_registry_selection_offset_returns_disjoint_ordered_slice(monkeypatch):
    rows = _musique_rows()
    monkeypatch.setattr(musique_module, "_load_musique", lambda *_args: rows)
    common: dict[str, Any] = {
        "name": "musique",
        "num_documents": 0,
        "num_shots": 0,
        "subset": "default",
        "split": "validation",
        "seed": 2,
    }
    first_four = list(get_ret_eval_generator(num_samples=4, **common))
    offset_two = list(
        get_ret_eval_generator(
            num_samples=2,
            data_kwargs={"selection_offset": 2},
            **common,
        )
    )
    assert [sample.meta["index"] for sample in offset_two] == [
        sample.meta["index"] for sample in first_four[2:4]
    ]


def test_answer_postprocessors_extract_expected_short_answers():
    assert hotpot_qa_answer_postprocess("Reasoning. Short Answer: Paris.", "PARIS.") == (
        "paris",
        "paris",
    )
    assert hotpot_qa_answer_postprocess("The answer is Midnight Oil.", "Midnight Oil") == (
        "midnight oil",
        "midnight oil",
    )
    assert hotpot_qa_answer_postprocess(
        "Reasoning. The final answer is: flotilla.", "flotilla"
    ) == ("flotilla", "flotilla")
    assert hotpot_qa_answer_postprocess("Armenian", "Armenian") == ("", "armenian")
    assert musique_answer_postprocess("Steps... Answer: Ada Lovelace.", "ada lovelace") == (
        "ada lovelace",
        "ada lovelace",
    )
    assert musique_answer_postprocess("The final answer is: Mount Everest.", "Mount Everest") == (
        "mount everest",
        "mount everest",
    )
    assert musique_answer_postprocess("Armenian", "Armenian") == ("", "armenian")
    assert musique_answer_postprocess(
        "<think>The answer might be wrong.</think>\nAnswer: Correct.",
        "Correct",
    ) == ("correct", "correct")
    assert musique_answer_postprocess(
        "<think>The answer might be Correct, but I am still reasoning...",
        "Correct",
    ) == ("", "correct")
    assert niah_answer_postprocess("Short Answer: Red, Blue and Green.", "red blue green") == (
        "red blue green",
        "red blue green",
    )


def test_sentence_packing_does_not_duplicate_first_sentence(monkeypatch):
    import nltk

    monkeypatch.setattr(nltk, "sent_tokenize", lambda _text: ["First.", "Second.", "Third."])
    assert split_sentences("ignored", max_chunk_size=15) == [
        "First. Second.",
        "Third.",
    ]
    monkeypatch.setattr(nltk, "sent_tokenize", lambda _text: ["x" * 35])
    chunks = split_sentences("ignored", max_chunk_size=10)
    assert "".join(chunks) == "x" * 35
    assert all(len(chunk) <= 10 for chunk in chunks)
    with pytest.raises(ValueError, match="positive"):
        split_sentences("text", max_chunk_size=0)
