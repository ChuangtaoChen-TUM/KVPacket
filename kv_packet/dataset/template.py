from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from .abc import RetEvalEntry

TokenizerType = PreTrainedTokenizer | PreTrainedTokenizerFast


def apply_chat_template(
    eval_entry: RetEvalEntry,
    tokenizer: TokenizerType,
    system_prompt: str = "",
) -> RetEvalEntry:
    context_str = eval_entry["preamble"]
    question_str = eval_entry["task_prompt"]

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{{DOCUMENT_PLACEHOLDER}}"},
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(formatted, str)

    pivot = f"{{DOCUMENT_PLACEHOLDER}}"
    start_idx = formatted.index(pivot)
    preamble = formatted[:start_idx] + context_str
    task_prompt = question_str + formatted[start_idx + len(pivot):]

    return RetEvalEntry(
        preamble=preamble,
        documents=eval_entry["documents"],
        task_prompt=task_prompt,
        query=eval_entry["query"],
        answer=eval_entry["answer"],
    )
