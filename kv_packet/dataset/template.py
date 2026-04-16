from typing import Protocol
from warnings import warn
from .abc import RetEvalEntry


class TemplateFunc(Protocol):
    def __call__(
        self,
        eval_entry: RetEvalEntry,
        **kwargs,
    ) -> RetEvalEntry:
        """
        A template function that takes in a EvalEntry, which consists of a context string,
        data strings and question string, and returns a new EvalEntry with formatted context and question strings.

        It is used to format the convert the raw input strings into the desired
        format for instruct models.
        """
        ...


def default_template(
    eval_entry: RetEvalEntry,
    **kwargs,
) -> RetEvalEntry:
    """ 
    A default template function that returns the context and question strings
    """
    return eval_entry


def llama_chat_template(
    eval_entry: RetEvalEntry,
    system_prompt: str = "",
    **kwargs,
) -> RetEvalEntry:
    begin_str = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
    if system_prompt:
        begin_str += f"{system_prompt}"

    begin_str += "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    context_str = eval_entry["preamble"]
    question_str = eval_entry["task_prompt"]

    begin_str += context_str
    end_str = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    end_str = f"{question_str}{end_str}"

    return RetEvalEntry(
        preamble=begin_str,
        documents=eval_entry["documents"],
        task_prompt=end_str,
        query=eval_entry["query"],
        answer=eval_entry["answer"]
    )


def qwen_3_chat_template(
    eval_entry: RetEvalEntry,
    system_prompt: str = "",
    **kwargs,
) -> RetEvalEntry:
    """
    Qwen 3 (and 2.5) template:
    <|im_start|>system\n{System}\n<|im_end|>\n<|im_start|>user\n{Context}{Docs}{Question}<|im_end|>\n<|im_start|>assistant\n
    """
    begin_str = ""

    if system_prompt:
        begin_str += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

    begin_str += "<|im_start|>user\n"

    context_str = eval_entry["preamble"]
    question_str = eval_entry["task_prompt"]

    begin_str += f"{context_str}"
    end_str = f"{question_str}<|im_end|>\n<|im_start|>assistant\n"

    return RetEvalEntry(
        preamble=begin_str,
        documents=eval_entry["documents"],
        task_prompt=end_str,
        query=eval_entry["query"],
        answer=eval_entry["answer"]
    )


TEMPLATE_FUNC_DICT: dict[str, TemplateFunc] = {
    "default": default_template,
    "llama_chat": llama_chat_template,
    "qwen_3_chat": qwen_3_chat_template,
}
