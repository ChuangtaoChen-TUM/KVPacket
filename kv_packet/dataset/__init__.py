from typing import Iterator, Callable
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .abc import (
    RetEvalGeneratorFunc,
    RetEvalEntry,
)
from .biography import (
    bio_ret_eval_generator,
    biography_answer_postprocess
)
from .hotpot_qa import (
    hotpot_qa_ret_eval_generator,
    hotpot_qa_answer_postprocess
)
from .niah import (
    niah_ret_eval_generator,
    niah_answer_postprocess
)
from .template import TEMPLATE_FUNC_DICT

__all__ = [
    "get_ret_eval_generator",
]


RET_EVAL_GENERATOR_DICT: dict[str, RetEvalGeneratorFunc] = {
    "biography": bio_ret_eval_generator,
    "hotpot_qa": hotpot_qa_ret_eval_generator,
    "niah": niah_ret_eval_generator,
}


ANSWER_POSTPROCESS_DICT: dict[str, Callable[[str, str], tuple[str, str]]] = {
    "biography": biography_answer_postprocess,
    "hotpot_qa": hotpot_qa_answer_postprocess,
    "niah": niah_answer_postprocess,
}

def get_ret_eval_generator(
    name: str,
    num_samples: int,
    num_data_strs: int,
    num_shots: int,
    subset: str,
    split: str,
    seed: int,
    template: str|None = None,
    data_kwargs: dict|None = None,
    template_kwargs: dict|None = None,
) -> Iterator[RetEvalEntry]:
    """ Get a retrieval evaluation generator

    Warning: Not all parameters can be chosen freely for each dataset. Please refer to the documentation of each dataset for more details.

    Args:
        name (str): The name of the evaluation generator. Should be one of the keys in RET_EVAL_GENERATOR_DICT.
        num_samples (int): The number of evaluation samples to generate.
        num_data_strs (int): The number of data strings to include in each sample.
        num_shots (int): The number of few-shot examples to include in the preamble. If 0, no few-shot examples are included.
        subset (str): The subset of the dataset to use. This is dataset-specific and will be passed to the evaluation generator.
        split (str): The split of the dataset to use ("train", "test"). This is dataset-specific and will be passed to the evaluation generator.
        seed (int): The random seed for shuffling the dataset. This is dataset-specific and will be passed to the evaluation generator.
        template (str|None): The name of the chat template function to apply to each evaluation entry. Should be one of the keys in TEMPLATE_FUNC_DICT.
            If None, no template function is applied.
        data_kwargs (dict|None): Additional keyword arguments to pass to the evaluation generator.
        template_kwargs (dict|None): Additional keyword arguments to pass to the template function.
    """
    template = template or "default"

    if not (template_func := TEMPLATE_FUNC_DICT.get(template)):
        raise ValueError(f"Unknown template: {template}")

    if not (eval_generator := RET_EVAL_GENERATOR_DICT.get(name)):
        raise ValueError(f"Unknown eval generator: {name}")

    def wrapped_eval_generator() -> Iterator[RetEvalEntry]:
        for eval_entry in eval_generator(
            num_samples=num_samples,
            num_data_strs=num_data_strs,
            num_shots=num_shots,
            subset=subset,
            split=split,
            seed=seed,
            **(data_kwargs or {}),
        ):
            yield template_func(eval_entry, **(template_kwargs or {}))

    return wrapped_eval_generator()
