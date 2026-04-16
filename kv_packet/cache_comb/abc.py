from typing import Protocol, TypedDict, Callable
from transformers import GenerationConfig, LlamaForCausalLM
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from ..cache import KVCache
from ..model import SupportedModel

TokenizerType = PreTrainedTokenizer | PreTrainedTokenizerFast

class ResultDict(TypedDict):
    ttft: float
    tp: int
    fp: int
    fn: int
    flops: int


class EvalCombFunc(Protocol):
    def __call__(
        self,
        model: SupportedModel,
        tokenizer: TokenizerType,
        generation_config: GenerationConfig|None,
        preamble: str,
        documents: list[str],
        task_prompt: str,
        document_kvs: list[KVCache],
        answer: str,
        answer_postprocess_func: Callable[[str, str], tuple[str, str]]|None = None,
        kwargs: dict|None = None
    ) -> ResultDict:
        """
        Functions to combine data cache given a context.

        The input to the model should be given in the format of:
        [preamble][documents_1][documents_2]...[documents_n][question]

        The model will generate after the question, and the output will be compared to the answer.
        For instruction-tuned models, the input format should be included in the context and question.

        For example:
        Context: "<|begin_of_text|><|start_header_id|>....user<|end_header_id|>\n\n[question_str]
        Question: "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        """
        ...
