from typing import TypedDict, Protocol, Iterator, Callable, Literal, TypeAlias

class RetEvalEntry(TypedDict):
    """
    An evaluation entry for Retrieval-Augmented Generation (RAG) tasks. 

    Attributes:
        preamble (str): The preamble text, often containing few-shot examples and previous context.
        documents (list[str]): A list of retrieved documents relevant to the query.
        task_prompt (str): The prompt after the documents for generating the answer.
        query (str): The query for the retrieval task.
        answer (str): The expected answer to the query.
    
    Note:
        - The attributes are organized in the format: "[preamble][document_1][document_2]... [task_prompt]".
        - The KV caches for the preamble is assumed to be pre-computed.
        - In evaluation, the documents in the EvalEntry are not used. Instead, they are retrieved based on the query.
        - The documents are provided for training purposes.
    """
    preamble: str
    documents: list[str]
    task_prompt: str
    query: str
    answer: str


class RetEvalGeneratorFunc(Protocol):
    def __call__(
        self,
        num_samples: int,
        num_data_strs: int,
        num_shots: int,
        subset: str,
        split: str,
        seed: int,
        **kwargs: dict
    ) -> Iterator[RetEvalEntry]:
        ...
