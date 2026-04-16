from typing import Iterator
import warnings
import random
import re
from datasets import load_dataset, Dataset
from .abc import RetEvalEntry #, RawPICEvalEntry, Conversation


def hotpot_qa_ret_eval_generator(
    num_samples: int,
    num_data_strs: int,
    num_shots: int,
    subset: str = "fullwiki",
    split: str = "validation",
    seed: int = 42,
    **kwargs
) -> Iterator[RetEvalEntry]:
    """
    Generates evaluation entries for the HotpotQA dataset. Each document is the concatenation
    of sentences from the context provided in the dataset.

    Args:
        num_samples (int): Number of samples to generate.
        num_data_strs (int): Number of data strings to include in each entry (not used here).
        num_shots (int): Number of few-shot examples to include in the preamble.
        subset (str): Subset of the HotpotQA dataset to use ("fullwiki" or "distractor").
        split (str): Split of the dataset to use ("train", "validation", or "test").
        seed (int): Random seed for shuffling the dataset.
        **kwargs: Additional keyword arguments.
        - add_instruction (bool): Whether to add instructions to the question prompt. Default is True.
        - add_cot (bool): Whether to add chain-of-thought prompting to the question prompt. Default is True.
    
    Yields:
        RetEvalEntry: An evaluation entry containing preamble, documents, task prompt, query, and answer.
    
    Note:
        - The hotpot QA dataset should be used by an instruction-tuned model as it requires multi-hop reasoning.
        - Few-shot prompting is not recommended for this dataset as the multi-hop content is missing in the context.
        - num_data_strs is not used in this generator since each document has its own sentences. The number of documents
            is determined by the number of sentences in the context.
    """
    add_inst = kwargs.pop("add_inst", True)
    add_cot = kwargs.pop("add_cot", True)
    difficulty: list[str]|None = kwargs.pop("difficulty", None)

    if difficulty is not None:
        difficulty = ["easy", "medium", "hard"]
    if kwargs:
        warnings.warn(f"Unused kwargs in hotpot_qa_eval_generator: {kwargs}")

    ds_split = load_dataset("hotpotqa/hotpot_qa", subset, split=split)
    assert isinstance(ds_split, Dataset)

    # ds_split = ds[split]
    ds_len = len(ds_split)

    all_indices = list(range(ds_len))
    random.seed(seed)
    random.shuffle(all_indices)

    def format_data_str(index: int) -> list[str]:
        item = ds_split[index]
        context = item['context']
        sentences = context['sentences']
        flattened_sentences = [
            "".join(sent) for sent in sentences
        ]
        return flattened_sentences

    if num_shots > 0:
        warnings.warn("few_shot_str is not recommended for HotpotQA")
        few_shot_indices = all_indices[:num_shots]
        all_indices = all_indices[num_shots:]
        few_shot_strs = []
        for idx in few_shot_indices:
            item = ds_split[idx]
            question = item['question']
            answer = item['answer']
            context_strs = format_data_str(idx)
            few_shot_str = f"Context: {' '.join(context_strs)}\nQuestion: {question}\nAnswer: {answer}\n"
            few_shot_strs.append(few_shot_str)
        few_shot_str = "\n".join(few_shot_strs)
    else:
        few_shot_str = ""

    if num_samples > len(all_indices):
        warnings.warn(f"num_samples ({num_samples}) is greater than dataset size ({len(all_indices)}). Reducing num_samples to dataset size.")
        num_samples = len(all_indices)

    for _ in range(num_samples):
        idx, all_indices = next_valid_sample(ds_split, all_indices, difficulty)
        item = ds_split[idx]
        question_str = item['question']
        if add_inst:
            question_str = f"Answer the following question based on the provided context.\n{question_str}"
        if add_cot:
            question_str += "You should get the final answer by thinking step by step.\n"
        if add_inst:
            question_str += "Your response should end with: 'Short Answer: <your final answer>'.\n"

        answer_str = item['answer']
        data_strs = format_data_str(idx)
        yield RetEvalEntry(
            preamble=few_shot_str,
            documents=data_strs,
            task_prompt=question_str,
            query=item['question'],
            answer=answer_str,
        )


def next_valid_sample(
    ds: Dataset,
    indices: list[int],
    difficulty: list[str]|None = None,
) -> tuple[int, list[int]]:
    counter = 0
    while counter < len(indices):
        idx = indices[counter]
        sample = ds[idx]
        level = sample['level']
        if difficulty is None or level in difficulty:
            return idx, indices[counter + 1 :]
        counter += 1
    raise ValueError("No valid sample found with the specified difficulty levels.")

            
def hotpot_qa_answer_postprocess(pred_answer: str, gold_answer: str) -> tuple[str, str]:
    parts = re.split(r'(?i)Answer\s*:', pred_answer)
    if len(parts) > 1:
        # We take the last part to get the content after the separator
        pred_answer = parts[-1].lower().strip().rstrip('.')
    else:
        pred_answer = ""

    gold_answer = gold_answer.lower().strip().rstrip('.')
    return pred_answer, gold_answer