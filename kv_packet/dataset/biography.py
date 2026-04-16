import random
import warnings
import pandas as pd
import numpy as np
from typing import Iterator
from datasets import load_dataset, DatasetDict, Dataset
from .abc import RetEvalEntry
from .utils import clean_with_prefixes

BIO_DATASET: dict[str, DatasetDict|None] = {}
TRAIN_RATIO = 0.5
BIO_ATTRIBUTES = (
    'birth_date', 'birth_place', 'current_location',
    'university', 'major', 'company',
)

def bio_ret_eval_generator(
    num_samples: int,
    num_data_strs: int,
    num_shots: int,
    subset: str = "1k",
    split: str = "train",
    seed: int = 42,
    **kwargs
) -> Iterator[RetEvalEntry]:
    """
    Generates evaluation entries for the biography dataset.

    Args:
        num_samples (int): Number of samples to generate.
        num_data_strs (int): Number of data strings to include in the documents entry.
        num_shots (int): Number of few-shot examples to include in the preamble.
        subset (str): Subset of the biography dataset to use.
        split (str): Split of the dataset to use ("train" or "test").
        seed (int): Random seed for reproducibility.
        **kwargs: Additional keyword arguments.
        - cache_dataset (bool): Whether to cache the loaded dataset for future use.
        - question_type (str): Type of question to generate ("completion" or "QA").
            This will affect the format of the task prompt, preamble and query. For "QA",
            the question will be drawn from a predefined set of question templates based on the attribute.
    
    Yields:
        RetEvalEntry: An evaluation entry containing preamble, documents, task prompt, query, and answer.

    Example:
        num_shots = 2, question_type = "completion"
            preamble = "Ian Anderson's was born .... Ian Anderson's is living ... ## Ian Anderson's genesis happened on ...
                Keith Ortiz was ...  ## Keith Ortiz's origin unfolded on ... "
            task_prompt = "## Jonathon Gross entered existence on"
            query = "## Jonathon Gross entered existence on"

        num_shots = 2, question_type = "QA"
            preamble = "Ian Anderson's was born .... Ian Anderson's is living ... Which university did Ian Anderson attend? Answer: ...
                Keith Ortiz was ...  Where was Keith Ortiz born? Answer: ..."
            task_prompt = "What is the birth date of Jonathon Gross?"
            query = "What is the birth date of Jonathon Gross?"

        num_shots = 0
            preamble = ""
    
        num_data_strs = 2
            documents = ["Jonathon Gross arrived ... ", "Courtney Vang was manifested in ... "]
    
    Note:
        For the biography dataset, we split both the train and test splits of the original dataset
        into two parts using TRAIN_RATIO. This is because the original train set at Hugginface only contains
        the documents, and test set only contains the questions for the same entities.
        Thus, the "train" and "test" splits here do not correspond directly to the original dataset splits at
        Huggingface.
    """
    cache_dataset = kwargs.pop("cache_dataset", True)
    question_type = kwargs.pop("question_type", "completion")

    if kwargs:
        warnings.warn(f"Unused kwargs in bio_eval_generator: {kwargs}")

    if cache_dataset:
        cached_ds = BIO_DATASET.get(subset, None)
    else:
        cached_ds = None

    if cached_ds is not None:
        ds = cached_ds
    else:
        ds = load_dataset("alex-karev/biographies", subset)
        assert isinstance(ds, DatasetDict)
        ds = filter_bio_dataset(ds)

    if cache_dataset:
        BIO_DATASET[subset] = ds

    ds_len = len(ds['test'])

    train_split_index = int(ds_len * TRAIN_RATIO)
    if split == "train":
        all_indices = list(range(train_split_index))
    elif split == "test":
        all_indices = list(range(train_split_index, ds_len))
    else:
        raise ValueError(f"Unknown split: {split}")

    rng = random.Random(seed)
    rng.shuffle(all_indices)

    few_shot_indices = all_indices[-num_shots:] if num_shots > 0 else []
    all_indices = all_indices[:-num_shots] if num_shots > 0 else all_indices

    def get_data_str(index: int) -> str:
        item = ds['train'][index]
        data_str = " ".join([
            rng.choice(
                item[attr]
            ) for attr in BIO_ATTRIBUTES
        ])
        return data_str


    def get_question_answer(index: int) -> tuple[str, str]:
        item = ds['test'][index]
        rand_key = rng.choice(BIO_ATTRIBUTES)
        if question_type == "completion":
            question = rng.choice(item[rand_key])
            question = f"## {question.rstrip()}"
        elif question_type == "QA":
            name = item['name']
            question = format_question(rand_key, name, rng)
        else:
            raise ValueError(f"Unknown question_type: {question_type}")

        answer = item['labels'][rand_key]
        return question, answer


    few_shot_strs: list[str] = []

    for fs_index in few_shot_indices:
        fs_data_str = get_data_str(fs_index)
        fs_question, fs_answer = get_question_answer(fs_index)
        few_shot_strs.append(f"{fs_data_str} {fs_question} {fs_answer}.")

    if few_shot_strs:
        few_shot_str = "\n".join(few_shot_strs) + "\n"
    else:
        few_shot_str = ""

    for i in range(num_samples):
        idx = all_indices[i]
        question_str, answer_str = get_question_answer(idx)
        data_indices = rng.sample(all_indices, num_data_strs)
        if idx not in data_indices and num_data_strs > 0:
            data_indices[0] = idx
        data_strs = [get_data_str(idx) for idx in data_indices]
        yield RetEvalEntry(
            preamble=few_shot_str,
            documents=data_strs,
            task_prompt=question_str,
            query=question_str,
            answer=answer_str,
        )
    return


def filter_bio_dataset(
    dataset: DatasetDict,
):
    """
    There are some entries in the biography dataset that have missing or invalid attributes.
    This function filters out such entries from both the train and test splits.
    Args:
        dataset (DatasetDict): The original biography dataset.
    """
    train_split = dataset['train']
    test_split = dataset['test']

    train_split_pd = train_split.to_pandas()
    test_split_pd = test_split.to_pandas()

    assert isinstance(train_split_pd, pd.DataFrame)
    assert isinstance(test_split_pd, pd.DataFrame)

    def is_valid(item: pd.Series) -> bool:
        birth_dates = item['birth_date']
        assert isinstance(birth_dates, np.ndarray)
        if len(birth_dates) == 0:
            return False
        first_str = birth_dates[0]
        if all(c == first_str for c in birth_dates):
            return False
        return True

    indices_to_keep = []

    for i in range(len(test_split_pd)):
        item = test_split_pd.iloc[i]
        if is_valid(item):
            indices_to_keep.append(i)

    print(f"Filtered dataset from {len(train_split_pd)} to {len(indices_to_keep)} items")
    filtered_train_split_pd = train_split_pd.iloc[indices_to_keep]
    filtered_train_split = Dataset.from_pandas(filtered_train_split_pd)
    filtered_test_split_pd = test_split_pd.iloc[indices_to_keep]
    filtered_test_split = Dataset.from_pandas(filtered_test_split_pd)

    return DatasetDict({
        'train': filtered_train_split,
        'test': filtered_test_split,
    })


def format_question(
    attribute: str,
    name: str,
    rng: random.Random|None = None
):
    question_template = {
        "birth_date": [
            "What is the birth date of {name}?",
            "When was {name} born?",
            "When did {name}'s birth occur?",
            "When did {name} come into this world?",
        ],
        "birth_place": [
            "Where was {name} born?",
            "Which place is the birthplace of {name}?",
            "What is the birthplace of {name}?",
            "Where did {name} originate from?",
        ],
        "current_location": [
            "Where is {name} currently located?",
            "What is the current location of {name}?",
            "Where can I find {name} now?",
            "Where does {name} live currently?",
            "Where is {name} residing at present?",
        ],
        "university": [
            "Which university did {name} attend?",
            "What university did {name} go to?",
            "Where did {name} study in higher education?",
            "Where did {name} go for higher education?",
        ],
        "major": [
            "What was {name}'s major in university?",
            "What did {name} study in university?",
            "What field did {name} specialize in at university?",
            "In which subject did {name} major during university?",
        ],
        "company": [
            "Which company does {name} work for?",
            "Where is {name} employed?",
            "What is the name of the company where {name} works?",
            "At which company is {name} employed?",
        ],
    }

    templates = question_template.get(attribute, [])
    if not templates:
        raise ValueError(f"Unknown attribute: {attribute}")

    if rng is None:
        rng = random.Random()
    template = rng.choice(templates)
    question = template.format(name=name)
    question += " Respond with the answer only. Answer:"
    return question


def biography_answer_postprocess(
    pred_answer: str,
    gold_answer: str
) -> tuple[str, str]:
    
    # 1. Normalize basic formatting
    pred_answer = pred_answer.lower().strip().rstrip(".")
    gold_answer = gold_answer.lower().strip().rstrip(".")

    # 2. List common chatty prefixes to remove
    # (Order matters: longer phrases first to avoid partial cuts)
    prefixes = [
        "studied",
        "attended",
        "works for",
        "university was",
        "employed by",
        " in ",
        " at ",
        " on ",
        " was ",
        " from ",
        " is ",
    ]

    pred_answer = clean_with_prefixes(pred_answer, prefixes).strip(" :.,")

    return pred_answer, gold_answer