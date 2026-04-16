from typing import TypedDict, TypeAlias
import torch
from transformers import GenerationConfig
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from ..model import SupportedModel

TokenizerType: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast

class GenerationOutput(TypedDict):
    sequences: list[torch.Tensor] # (num_seq) [generated_seq_len]
    logits: list[torch.Tensor]    # (num_seq) [generated_seq_len, vocab_size]
    text: list[str]               # (num_seq) strings


class GenerateCacheStateDict(TypedDict):
    cache: dict[str, GenerationOutput]
    device: str|None


class GenerationCache:
    """ Cache for storing generation outputs to avoid redundant computations. """
    def __init__(
        self,
        device: torch.device|None = None,
    ):
        self.cache: dict[str, GenerationOutput] = {}
        self.device = device


    def get(self, content: str, device: torch.device|None=None) -> GenerationOutput | None:
        generation = self.cache.get(content, None)
        if generation is None:
            return None
        if device is not None and self.device != device:
            generation = GenerationOutput(
                sequences=[seq.to(device) for seq in generation["sequences"]],
                logits=[logit.to(device) for logit in generation["logits"]],
                text=generation["text"],
            )
        return generation


    def add(self, content: str, generation: GenerationOutput):
        """ Add a generation output to the cache. """
        if self.device is not None:
            generation["sequences"] = [seq.to(self.device) for seq in generation["sequences"]]
            generation["logits"] = [logit.to(self.device) for logit in generation["logits"]]
        self.cache[content] = generation


    def __contains__(self, content: str) -> bool:
        """ Check if a generation output is in the cache. """
        return content in self.cache


    def to_state_dict(self) -> GenerateCacheStateDict:
        """ Convert the cache to a state dictionary for saving. """
        return GenerateCacheStateDict(
            cache=self.cache,
            device=str(self.device) if self.device is not None else None,
        )


    @classmethod
    def from_state_dict(cls, state_dict: GenerateCacheStateDict) -> "GenerationCache":
        """ Create a GenerationCache from a state dictionary. """
        device = torch.device(state_dict["device"]) if state_dict["device"] is not None else None
        gen_cache = cls(device=device)
        for content, generation in state_dict["cache"].items():
            gen_cache.add(content, generation)
        return gen_cache


    @classmethod
    def load_from_file(cls, path: str, device: torch.device|None=None) -> "GenerationCache":
        """ Load a GenerationCache from a file. """
        state_dict = torch.load(path)
        if device is not None:
            state_dict["device"] = str(device)
        return cls.from_state_dict(state_dict)


def get_generation(
    model: SupportedModel,
    tokenizer: TokenizerType,
    input_strs: list[str]|None=None,
    input_ids: torch.Tensor|None=None,
    input_embeds: torch.Tensor|None=None,
    attention_mask: torch.Tensor|None=None,
    generation_config: GenerationConfig|None=None,
) -> GenerationOutput:
    """
    Generate text using the model and tokenizer.
    
    Args:
        model: The language model to use for generation.
        tokenizer: The tokenizer corresponding to the model.
        input_strs: A list of input strings to generate text from.
        generation_config: Optional generation configuration.
    
    Returns:
        A GenerationOutput containing sequences, logits, and text.
    """
    assert isinstance(tokenizer.pad_token_id, int)
    assert isinstance(tokenizer.eos_token_id, int)

    if input_strs is None and input_ids is None and input_embeds is None:
        raise ValueError("At least one of input_strs, input_ids, or input_embeds must be provided.")

    if int(input_ids is not None) + int(input_embeds is not None) + int(input_strs is not None) != 1:
        raise ValueError("Only one of input_strs, input_ids, or input_embeds should be provided.")

    if input_strs is not None:
        inputs = tokenizer(
            input_strs,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(model.device) # [batch_size, seq_len]

        _input_ids = inputs["input_ids"]
        assert isinstance(_input_ids, torch.Tensor)
        input_ids = _input_ids

        if attention_mask is None:
            _attention_mask = inputs["attention_mask"]
            assert isinstance(_attention_mask, torch.Tensor)
            attention_mask = _attention_mask

    if generation_config is None:
        generation_config = model.generation_config

    generation_config.return_dict_in_generate = True
    generation_config.output_logits = True

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            generation_config=generation_config,
        )
    
    assert isinstance(generation_output, GenerateDecoderOnlyOutput)
    sequence = generation_output.sequences # [batch_size, seq_len]
    logits = generation_output.logits # tuple (gen_len) of [batch_size, vocab_size]
    
    assert logits is not None
    logit_tensor = torch.stack(logits, dim=1) # [batch_size, gen_len, vocab_size]

    if input_ids is not None:
        num_seq = input_ids.size(0)
        start_index = input_ids.size(1)
    elif input_embeds is not None:
        num_seq = input_embeds.size(0)
        start_index = 0
    else:
        raise ValueError("Either input_ids or input_embeds must be provided.")

    generation = GenerationOutput(
        sequences=[],
        logits=[],
        text=[]
    )

    for i in range(num_seq):
        seq = sequence[i][start_index:] # [generated_seq_len]
        logits_t = logit_tensor[i]      # [gen_len, vocab_size]
        eos_indices = (seq == tokenizer.eos_token_id).nonzero()
        if eos_indices.numel() > 0:
            end_index = int(eos_indices[0].item()) + 1
            seq = seq[:end_index]
            logits_t = logits_t[:end_index]

        text = tokenizer.decode(seq, skip_special_tokens=False)
        assert isinstance(text, str)
        generation["sequences"].append(seq)
        generation["logits"].append(logits_t)
        generation["text"].append(text)
    return generation


def get_answers(
    generated_tokens: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: TokenizerType
) -> list[str]:
    """ Extract answers from generated tokens based on input IDs and tokenizer. """
    num_seq = input_ids.size(0)
    start_index = input_ids.size(1)
    answers: list[str] = []
    for i in range(num_seq):
        generation = generated_tokens[i][start_index:]
        eos_indices = (generation == tokenizer.eos_token_id).nonzero() # type: ignore
        if eos_indices.numel() > 0:
            end_index = eos_indices[0].item()
            generation = generation[:end_index]
        generated_text = tokenizer.decode(generation, skip_special_tokens=True)
        assert isinstance(generated_text, str)
        answers.append(generated_text)
    return answers
