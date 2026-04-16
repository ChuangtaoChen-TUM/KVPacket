from torch import Tensor
import torch

class TensorCache:
    """
    Cache for storing hidden states at multiple layers.

    Hidden state shape: (batch_size, seq_len, hidden_dim)
    """
    def __init__(self):
        self._cache: dict[int, list[Tensor]] = {}


    @property
    def layers(self) -> list[int]:
        """ Get the list of layer indices in the cache """
        return list(self._cache.keys())


    def update(self, layer: int, hidden_state: Tensor):
        """ Add a new hidden state to the cache for a specific layer """
        if layer not in self._cache:
            self._cache[layer] = []
        self._cache[layer].append(hidden_state)


    def _consolidate(self, layer: int) -> None|Tensor:
        """ Consolidate the list of hidden states for a layer into a single tensor """
        hs_list = self._cache.get(layer, [])

        if len(hs_list) == 0:
            return None

        if len(hs_list) == 1:
            # If there's only one hidden state, no need to consolidate
            return hs_list[0]

        hidden_states = torch.cat(hs_list, dim=1)  # Concatenate along seq_len dimension
        self._cache[layer] = [hidden_states]

        return hidden_states


    def get_consolidated(self) -> dict[int, Tensor]:
        """ Get a dictionary of consolidated hidden states for all layers """
        consolidated_hs = {}
        for layer in self._cache.keys():
            hidden_state = self._consolidate(layer)
            if hidden_state is not None:
                consolidated_hs[layer] = hidden_state
        return consolidated_hs


    def __getitem__(self, index: int) -> Tensor:
        """ Get the consolidated hidden state for a specific layer """
        hs = self._consolidate(index)
        if hs is None:
            raise KeyError(f"Layer {index} not found in TensorCache.")
        return hs


    def copy(self, clone_tensor: bool=False) -> 'TensorCache':
        """ Create a copy of the TensorCache. If clone_tensor is True, clone the tensors. """
        new_cache = TensorCache()
        for layer, hs_list in self._cache.items():
            if clone_tensor:
                new_hs_list = [hs.clone() for hs in hs_list]
            else:
                new_hs_list = [hs for hs in hs_list]
            new_cache._cache[layer] = new_hs_list
        return new_cache


    def __contains__(self, index: int) -> bool:
        """ Check if a layer index exists in the cache """
        return index in self._cache


def concate_tensor_caches(caches: list[TensorCache]) -> TensorCache:
    """ Concatenate multiple TensorCache instances along the batch dimension """
    if len(caches) == 0:
        raise ValueError("No TensorCache instances to concatenate.")

    new_cache = TensorCache()
    
    for cache in caches:
        for layer, hs_list in cache._cache.items():
            if layer not in new_cache._cache:
                new_cache._cache[layer] = []
            new_cache._cache[layer].extend(hs_list)

    return new_cache
