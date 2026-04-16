""" Utility functions for configuration management, including broadcasting dictionaries and loading config files. """
import copy
import json
import os
import re
from typing import Any, is_typeddict, get_type_hints, get_origin


def has_cycle(obj, path=None):
    """
    Detects if the object (dict or list) contains a reference loop (cycle).
    
    Args:
        obj: The object to check.
        path (set): Internal set of visited object IDs in the current recursion stack.
        
    Returns:
        bool: True if a cycle is detected, False otherwise.
    """
    if path is None:
        path = set()
    
    obj_id = id(obj)
    # If the current object ID is already in the current traversal path, we found a cycle.
    if obj_id in path:
        return True
    
    path.add(obj_id)

    # Check children
    if isinstance(obj, (dict, list)):
        objs = obj.values() if isinstance(obj, dict) else obj
    else:
        objs = []

    for v in objs:
        if isinstance(v, (dict, list)) and has_cycle(v, path):
            return True
    
    # Backtrack: remove from path as we go back up the recursion stack
    path.remove(obj_id)
    return False


def broadcast_dict(base_dict: dict, target_dict: dict, overwrite: bool = False) -> dict:
    """
    Recursively updates target_dict with values from base_dict.
    
    This function strictly forbids cyclic references in base_dict to ensure 
    clean data merging.
    
    Args:
        base_dict (dict): The source dictionary.
        target_dict (dict): The destination dictionary.
        overwrite (bool): If True, existing keys in target_dict will be overwritten 
                          by values from base_dict (unless both are dictionaries, 
                          in which case they merge recursively).
                          
    Returns:
        dict: The updated target_dict.
        
    Raises:
        ValueError: If base_dict contains a cyclic dependency.
    """
    
    # Step 1: Validate base_dict is acyclic
    # We only check the top-level call. Recursive calls will use _broadcast_recursive
    if has_cycle(base_dict):
        raise ValueError("Cyclic dependency detected in base_dict. Cannot broadcast safely.")
        
    return _broadcast_recursive(base_dict, target_dict, overwrite)


def _broadcast_recursive(base_dict: dict, target_dict: dict, overwrite: bool) -> dict:
    """
    Internal helper to perform the broadcast without re-checking for cycles.
    """
    for key, value in base_dict.items():
        # Case 1: Key does not exist in target -> Add it
        if key not in target_dict:
            target_dict[key] = copy.deepcopy(value)
        
        # Case 2: Key exists in both
        else:
            target_value = target_dict[key]
            
            # Sub-case 2a: Both values are dictionaries -> Recurse (Deep Merge)
            if isinstance(value, dict) and isinstance(target_value, dict):
                _broadcast_recursive(value, target_value, overwrite)
            
            # Sub-case 2b: Conflict (Leaf nodes or type mismatch)
            else:
                if overwrite:
                    target_dict[key] = copy.deepcopy(value)
                # If overwrite is False, we do nothing

    return target_dict


def gather_config_files(
    file_or_path: str,
    pattern: str | None = None,
    skip_pattern: str | None = r"_default\.json"
) -> list[str]:
    """
    Gathers configuration files based on regex patterns.
    
    :param file_or_path: Path to a specific file or a directory to search.
    :param pattern: Regex string to include files. If None, includes all non-skipped files.
    :param skip_pattern: Regex string to exclude files. Defaults to excluding files ending in _default.json.
    """
    config_files: list[str] = []

    def add_file(path: str) -> None:
        """
        Validates the file against patterns and adds it to the list if valid.
        """
        filename = os.path.basename(path)

        # 1. Check exclusion (Skip Pattern)
        # We check this first to immediately disqualify unwanted files.
        if skip_pattern and re.search(skip_pattern, filename):
            return

        # 2. Check inclusion (Pattern)
        # If pattern is None, we assume everything is valid (unless skipped).
        if pattern is None or re.search(pattern, filename):
            config_files.append(path)

    # Main logic
    if os.path.isfile(file_or_path):
        add_file(file_or_path)
    elif os.path.isdir(file_or_path):
        for fname in os.listdir(file_or_path):
            full_path = os.path.join(file_or_path, fname)
            # Ensure we only process files, mimicking the original logic's intent
            if os.path.isfile(full_path):
                add_file(full_path)
    else:
        raise ValueError(f"Path {file_or_path} is neither a file nor a directory.")

    return config_files



def load_config_file(
    file_name: str,
    default_config_file: str|None = None
) -> dict[str, Any]:
    """ Loads a JSON config file and optionally merges it with a default config file 
    
    Args:
        file_name (str): The path to the JSON config file to load.
        default_config_file (str|None): The filename of the default config to merge with, located in the same directory as file_name.
    """
    file_folder = os.path.dirname(file_name)
    if default_config_file is None:
        default_config = {}
    else:
        default_config_path = os.path.join(file_folder, default_config_file)
        if os.path.isfile(default_config_path):
            with open(default_config_path, 'r') as f:
                default_config = json.load(f)
        else:
            default_config = {}
    
    with open(file_name, 'r') as f:
        config = json.load(f)
    
    # Broadcast default config into loaded config
    broadcast_dict(default_config, config, overwrite=False)
    return config


def load_config_files(
    file_or_path: str,
    pattern: str | None = None,
    skip_pattern: str | None = r"_default\.json",
    default_config_file: str | None = "_default.json"
) -> list[dict[str, Any]]:
    """ Loads multiple config files from a directory or a single file, applying default configs as needed. 
    Args:
        file_or_path (str): Path to a single config file or a directory containing config files.
        pattern (str|None): Regex pattern to include files. If None, includes all non-skipped files.
        skip_pattern (str|None): Regex pattern to exclude files. Defaults to excluding files ending in _default.json.
        default_config_file (str|None): The filename of the default config to merge with, located in the same directory as the config files. Defaults to "_default.json".
    """
    config_files = gather_config_files(file_or_path, pattern, skip_pattern)
    configs: list[dict[str, Any]] = []
    for cfg_file in config_files:
        cfg = load_config_file(cfg_file, default_config_file)
        configs.append(cfg)
    return configs
