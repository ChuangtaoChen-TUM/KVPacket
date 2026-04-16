# python draw_main_results.py ./eval_config/llama_3_1_8b/niah ./eval_config/llama_3_1_8b/hotpot_qa/ ./eval_config/llama_3_1_8b/musique/ ./eval_config/llama_3_1_8b/biography/ --dataset-names Needle-in-a-Haystack HotpotQA MusiQue Biography --output-file llama_results.svg
# python draw_main_results.py ./eval_config/qwen_3_4b/niah ./eval_config/qwen_3_4b/hotpot_qa/ ./eval_config/qwen_3_4b/musique/ ./eval_config/qwen_3_4b/biography/ --dataset-names Needle-in-a-Haystack HotpotQA MusiQue Biography --output-file qwen_3_4b_results.svg --output-file qwen_3_4b_results.svg
import argparse
import os
import json
import re
from pathlib import Path
from typing import Any
from matplotlib import pyplot as plt


def gather_results(result_dict: dict, key_lists: list[list[str]]) -> dict[str, float]:
    """
    Gathers results from a dictionary based on a list of keys.

    Args:
        result_dict (dict): A dictionary containing results.
        key_lists (list[list[str]]): A list of result key (list for nested keys).

    Returns:
        list: A list of results corresponding to the provided keys.
    """
    gathered_results: dict[str, float] = {}

    for key_list in key_lists:
        current_dict = result_dict
        # print(current_dict)
        for key in key_list:
            current_dict = current_dict[key]
    
        assert isinstance(current_dict, (int, float))
        gathered_results[".".join(key_list)] = current_dict
    return gathered_results


def gather_files(file_or_folder: str) -> list[str]:
    if os.path.isfile(file_or_folder):
        return [file_or_folder]
    elif os.path.isdir(file_or_folder):
        return [
            os.path.join(file_or_folder, f)
            for f in os.listdir(file_or_folder)
            if os.path.isfile(os.path.join(file_or_folder, f))
        ]
    else:
        raise ValueError(f"{file_or_folder} is neither a file nor a directory.")


def load_json_file(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def format_str(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def convert_file_name(file_name: str) -> tuple[str, str]:
    file_name_map: dict[str, str] = {
        "no_recompute": "No Recompute",
        "full_recompute": "Full Recompute",
        "no_cache": "No Cache",
        "sink": "Sink",
        "kv_packet": "KV Packet",
    }
    if (converted_name := file_name_map.get(file_name)) is not None:
        return converted_name, ""

    cache_blend_patter = re.compile(r"cache_blend_(\d+\.?\d*)")
    match = cache_blend_patter.match(file_name)
    if match:
        blend_value = match.group(1)
        blend_value = blend_value.rstrip('.')
        return "Cache Blend", f"{blend_value}%"

    epic_pattern = re.compile(r"epic_(\d+)")
    match = epic_pattern.match(file_name)
    if match:
        epic_value = match.group(1)
        return "EPIC", f"{epic_value}"
    
    kv_packet_pattern = re.compile(r"kv_packet_(\d+)_(\d+)")
    match = kv_packet_pattern.match(file_name)
    if match:
        header_len = match.group(1)
        trailer_len = match.group(2)
        return "KV Packet", f"{header_len}, {trailer_len}"

    random_pattern = re.compile(r"rand_recompute_(\d+\.?\d*)")
    match = random_pattern.match(file_name)
    if match:
        recompute_ratio = match.group(1)
        recompute_ratio = recompute_ratio.rstrip('.')
        return "Random Recompute", f"{recompute_ratio}%"
    
    a3_pattern = re.compile(r"a3_(\d+)")
    match = a3_pattern.match(file_name)
    if match:
        a3_value = match.group(1)
        return "A3", f"{a3_value}%"
    
    sam_kv_pattern = re.compile(r"sam_kv_(small|large)")
    match = sam_kv_pattern.match(file_name)
    if match:
        model_size = match.group(1)
        return "SAM-KV", model_size.capitalize()

    raise ValueError(f"Unrecognized file name pattern: {file_name}")

def plot_results(result_by_dataset: dict[str, dict[tuple[str, str], dict[str, float]]], output_file: str) -> None:
    style_config = {
        "No Recompute": {"color": "gray", "marker": "X", "s": 100, "zorder": 2},
        "Full Recompute": {"color": "black", "marker": "s", "s": 100, "zorder": 2},
        "No Cache": {"color": "red", "marker": "d", "s": 100, "zorder": 2},
        "A3": {"color": "brown", "marker": "v", "s": 100, "zorder": 2},
        "Cache Blend": {"color": "blue", "marker": "o", "s": 80, "zorder": 2},
        "EPIC": {"color": "green", "marker": "^", "s": 80, "zorder": 2},
        "KV Packet": {"color": "purple", "marker": "*", "s": 350, "zorder": 10},
        "Random Recompute": {"color": "orange", "marker": ".", "s": 80, "zorder": 2},
        "SAM-KV": {"color": "gold", "marker": "P", "s": 80, "zorder": 2},
        "default": {"color": "orange", "marker": ".", "s": 50, "zorder": 1}
    }

    num_datasets = len(result_by_dataset)
    fig, axes = plt.subplots(2, num_datasets, figsize=(3 * num_datasets, 6),
                            squeeze=False)

    plot_defs = [
        ("flops", "f1", "FLOPs", "F1 Score"),
        ("ttft", "f1", "TTFT", "F1 Score"),
    ]

    for col_idx, (dataset_name, dataset_data) in enumerate(result_by_dataset.items()):
        for row_idx, (x_key, y_key, x_label, y_label) in enumerate(plot_defs):
            ax = axes[row_idx, col_idx]

            # --- Group points by category for dashed connecting lines ---
            category_points: dict[str, list[tuple[float, float]]] = {}
            for (category, extra_info), metrics in dataset_data.items():
                if category == "SAM-KV" and x_key == "ttft":
                    continue
                if x_key in metrics and y_key in metrics:
                    category_points.setdefault(category, []).append(
                        (metrics[x_key], metrics[y_key])
                    )

            # Draw dashed lines connecting points of the same series
            for category, points in category_points.items():
                if len(points) < 2:
                    continue
                style = style_config.get(category, style_config["default"])
                points_sorted = sorted(points, key=lambda p: p[0])
                xs = [p[0] for p in points_sorted]
                ys = [p[1] for p in points_sorted]
                ax.plot(xs, ys, color=style["color"], linestyle="--", linewidth=1.6,
                        alpha=0.5, zorder=style["zorder"] - 1)

            # --- Scatter points ---
            for (category, extra_info), metrics in dataset_data.items():
                style = style_config.get(category, style_config["default"])

                if category == "SAM-KV" and x_key == "ttft":
                    continue

                if x_key in metrics and y_key in metrics:
                    x_val = metrics[x_key]
                    y_val = metrics[y_key]
                    ax.scatter(x_val, y_val, label=category, alpha=0.7, **style)

            # --- Axes labels & title ---
            if row_idx == 0:
                ax.set_title(dataset_name, fontsize=14, pad=15)

            if col_idx == 0:
                ax.set_ylabel(y_label, fontsize=12)

            ax.set_xlabel(x_label, fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.4)

            ax.annotate("Better", xy=(0.05, 0.92), xytext=(0.18, 0.8),
                        xycoords='axes fraction', textcoords='axes fraction',
                        fontsize=7, fontweight='bold', color='green', alpha=0.7,
                        arrowprops=dict(facecolor='green', edgecolor='none', shrink=0.1,
                                        width=5, headwidth=10, headlength=10, alpha=0.7))

    # --- Standardized Legend ---
    # KV Packet is first, then the rest
    order_map = ["KV Packet", "Full Recompute", "No Recompute", "No Cache", "Random Recompute",
                 "A3", "SAM-KV", "Cache Blend", "EPIC"]

    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_handles = [by_label[k] for k in order_map if k in by_label]
    ordered_labels = [rf"$\mathbf{{{k}}}$" if k == "KV Packet" else k for k in order_map if k in by_label]

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, left=0.08)

    fig.legend(
        ordered_handles, ordered_labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=len(ordered_labels) // 2 + 1 if len(ordered_labels) > 5 else len(ordered_labels),
        frameon=True, edgecolor="black"
    )
    plt.savefig(output_file, dpi=300)

def plot_results_old(result_by_dataset: dict[str, dict[tuple[str, str], dict[str, float]]], output_file: str) -> None:
    style_config = {
        "No Recompute": {"color": "gray", "marker": "X", "s": 100, "zorder": 2},
        "Full Recompute": {"color": "black", "marker": "s", "s": 100, "zorder": 2},
        "No Cache": {"color": "red", "marker": "d", "s": 100, "zorder": 2},
        "A3": {"color": "brown", "marker": "v", "s": 100, "zorder": 2},
        "Cache Blend": {"color": "blue", "marker": "o", "s": 80, "zorder": 2},
        "EPIC": {"color": "green", "marker": "^", "s": 80, "zorder": 2},
        "KV Packet": {"color": "purple", "marker": "*", "s": 150, "zorder": 10},
        "Random Recompute": {"color": "orange", "marker": ".", "s": 80, "zorder": 2},
        "SAM-KV": {"color": "pink", "marker": "P", "s": 80, "zorder": 2},
        "default": {"color": "orange", "marker": ".", "s": 50, "zorder": 1}
    }

    num_datasets = len(result_by_dataset)
    # sharey='row' ensures all F1 scores in a row use the same scale
    fig, axes = plt.subplots(2, num_datasets, figsize=(3 * num_datasets, 6), 
                            squeeze=False)
    
    plot_defs = [
        ("flops", "f1", "FLOPs", "F1 Score"),
        ("ttft", "f1", "TTFT", "F1 Score"),
    ]

    for col_idx, (dataset_name, dataset_data) in enumerate(result_by_dataset.items()):
        for row_idx, (x_key, y_key, x_label, y_label) in enumerate(plot_defs):
            ax = axes[row_idx, col_idx]
            
            for (category, extra_info), metrics in dataset_data.items():
                style = style_config.get(category, style_config["default"])
                
                # Skip SAM-KV for TTFT rows since they are too large
                if category == "SAM-KV" and x_key == "ttft":
                    continue

                if x_key in metrics and y_key in metrics:
                    x_val = metrics[x_key]
                    y_val = metrics[y_key]
                    ax.scatter(x_val, y_val, label=category, alpha=0.7, **style)
                    
                    if extra_info:
                        ax.annotate(extra_info, (x_val, y_val), xytext=(4, 4), 
                                    textcoords='offset points', fontsize=7, alpha=0.7)

            # --- Merged Label Logic ---
            
            # 1. Dataset Names only on the Top Row
            if row_idx == 0:
                ax.set_title(dataset_name, fontsize=14, pad=15)

            # 2. Y-Axis Label (F1 Score) only on the Leftmost Column
            if col_idx == 0:
                ax.set_ylabel(y_label, fontsize=12)
            
            # 3. X-Axis Labels (FLOPs / TTFT) on every plot for clarity, 
            #    or only on bottom if you prefer. Here we keep them for readability.
            ax.set_xlabel(x_label, fontsize=11)
            
            ax.grid(True, linestyle='--', alpha=0.4)

            # Better Arrow - Top Left
            ax.annotate("Better", xy=(0.05, 0.92), xytext=(0.18, 0.8),
                        xycoords='axes fraction', textcoords='axes fraction',
                        fontsize=7, fontweight='bold', color='green', alpha=0.7,
                        arrowprops=dict(facecolor='green', edgecolor='none', shrink=0.1, width=5, headwidth=10, headlength=10, alpha=0.7))

    # --- Row Titles (Merged Labels) ---
    # We add text to the left of the rows to describe the comparison
    fig.text(0.02, 0.75, 'F1 Score vs FLOPs', va='center', rotation='vertical', fontsize=12, fontweight='bold', color='gray')
    fig.text(0.02, 0.35, 'F1 Score vs TTFT', va='center', rotation='vertical', fontsize=12, fontweight='bold', color='gray')

    # --- Standardized Legend ---
    order_map = ["Full Recompute", "No Recompute", "No Cache", "Random Recompute", 
                 "A3", "SAM-KV", "Cache Blend", "EPIC", "KV Packet"]
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_handles = [by_label[k] for k in order_map if k in by_label]
    ordered_labels = [rf"$\mathbf{{{k}}}$" if k == "KV Packet" else k for k in order_map if k in by_label]

    plt.tight_layout()
    # Adjust to make room for the legend and the side row-labels
    plt.subplots_adjust(bottom=0.22, left=0.08) 
    
    fig.legend(
        ordered_handles, ordered_labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=len(ordered_labels) // 2 + 1 if len(ordered_labels) > 5 else len(ordered_labels),
        frameon=True, edgecolor="black"
    )
    plt.savefig(output_file, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather results from a dictionary based on provided keys.")
    parser.add_argument("result_folders", type=str, nargs='+', help="Paths to result folder files. A eval_results sub_folder should be found in each path.")
    parser.add_argument("--dataset-names", type=str, nargs='+', default=None, help="Optional list of dataset names corresponding to each result folder for labeling purposes.")
    parser.add_argument("--output-file", type=str, default="gathered_results_plot.svg", help="Output file name for the plot.")
    keys = [
        "precision",
        "recall",
        "f1",
        "ttft",
        "flops"
    ]
    args = parser.parse_args()

    if args.dataset_names and len(args.dataset_names) != len(args.result_folders):
        raise ValueError("The number of dataset names must match the number of result folders.")

    if args.dataset_names is not None:
        dataset_names = args.dataset_names
    else:
        dataset_names = [Path(path).name for path in args.result_folders]

    result_by_dataset: dict[str, dict[tuple[str, str], dict[str, float]]] = {}
    for path, dataset_name in zip(args.result_folders, dataset_names):
        all_results: dict[tuple[str, str], dict[str, float]] = {}
        files = gather_files(os.path.join(path, "eval_results"))
        print(f"Processing dataset '{dataset_name}' with {len(files)} result files.")
        print(f"Found files: {files}")
        for each_file in files:
            print(f"Processing file: {each_file}")
            result_data = load_json_file(each_file).get("result", {})
            assert result_data, f"No 'result' key found in {each_file}"
            key_lists = [[key] for key in keys]
            gathered_data = gather_results(result_data, key_lists)

            try:
                converted_name, extra_info = convert_file_name(
                    os.path.splitext(os.path.basename(each_file))[0].removesuffix("_result")
                )
            except ValueError as e:
                print(f"Skipping file {each_file}: {e}")
                continue
            all_results[(converted_name, extra_info)] = gathered_data
        
        result_by_dataset[dataset_name] = all_results

    print("Gathered Results by Dataset:")
    for dataset_name, results in result_by_dataset.items():
        print(f"\nDataset: {dataset_name}")
        for (category, extra_info), metrics in results.items():
            print(f"  {category} ({extra_info}): {metrics}")
    plot_results(result_by_dataset, output_file=args.output_file)