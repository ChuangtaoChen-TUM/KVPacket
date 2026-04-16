import torch
import pathlib
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import colorsys
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import matplotlib as mpl
import matplotlib.ticker as ticker


tokenizer_path: str|None = None
assert tokenizer_path is not None, "Please specify the tokenizer path"
tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
    tokenizer_path, use_fast=True, trust_remote_code=True
)

attn_score_path = "/mnt/storage/data2/FOLDER/cache"

SELECT_INDICES = list(range(10))
SELECT_LAYERS = slice(None, None)  # Last 5 layers
MAX_NUM_DOC = 2
MOVING_AVG_WINDOW = 1

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
COLOR_DICT = {
    "context": COLORS[2],
    "header": COLORS[1],
    "document": COLORS[0],
    "trailer": COLORS[3],
    "other": COLORS[4],
}

mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def resample_segment(values: list[float], target_len: int = 50) -> np.ndarray | None:
    if len(values) == 0:
        return None
    x_old = np.linspace(0, 1, len(values))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, values)


def bold_text(text: str) -> str:
    return r"$\bf{" + text + r"}$"


def build_aligned_sample(sample: list[list[float]], target_len: int = 50) -> np.ndarray:
    aligned = []

    for seg in sample:
        resampled = resample_segment(seg, target_len)
        if resampled is not None:
            aligned.append(resampled)

    if len(aligned) == 0:
        return np.array([])

    return np.concatenate(aligned)
    
def moving_average(tensor: torch.Tensor, window_size: int) -> torch.Tensor:
    # 1. Add batch and channel dimensions (Required for conv1d)
    # Shape: [Length] -> [1, 1, Length]
    x = tensor.view(1, 1, -1)
    
    # 2. Calculate padding needed for 'same' size
    # For a centered window, we pad half the window size on each side
    pad_left = window_size // 2
    pad_right = window_size // 2 if window_size % 2 != 0 else (window_size // 2) - 1
    
    # 3. Apply Replicate Padding (pads with the edge values)
    # F.pad expects (padding_left, padding_right)
    x_padded = F.pad(x, (pad_left, pad_right), mode='replicate')
    
    # 4. Define the averaging kernel
    kernel = torch.ones(1, 1, window_size, device=tensor.device) / window_size
    
    # 5. Convolve (no extra padding here since we did it manually)
    result = F.conv1d(x_padded, kernel)
    
    return result.view(-1)


def get_segment_type(seg_idx: int) -> str:
    if seg_idx == 0:
        return "context"
    else:
        mod = (seg_idx - 1) % 3
        if mod == 0:
            return "header"
        elif mod == 1:
            return "document"
        else:
            return "trailer"


def plot_multi_panel_grid(
    all_nr_data,
    all_kv_data,
    dataset_names=None,
    fig_name="attn_score_plot.svg",
    target_len=100
):
    num_folders = len(all_nr_data)
    # Set figsize to exactly (3 * num_folders, 6)
    fig, axes = plt.subplots(nrows=2, ncols=num_folders, figsize=(2.5 * num_folders, 4), sharey="col")

    if num_folders == 1:
        axes = np.array(axes).reshape(2, 1)

    for col in range(num_folders):
        datasets = [all_nr_data[col], all_kv_data[col]]
        row_titles = ["No Recompute", "KV Packet"]

        max_segments = max(len(s) for d in datasets for s in d)
        segment_widths = []
        for seg_idx in range(max_segments):
            lens = [len(sample[seg_idx]) for d in datasets for sample in d if seg_idx < len(sample)]
            segment_widths.append(np.sqrt(np.mean(lens)) if lens else 0)

        # --- Compute col-level scale across both rows ---
        col_max_val = 0.0
        all_segments_data = {}  # (row, seg_idx) -> resampled_matrix

        for row in range(2):
            curr_dataset = datasets[row]
            for seg_idx in range(max_segments):
                resampled_matrix = []
                for sample in curr_dataset:
                    if seg_idx < len(sample) and len(sample[seg_idx]) > 0:
                        resampled = resample_segment(sample[seg_idx], target_len)
                        if resampled is not None:
                            resampled_matrix.append(resampled)
                if resampled_matrix:
                    m = np.array(resampled_matrix)
                    all_segments_data[(row, seg_idx)] = m
                    col_max_val = max(col_max_val, np.max(np.abs(m)))

        mag = int(np.floor(np.log10(col_max_val))) if col_max_val > 0 else 0
        scale = 10 ** mag

        for row in range(2):
            ax = axes[row, col]
            curr_dataset = datasets[row]
            current_x_start = 0.0

            for seg_idx in range(max_segments):
                width = segment_widths[seg_idx]
                if width <= 0:
                    continue

                seg_type = get_segment_type(seg_idx)
                color = COLOR_DICT.get(seg_type, COLORS[4])

                has_data = (row, seg_idx) in all_segments_data
                if has_data:
                    ax.axvspan(current_x_start, current_x_start + width,
                            color=color, alpha=0.1, linewidth=0, zorder=1)

                if has_data:
                    m = all_segments_data[(row, seg_idx)]
                    mean_curve = np.mean(m, axis=0) / scale
                    std_curve = np.std(m, axis=0) / scale
                    x_vals = np.linspace(current_x_start, current_x_start + width, target_len)
                    ax.plot(x_vals, mean_curve, color=color, lw=1.5, zorder=3)
                    ax.fill_between(x_vals, mean_curve - std_curve, mean_curve + std_curve,
                                    color=color, alpha=0.4, lw=0, zorder=2)

                x_end = current_x_start + width
                if seg_idx < max_segments - 1:
                    ax.axvline(x=x_end, color='black', ls='--', alpha=0.1, lw=0.7)
                current_x_start = x_end

            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            # Only place the exponent label on the left-most column's axes
            if col == 0:
                ax.text(0, 1.01, f"$\\times10^{{{mag}}}$",
                        transform=ax.transAxes, fontsize=9, va='bottom', ha='left')

            if row == 0:
                ax.set_title(dataset_names[col] if dataset_names else f"Set {col}")
            if row == 1:
                ax.set_xlabel("Token Index")
            if col == 0:
                ax.set_ylabel(f"{bold_text(row_titles[row])}\nAttention Score")
            ax.set_xticks([])

    # --- Global Legend ---
    legend_elements = [
        Line2D([0], [0], color=COLOR_DICT["context"], lw=2, label="Context"),
        Line2D([0], [0], color=COLOR_DICT["header"], lw=2, label="Header"),
        Line2D([0], [0], color=COLOR_DICT["document"], lw=2, label="Document"),
        Line2D([0], [0], color=COLOR_DICT["trailer"], lw=2, label="Trailer"),
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
               ncol=4, frameon=True, fancybox=True)

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    for ax_row in axes:
        for ax in ax_row:
            ax.yaxis.offsetText.set_visible(False)
            ax.yaxis.offsetText.set_text('')
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()



def process_folder(attn_score_folder: str | pathlib.Path) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    no_recompute_file = pathlib.Path(attn_score_folder) / "no_recompute_attn_scores.pt"
    kv_packet_file = pathlib.Path(attn_score_folder) / "kv_packet_attn_scores.pt"

    nr_data: list[list[list[float]]] = []
    kv_data: list[list[list[float]]] = []

    base_dict = torch.load(no_recompute_file)
    kv_dict = torch.load(kv_packet_file)

    assert base_dict["header_len"] == 0, "Expected header_len to be 0 for no recompute"
    assert base_dict["trailer_len"] == 0, "Expected trailer_len to be 0 for no recompute"

    for select_index in SELECT_INDICES:

        # =========================
        # No Recompute
        # =========================
        current_segments: list[list[float]] = []

        entry, attn_scores = base_dict["attention_scores"][select_index]
        attn_scores = attn_scores.float().softmax(dim=-1).mean(0)

        documents = entry["documents"]
        doc_ids = tokenizer(documents, add_special_tokens=False)["input_ids"]
        assert isinstance(doc_ids, list)
        doc_lens = [len(ids) for ids in doc_ids]

        ctx_ids = tokenizer(entry["preamble"], add_special_tokens=False)["input_ids"]
        assert isinstance(ctx_ids, list)
        ctx_len = len(ctx_ids)

        current_segments.append(attn_scores[:ctx_len].tolist())
        current_pos = ctx_len

        for i, doc_len in enumerate(doc_lens):
            current_segments.append([])  # header (empty)
            current_segments.append(
                moving_average(
                    attn_scores[current_pos: current_pos + doc_len],
                    MOVING_AVG_WINDOW
                ).tolist()
            )
            current_pos += doc_len
            current_segments.append([])  # trailer (empty)

            if i + 1 >= MAX_NUM_DOC:
                break

        nr_data.append(current_segments)

        # =========================
        # KV Packet
        # =========================
        current_segments = []

        entry, attn_scores = kv_dict["attention_scores"][select_index]
        attn_scores = attn_scores.float().softmax(dim=-1)
        attn_scores = attn_scores[SELECT_LAYERS].mean(0)

        documents = entry["documents"]
        doc_ids = tokenizer(documents, add_special_tokens=False)["input_ids"]
        assert isinstance(doc_ids, list)
        doc_lens = [len(ids) for ids in doc_ids]

        ctx_ids = tokenizer(entry["preamble"], add_special_tokens=False)["input_ids"]
        assert isinstance(ctx_ids, list)
        ctx_len = len(ctx_ids)

        current_segments.append(attn_scores[:ctx_len].tolist())
        current_pos = ctx_len

        header_len = kv_dict["header_len"]
        trailer_len = kv_dict["trailer_len"]

        for i, doc_len in enumerate(doc_lens):
            current_segments.append(
                attn_scores[current_pos: current_pos + header_len].tolist()
            )
            current_pos += header_len

            current_segments.append(
                moving_average(
                    attn_scores[current_pos: current_pos + doc_len],
                    MOVING_AVG_WINDOW
                ).tolist()
            )
            current_pos += doc_len

            current_segments.append(
                attn_scores[current_pos: current_pos + trailer_len].tolist()
            )
            current_pos += trailer_len

            if i + 1 >= MAX_NUM_DOC:
                break

        kv_data.append(current_segments)

    return nr_data, kv_data



def adjust_color_lightness(base_color_hex: str, factor: float) -> tuple[float|int, float|int, float|int]|str:
    """
    Adjusts the lightness of a color.
    factor: 0.0 (dark) to 1.0 (light).
    """
    try:
        c = mcolors.to_rgb(base_color_hex)
        # Convert RGB to HLS (Hue, Lightness, Saturation)
        h, _, s = colorsys.rgb_to_hls(*c)
        
        # We compress the factor into a usable lightness range (e.g., 0.3 to 0.8)
        # so lines are neither pitch black nor invisible white.
        new_l = 0.3 + (factor * 0.5) 
        
        return colorsys.hls_to_rgb(h, new_l, s)
    except Exception as e:
        print(f"Color error: {e}")
        return base_color_hex


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw attention score plot")
    parser.add_argument(
        "attn_score_folders",
        nargs="+",
        type=str,
        help="Path to the attention score folder of KV Packet"
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        type=str,
        default=None,
        help="Optional custom names for the datasets (in order). If not provided, defaults to folder names."
    )

    args = parser.parse_args()
    attn_score_folders = args.attn_score_folders
    dataset_names = args.dataset_names

    if dataset_names is not None and len(dataset_names) != len(attn_score_folders):
        raise ValueError("Number of dataset names must match number of attention score folders")


    all_nr_data: list[list[list[list[float]]]] = []
    all_kv_data: list[list[list[list[float]]]] = []

    for folder in attn_score_folders:
        nr_data, kv_data = process_folder(folder)
        all_nr_data.append(nr_data)
        all_kv_data.append(kv_data)

    plot_multi_panel_grid(
        all_nr_data,
        all_kv_data,
        dataset_names=dataset_names,
        fig_name="attn_score_plot.svg"
    )