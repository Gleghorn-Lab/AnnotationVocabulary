import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datasets import load_dataset


def find_sequence_column(dataset):
    possible_names = ['seq', 'seqs', 'sequence']
    for col in dataset.column_names:
        if col.lower().startswith('seq') or col.lower() == 'sequence':
            return col
    raise ValueError("Could not find a column name starting with 'seq' or named 'sequence'")


def load_dataset_splits(path, together=False):
    dataset = load_dataset(path)
    if together:
        all_seqs = set()
        for split in dataset.keys():
            seq_column = find_sequence_column(dataset[split])
            all_seqs.update(set(dataset[split][seq_column]))
        return {"combined": all_seqs}
    else:
        split_sets = {}
        for split in dataset.keys():
            seq_column = find_sequence_column(dataset[split])
            split_sets[split] = set(dataset[split][seq_column])
        return split_sets


def create_overlap_matrix(dataset_splits, use_percentage=False):
    all_splits = [(path, split) for path, splits in dataset_splits.items() for split in splits]
    n = len(all_splits)
    overlap_matrix = np.zeros((n, n), dtype=float if use_percentage else int)
    
    for i in tqdm(range(n), desc='Finding intersection'):
        for j in range(n):
            path_i, split_i = all_splits[i]
            path_j, split_j = all_splits[j]
            intersection = len(dataset_splits[path_i][split_i].intersection(dataset_splits[path_j][split_j]))
            if use_percentage:
                smaller_set_size = min(len(dataset_splits[path_i][split_i]), len(dataset_splits[path_j][split_j]))
                overlap_matrix[i, j] = (intersection / smaller_set_size) * 100
            else:
                overlap_matrix[i, j] = intersection
    
    return overlap_matrix, all_splits


def plot_overlap_matrix(overlap_matrix, split_names, use_percentage=False):
    # Create a custom colormap for absolute numbers
    colors_abs = ['#FFFFFF', '#EBF5FB', '#D6EAF8', '#AED6F1', '#85C1E9', '#5DADE2', '#3498DB', '#2E86C1', '#2874A6', '#21618C', '#1B4F72']
    cmap_abs = LinearSegmentedColormap.from_list('custom_cmap', colors_abs, N=len(colors_abs))
    
    # Use a standard colormap for percentages
    cmap_percent = 'coolwarm'
    
    # Get dimensions
    dim = len(overlap_matrix)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(dim, dim))
    
    # Set color scaling and format
    if use_percentage:
        norm = plt.Normalize(vmin=0, vmax=100)
        fmt = '.1f'
        cmap = cmap_percent
        cbar_label = 'Percentage of overlap'
    else:
        norm = LogNorm(vmin=1, vmax=overlap_matrix.max())
        fmt = 'd'
        cmap = cmap_abs
        cbar_label = 'Number of overlapping sequences'
    
    # Plot heatmap
    sns.heatmap(overlap_matrix, annot=True, fmt=fmt, 
                xticklabels=split_names, yticklabels=split_names, 
                cmap=cmap, norm=norm,
                cbar=False,  # We'll add the colorbar manually
                ax=ax, square=True)
    
    # Create a smaller colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = plt.colorbar(ax.collections[0], cax=cax)
    cbar.set_label(cbar_label, fontsize=dim+5, labelpad=15)
    cbar.ax.tick_params(labelsize=dim)  # Corrected line for colorbar tick label font size

    # Rotate x-axis labels and set font sizes
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=dim)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=dim)
    
    # Set title
    ax.set_title('Sequence Overlap Between Dataset Splits', fontsize=dim+5, pad=20)

    plt.tight_layout()

    # Save the figure
    plt.savefig('dataset_splits_overlap_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()



def main(together=False, use_percentage=False):
    multilabel = [
        {'path': 'lhallee/EC_reg', 'nickname': 'EC'},
        {'path': 'lhallee/CC_reg', 'nickname': 'CC'},
        {'path': 'lhallee/MF_reg', 'nickname': 'MF'},
        {'path': 'lhallee/BP_reg', 'nickname': 'BP'},
    ]

    classification = [
        {'path': 'lhallee/dl_binary_reg', 'nickname': 'DL2'},
        {'path': 'lhallee/dl_ten_reg', 'nickname': 'DL10'},
        {'path': 'lhallee/MetalIonBinding_reg', 'nickname': 'MB'},
    ]

    ss = [
        {'path': 'lhallee/ssq3', 'nickname': 'SS3'},
        {'path': 'lhallee/ssq8', 'nickname': 'SS8'},
    ]

    ours = [
        {'path': 'GleghornLab/EXP_annotations', 'nickname': 'EXP'},
        {'path': 'GleghornLab/RED_annotations', 'nickname': 'RED'},
        {'path': 'lhallee/annotations_uniref90_all', 'nickname': 'RED_all'},
    ]

    clean = [
    {'path': 'lhallee/CLEAN', 'nickname': 'CLEAN'}
    ]

    other = [
        {'path': 'lhallee/pinui_human_set', 'nickname': 'HPPI'},
        {'path': 'lhallee/pinui_yeast_set', 'nickname': 'YPPI'},
        {'path': 'lhallee/SwissProt', 'nickname': 'Swiss'}
    ]

    datasets_info = multilabel + classification + ss + ours + clean + other

    # Load datasets and their splits
    dataset_splits = {}
    for info in datasets_info:
        try:
            dataset_splits[info['path']] = load_dataset_splits(info['path'], together=together)
        except ValueError as e:
            print(f"Error loading {info['path']}: {str(e)}")
            continue

    # Create overlap matrix
    overlap_matrix, all_splits = create_overlap_matrix(dataset_splits, use_percentage)

    if together:
        split_names = [info['nickname'] for info in datasets_info if info['path'] in dataset_splits]
    else:
        split_names = [f"{info['nickname']}_{split}" for info in datasets_info if info['path'] in dataset_splits for split in dataset_splits[info['path']].keys()]

    # Calculate total unique sequences
    all_sequences = set()
    for splits in dataset_splits.values():
        for seq_set in splits.values():
            all_sequences.update(seq_set)
    total_sequences = len(all_sequences)

    plot_overlap_matrix(overlap_matrix, split_names, use_percentage)

    print(f"Overlap matrix has been saved as 'dataset_splits_overlap_matrix.png'")
    print(f"Total unique sequences across all datasets and splits: {total_sequences}")


if __name__ == "__main__":
    main(together=True, use_percentage=True)  # Set use_percentage to True to use percentages
