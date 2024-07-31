import numpy as np
import umap
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List
import seaborn as sns
from functools import partial

DEFAULT_DATA_PATHS = [
    'GleghornLab/EC_reg', 'GleghornLab/CC_reg',
    'GleghornLab/MF_reg', 'GleghornLab/BP_reg',
    'GleghornLab/MB_reg', 'GleghornLab/DL2_reg',
    'GleghornLab/DL10_reg'
]

MULTILABEL_DATASETS = ['EC_reg', 'CC_reg', 'MF_reg', 'BP_reg']
BINARY_DATASETS = ['MB_reg', 'DL2_reg']

def process_column(example, col_name):
    if isinstance(example[col_name], list) and len(example[col_name]) > 0 and isinstance(example[col_name][0], float):
        example[col_name] = [int(x) for x in example[col_name]]
    return example

def create_data_arrays(embed_dict: Dict[str, np.ndarray], sequence_labels: Dict[str, List[int]], is_multilabel: bool):
    X = np.concatenate(list(embed_dict.values()))
    if is_multilabel:
        y = np.array([sequence_labels[seq] for seq in embed_dict.keys()])
    else:
        y = np.array([sequence_labels[seq] for seq in embed_dict.keys()])
    return X, y

def plot_2d_scatter(X_pca, X_tsne, X_umap, y, title, save_name, is_multilabel):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))
    fig.suptitle(title, fontsize=16)

    if is_multilabel:
        # Create a color map based on the sum of positive labels
        y_sum = np.sum(y, axis=1)
        norm = plt.Normalize(y_sum.min(), y_sum.max())
        cmap = plt.cm.get_cmap('viridis')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
    else:
        unique_labels = np.unique(y)
        color_palette = sns.color_palette("husl", n_colors=len(unique_labels))
        color_dict = dict(zip(unique_labels, color_palette))

    for ax, X_2d, method in zip([ax1, ax2, ax3], [X_pca, X_tsne, X_umap], ['PCA', 't-SNE', 'UMAP']):
        if is_multilabel:
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_sum, cmap=cmap, norm=norm)
        else:
            for label in unique_labels:
                mask = y == label
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color_dict[label]], label=str(label))

        ax.set_title(f'{method} Visualization')
        ax.set_xlabel('First component')
        ax.set_ylabel('Second component')

    if is_multilabel:
        fig.colorbar(sm, ax=ax3, label='Number of Positive Labels')
    else:
        handles, labels = ax3.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='center right')

    plt.tight_layout()
    plt.savefig(title.replace(' ', '_') + '_' + save_name + '.png', dpi=300, bbox_inches='tight')
    plt.close()

def main(sequence_embeddings: Dict[str, np.ndarray], sequence_labels: Dict[str, List[int]], split: str, data_type: str):
    is_multilabel = any(dataset in data_type for dataset in MULTILABEL_DATASETS)
    is_binary = any(dataset in data_type for dataset in BINARY_DATASETS)

    X, y = create_data_arrays(sequence_embeddings, sequence_labels, is_multilabel)

    if is_binary:
        y = y.astype(int)

    save_name = split + '_' + data_type

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # UMAP
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X)

    plot_2d_scatter(X_pca, X_tsne, X_umap, y, f'{data_type} Visualization', save_name, is_multilabel)

if __name__ == '__main__':
    embedding_dataset = load_dataset('lhallee/plm_embeddings')
    process_col = partial(process_column, col_name='labels')

    for split, data in embedding_dataset.items():
        data = data
        sequences, vectors = data['seqs'], data['vectors']
        vectors = [np.array(vec) for vec in vectors]
        sequence_dict = dict(zip(sequences, vectors))
        for downstream_dataset in DEFAULT_DATA_PATHS:
            down = load_dataset(downstream_dataset, split='test').shuffle(seed=42)
            down = down.map(process_col)
            seqs, labels = down['seqs'], down['labels']
            label_dict = dict(zip(seqs, labels))
            embeds = [sequence_dict[seq[:2048]] for seq in seqs]
            embed_dict = dict(zip(seqs, embeds))
            main(embed_dict, label_dict, split=split, data_type=downstream_dataset.split('/')[-1])