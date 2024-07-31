import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import biotite.sequence as seq
import biotite.sequence.align as align
import random
import math
from datasets import load_dataset
from tqdm.auto import tqdm
import multiprocessing
from functools import partial


def generate_random_amino_acid_sequence(length):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    return ''.join(random.choice(amino_acids) for _ in range(length))


class AminoSequenceAligner:
    def __init__(self):
        self.matrix = align.SubstitutionMatrix.std_protein_matrix()
        self.valid_characters = set(self.matrix.get_alphabet1())

    def align(self, seqA, seqB, metrics=None):
        if metrics is None:
            metrics = ['ratio', 'error', 'norm', 'error_ratio']
        
        seqA = seq.ProteinSequence(''.join(c if c in self.valid_characters else 'X' for c in seqA))
        seqB = seq.ProteinSequence(''.join(c if c in self.valid_characters else 'X' for c in seqB))
        max_len = max(len(seqA), len(seqB))
        self_score_a = align.align_optimal(seqA, seqA, self.matrix)[0].score
        self_score_b = align.align_optimal(seqB, seqB, self.matrix)[0].score
        score = align.align_optimal(seqA, seqB, self.matrix)[0].score
        
        results = {}
        if 'ratio' in metrics:
            results['ratio'] = score / self_score_a
        if 'error' in metrics:
            results['error'] = (self_score_a - score) / max_len
        if 'norm' in metrics:
            results['norm'] = score / (math.sqrt(self_score_a * self_score_b))
        if 'error_ratio' in metrics:
            results['error_ratio'] = max_len / (self_score_a - score + max_len)
        
        return results


def mutate_sequence(sequence, mutation_rate):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    mutated_sequence = list(sequence)
    for i in range(len(mutated_sequence)):
        if random.random() < mutation_rate:
            mutated_sequence[i] = random.choice(amino_acids)
    return ''.join(mutated_sequence)


def process_batch(batch, mode, metrics, all_seqs, mutation_rate=None):
    aligner = AminoSequenceAligner()
    results = {metric: [] for metric in metrics}
    for seq in batch:
        if mode == 'diff':
            other_seq = random.choice(all_seqs)
        elif mode == 'random':
            other_seq = generate_random_amino_acid_sequence(len(seq))
        elif mode == 'self':
            if mutation_rate is not None:
                other_seq = mutate_sequence(seq, mutation_rate)
            else:
                other_seq = seq
        else:
            raise ValueError("Invalid mode. Choose 'diff', 'random', or 'self'.")
        
        batch_results = aligner.align(seq, other_seq, metrics)
        for metric in metrics:
            results[metric].append(batch_results[metric])
    return results


def get_scores(seqs, mode='diff', metrics=None, mutation_rate=None):
    if metrics is None:
        metrics = ['ratio', 'error', 'norm', 'error_ratio']
    
    num_cpus = max(1, multiprocessing.cpu_count() - 4)
    batch_size = max(1, len(seqs) // (num_cpus * 10))  # Smaller batches for more frequent updates
    batches = [seqs[i:i+batch_size] for i in range(0, len(seqs), batch_size)]
    
    with multiprocessing.Pool(num_cpus) as pool:
        results = list(tqdm(
            pool.imap(partial(process_batch, mode=mode, metrics=metrics, all_seqs=seqs, mutation_rate=mutation_rate), batches),
            total=len(batches),
            desc=f"Processing {mode} alignments" + (f" (mutation rate: {mutation_rate})" if mutation_rate else "")
        ))
    
    scores = {metric: [] for metric in metrics}
    for batch_result in results:
        for metric in metrics:
            scores[metric].extend(batch_result[metric])
    
    return scores


# Update the plot_histogram function to adjust subplot titles
def plot_histogram(scores, title, score_type, ax):
    mean = np.mean(scores)
    std_dev = np.std(scores)
    
    sns.histplot(scores, bins=100, kde=True, color='lightblue', edgecolor='black', alpha=0.7, ax=ax)
    ax.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    for i in range(1, 4):
        ax.axvline(mean - i * std_dev, color='green', linestyle='dashed', linewidth=1)
        ax.axvline(mean + i * std_dev, color='green', linestyle='dashed', linewidth=1, label=f'{i} Std Dev' if i == 1 else '')
    ax.set_xlabel(f'Alignment Score', fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    ax.set_title(title, fontsize=18, pad=20, fontweight='bold')  # Make subplot titles bold
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Improve x-axis formatting
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust y-axis to start from 0
    ax.set_ylim(bottom=0)


if __name__ == '__main__':
    data = load_dataset('lhallee/new_annotation_vocab', split='test')
    seqs = data['seqs']

    metrics_to_calculate = ['error_ratio']

    # Set the style for the plots
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

    fig, axes = plt.subplots(2, 3, figsize=(30, 20))

    modes = [
        ('diff', 'Random Paired Alignment'),
        ('random', 'Random Sequence Alignment'),
        ('self', 'Self Alignment'),
        ('self', 'Self Alignment (25% mutation)'),
        ('self', 'Self Alignment (50% mutation)'),
        ('self', 'Self Alignment (75% mutation)')
    ]

    mutation_rates = [None, None, None, 0.25, 0.5, 0.75]

    for i, ((mode, title), mutation_rate) in enumerate(zip(modes, mutation_rates)):
        row = i // 3
        col = i % 3
        scores = get_scores(seqs, mode, metrics_to_calculate, mutation_rate)
        for metric in metrics_to_calculate:
            plot_histogram(scores[metric], title, metric, axes[row, col])

    # Adjust the layout and spacing
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Remove any excess white space
    fig.set_tight_layout(True)

    # Save the figure with high resolution
    plt.savefig('combined_alignment_scores.png', dpi=300, bbox_inches='tight')
    plt.close()

