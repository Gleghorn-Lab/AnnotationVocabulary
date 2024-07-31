import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio import Align
from Bio.Align import substitution_matrices
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def sanitize_sequence(sequence):
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    return ''.join(aa if aa in standard_aa else 'X' for aa in sequence.upper())

def calculate_percent_identity(seq1, seq2):
    seq1 = sanitize_sequence(seq1)
    seq2 = sanitize_sequence(seq2)

    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    
    alignment = aligner.align(seq1, seq2)[0]
    identical_count = sum(a == b for a, b in zip(alignment.target, alignment.query) if a != '-' and b != '-')
    total_length = len(seq1)  # Use the length of the original sequence
    
    return (identical_count / total_length) * 100

def process_pair(args):
    i, j, seq1, seq2 = args
    return i, j, calculate_percent_identity(seq1, seq2)

def calculate_identity_matrix(sequences):
    n = len(sequences)
    identity_matrix = np.zeros((n, n))
    
    pairs = [(i, j, sequences[i], sequences[j]) 
             for i in range(n) 
             for j in range(i, n)]  # Only calculate upper triangle
    
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        results = list(tqdm(pool.imap(process_pair, pairs), total=len(pairs), desc="Calculating identities"))
    
    for i, j, identity in results:
        identity_matrix[i, j] = identity
        if i != j:
            identity_matrix[j, i] = identity  # Mirror the result
    
    return identity_matrix

def plot_heatmap(identity_matrix, output_file):
    plt.figure(figsize=(12, 10))
    sns.heatmap(identity_matrix, xticklabels=False, yticklabels=False, 
                cmap="YlOrRd", annot=False, cbar_kws={'label': 'Sequence Identity (%)'})
    plt.title(f"Sequence Identity Heatmap (n={len(identity_matrix)})")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def main(fasta_file, output_file):
    # Read sequences from FASTA file
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    sequences = [str(seq.seq) for seq in sequences]
    
    print(f"Processing {len(sequences)} sequences...")
    
    # Calculate identity matrix
    identity_matrix = calculate_identity_matrix(sequences)
    
    # Plot heatmap
    plot_heatmap(identity_matrix, output_file)
    
    print(f"Heatmap saved as {output_file}")

if __name__ == "__main__":
    fasta_file = "AnnVocabTesttop100.fasta"  # Replace with your FASTA file path
    output_file = "sequence_identity_heatmap.png"
    main(fasta_file, output_file)