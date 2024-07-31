import multiprocessing as mp
import os
import numpy as np
from functools import partial
from datasets import load_dataset
from Bio import Align
from Bio.Align import substitution_matrices
from tqdm.auto import tqdm


def sanitize_sequence(sequence):
    # Define the standard amino acid letters
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    # Replace any character that's not a standard amino acid with 'X'
    return ''.join(aa if aa in standard_aa else 'X' for aa in sequence.upper())


def calculate_percent_identity(seq1, seq2):
    # Sanitize both sequences
    seq1 = sanitize_sequence(seq1)
    seq2 = sanitize_sequence(seq2)

    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    
    alignment = aligner.align(seq1, seq2)[0]
    identical_count = sum(a == b for a, b in zip(alignment.target, alignment.query) if a != '-' and b != '-')
    total_length = alignment.shape[1]
    
    return (identical_count / total_length) * 100


def process_sequence(query_sequence, idx_seq):
    idx, seq = idx_seq
    percent_identity = calculate_percent_identity(query_sequence, seq)
    return (idx, percent_identity)


def find_top_similar_sequences(query_sequence, sequence_list, top_n=100, num_processes=None):
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 4)
    print('Number of processes: ', num_processes)

    with mp.Pool(processes=num_processes) as pool:
        process_func = partial(process_sequence, query_sequence)
        similarities = list(tqdm(
            pool.imap(process_func, enumerate(sequence_list)),
            total=len(sequence_list),
            desc="Processing sequences"
        ))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


def write_results_to_file(results, split_name, output_file):
    with open(output_file, 'a') as f:
        f.write(f"\n--- Results for {split_name} ---\n")
        for idx, (seq_idx, percent_identity) in enumerate(results, 1):
            f.write(f"{idx}. Sequence {seq_idx + 1}: {percent_identity:.2f}% identity\n")
        
        # Calculate and write average sequence identity
        avg_identity = np.mean([identity for _, identity in results])
        f.write(f"\nAverage sequence identity of top 100: {avg_identity:.2f}%\n")


if __name__ == "__main__":
    query_sequence = "MSDLDRQIEQLKKCEPLKESEVKALCLKAMEILVEESNVQRVDAPVTICGDIHGQFYDMKELFKVGGDCPKTNYLFLGDFVDRGFYSVETFLLLLALKVRYPDRITLIRGNHESRQITQVYGFYDECLRKYGSVNVWRYCTDIFDYLSLSALIENKIFSVHGGLSPAISTLDQIRTIDRKQEVPHDGAMCDLLWSDPEDIVDGWGLSPRGAGFLFGGSVVSSFNHANNIDYICRAHQLVMEGYKWMFNNQIVTVWSAPNYCYRCGNVAAILELDENLNKQFRVFDAAPQESRGTPAKKPAPDYFL"
    output_file = "sequence_similarity_results.txt"
    
    # Clear the output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    data = load_dataset('GleghornLab/AnnotationVocab')
    
    for split_name, split in data.items():
        print(f"Processing {split_name} split...")
        sequences = list(set(split['seqs']))  # Remove duplicates
        print(f"Number of unique sequences in {split_name}: {len(sequences)}")

        top_similar = find_top_similar_sequences(query_sequence, sequences)
        
        write_results_to_file(top_similar, split_name, output_file)
        
        print(f"Results for {split_name} written to {output_file}")

    print(f"All results have been written to {output_file}")