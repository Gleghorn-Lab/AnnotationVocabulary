import multiprocessing as mp
from functools import partial
from datasets import load_dataset
from Bio import Align
from Bio.Align import substitution_matrices
from tqdm.auto import tqdm
import numpy as np

def sanitize_sequence(sequence):
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    return ''.join(aa if aa in standard_aa else 'X' for aa in sequence.upper())

def calculate_percent_identity(seq1, seq2, aligner):
    seq1 = sanitize_sequence(seq1)
    seq2 = sanitize_sequence(seq2)
    alignment = aligner.align(seq1, seq2)[0]
    identical_count = sum(a == b for a, b in zip(alignment.target, alignment.query) if a != '-' and b != '-')
    total_length = alignment.shape[1]
    return (identical_count / total_length) * 100

def process_sequence(test_seq, ref_sequences, threshold=90):
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    similarities = []
    for idx, ref_seq in tqdm(enumerate(ref_sequences), total=len(ref_sequences)):
        identity = calculate_percent_identity(test_seq, ref_seq, aligner)
        similarities.append((idx, identity))
        if identity >= threshold:
            # Early exit if we find a match above the threshold
            return None
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:100]


def process_sequence_chunk(test_seq, ref_sequences):
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    similarities = []
    for idx, ref_seq in tqdm(enumerate(ref_sequences), total=len(ref_sequences)):
        identity = calculate_percent_identity(test_seq, ref_seq, aligner)
        similarities.append((idx, identity))
    return similarities


def parallel_process_sequence(seq, refs):
    num_processes = max(1, mp.cpu_count() - 4)
    chunk_size = len(refs) // (num_processes * 10)
    refs_chunks = [refs[i:i + chunk_size] for i in range(0, len(refs), chunk_size)]

    # Create a partial function with fixed seq and threshold
    process_chunk = partial(process_sequence_chunk, seq)
    # Create a multiprocessing pool
    with mp.Pool(processes=num_processes) as pool:
        # Process chunks in parallel with progress bar
        results = list(tqdm(
            pool.imap(process_chunk, refs_chunks),
            total=len(refs_chunks),
            desc="Processing reference chunks",
            unit="chunk"
        ))
    # Combine results from all chunks
    combined_results = [item for sublist in results for item in sublist]
    combined_results.sort(key=lambda x: x[1], reverse=True)
    return combined_results


def process_and_write(args):
    test_seq_idx, test_seq, ref_sequences, threshold, output_file = args
    similarities = process_sequence(test_seq, ref_sequences, threshold)
    if similarities is not None:
        with open(output_file, 'a') as f:
            f.write(f"Test Sequence Index: {test_seq_idx}\n")
            f.write(f"Sequence:\n{test_seq}\n")
            f.write("Top similarities:\n")
            for rank, (ref_idx, identity) in enumerate(similarities, 1):
                f.write(f"  {rank}. Reference Sequence {ref_idx + 1}: {identity:.2f}% identity\n")
            avg_identity = np.mean([identity for _, identity in similarities])
            f.write(f"Average identity of top matches: {avg_identity:.2f}%\n\n")
        return 1
    return 0

def find_and_write_unique_sequences(test_sequences, ref_sequences, output_file, threshold=90, num_processes=None):
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 4)
    print('Number of processes: ', num_processes)

    with open(output_file, 'w') as f:
        f.write(f"Unique sequences from the test set (no matches >= {threshold}% identity):\n\n")

    with mp.Pool(processes=num_processes) as pool:
        args_list = [(idx, test_seq, ref_sequences, threshold, output_file) for idx, test_seq in enumerate(test_sequences)]
        unique_count = sum(tqdm(
            pool.imap(process_and_write, args_list),
            total=len(test_sequences),
            desc="Processing sequences",
            unit="seq"
        ))
    
    return unique_count

if __name__ == "__main__":
    output_file = "unique_test_sequences.txt"
    threshold = 95  # Set the threshold for uniqueness
    
    print("Loading dataset...")
    data = load_dataset('GleghornLab/AnnotationVocab')
    
    print("Preparing reference sequences...")
    ref_sequences = list(set(data['train']['seqs'] + data['exp']['seqs'] + data['long_exp']['seqs']))
    print(f"Number of unique reference sequences: {len(ref_sequences)}")
    
    print("Preparing test sequences...")
    test_sequences = list(set(data['test']['seqs']))
    print(f"Number of unique test sequences: {len(test_sequences)}")
    
    print("Finding and writing unique sequences...")
    unique_count = find_and_write_unique_sequences(test_sequences, ref_sequences, output_file, threshold)
    
    print(f"Found {unique_count} unique sequences in the test set.")
    print(f"Results have been written to {output_file}")