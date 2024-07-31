import torch
import argparse
import csv
import random
import time
import numpy as np
from functools import partial
from datasets import load_dataset
from transformers import EsmTokenizer
from tqdm.auto import tqdm
from udev_public.models.esm.custom_esm import CustomEsmForMaskedLM
from udev_public.data.data_collators import SequenceAnnotationCollator
from udev_public.data.nlp_dataset_classes import SequenceAnnotationDataset
from udev_public.metrics.language_modeling import compute_metrics_mlm_from_pred
from udev_public.data.data_utils import get_max_from_list_of_lists, process_column
from udev_public.trainers.huggingface import hf_trainer_from_args

"""
ESM2_MODELS = [
    "lhallee/esm_ann_dual_exp_18000",
    "facebook/esm2_t12_35M_UR50D",
    "facebook/esm2_t30_150M_UR50D",
    ]
"""
ESM2_MODELS = [
    "lhallee/esm_ann_dual_RED_15000",
    "lhallee/esm_ann_dual_RED_all_42000",
    "lhallee/esm_ann_dual_RED_all_20000",
    "lhallee/esm_ann_dual_RED_all_9000",
    "facebook/esm2_t12_35M_UR50D",
    "facebook/esm2_t30_150M_UR50D",
    ]


DEFAULT_MASK_PERCENTAGES = [0.05, 0.15, 0.30, 0.50, 0.70]
N_TRIALS = 5


def get_args():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--token', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='GleghornLab/RED_annotations')
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--columns', type=str, nargs='+')
    parser.add_argument('--model_paths', type=str, nargs='+', help='Paths to model weights')
    parser.add_argument('--mask_percentages', type=float, nargs='+', help='Mask percentages to evaluate')
    parser.add_argument('--output_prefix', type=str, default='mlm_evaluation', help='Prefix for output CSV files')
    return parser.parse_args()


def set_random_seed(seed=None):
    """
    Set the random seed for PyTorch and NumPy.
    If no seed is provided, generate a random seed.
    
    Args:
    seed (int, optional): The seed to set. If None, a random seed will be generated.
    verbose (bool): Whether to print debug information
    
    Returns:
    int: The seed that was set
    """
    if seed is None:
        # Use current time for seed generation
        seed = int(time.time() * 1000) % 100
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(seed)


def evaluate_model(args, model, dataset, data_collator):
    set_random_seed()
    trainer = hf_trainer_from_args(args,
                                   model,
                                   train_dataset=dataset,
                                   valid_dataset=dataset,
                                   compute_metrics=compute_metrics_mlm_from_pred,
                                   data_collator=data_collator)
    metrics = trainer.evaluate()
    trainer.accelerator.free_memory()
    torch.cuda.empty_cache()
    return metrics


def main():
    args = get_args()
    args.grad_accum, args.lr, args.num_epochs = 1, 1e-4, 1 # to get trainer successfully
    dataset = load_dataset(args.data_path, token=args.token)
    
    if args.columns is None:
        seq_col, ann_col = 'seqs', 'combined'
    else:
        seq_col, ann_col = args.columns
    
    try:
        process = partial(process_column, col_name=ann_col)
        dataset = dataset.map(process)
    except:
        pass

    try:
        test_set = dataset['test']
    except:
        dataset = dataset['train'].shuffle(seed=42).train_test_split(test_size=1000, seed=42)
        test_set = dataset['test']

    vocab_size = max(get_max_from_list_of_lists(dataset['train'][ann_col]), get_max_from_list_of_lists(test_set[ann_col]))
    print('Vocab size ', vocab_size)
    test_dataset = SequenceAnnotationDataset(test_set, seq_col, ann_col, max_length=args.max_length)

    model_paths = args.model_paths if args.model_paths is not None else ESM2_MODELS
    mask_percentages = args.mask_percentages if args.mask_percentages is not None else DEFAULT_MASK_PERCENTAGES
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')

    # Initialize results dictionary
    results = {metric: {f"{model}_{perc}": [] for model in model_paths for perc in mask_percentages} 
               for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1']}

    for model_path in tqdm(model_paths, desc='Evaluating'):
        print(f"Evaluating model: {model_path}")
        args.save_path = model_path
        model = CustomEsmForMaskedLM.from_pretrained(model_path, token=args.token)
        
        if 'facebook' in model_path.lower():
            weights = {'both_weight':0.0, 'seq_weight':1.0, 'ann_weight':0.0}
        else:
            weights = {'both_weight':1.0, 'seq_weight':0.0, 'ann_weight':0.0}
        data_collator = SequenceAnnotationCollator(tokenizer, mask_sequence=True, mask_annotation=False, **weights)

        for mask_percentage in mask_percentages:
            print(f"  Evaluating with mask percentage: {mask_percentage}")
            data_collator.mlm_probability = mask_percentage

            for trial in range(N_TRIALS):
                print(f"    Trial {trial+1}/{N_TRIALS}")
                test_dataset.shuffle()
                metrics = evaluate_model(args, model, test_dataset, data_collator)
                
                column_name = f"{model_path}_{mask_percentage}"
                results['loss'][column_name].append(metrics['eval_loss'])
                results['accuracy'][column_name].append(metrics['eval_accuracy'])
                print()
                print(metrics['eval_accuracy'])
                results['precision'][column_name].append(metrics['eval_precision'])
                results['recall'][column_name].append(metrics['eval_recall'])
                results['f1'][column_name].append(metrics['eval_f1'])

    # Write results to separate CSV files
    for metric, data in results.items():
        filename = f"{args.output_prefix}_{metric}.csv"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['Trial'] + list(data.keys()))
            # Write data
            for i in range(N_TRIALS):
                writer.writerow([f"Trial {i+1}"] + [data[col][i] for col in data.keys()])
        print(f"Results for {metric} saved to {filename}")


if __name__ == '__main__':
    main()
