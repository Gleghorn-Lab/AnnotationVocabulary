import torch
import argparse
from torchinfo import summary
from datasets import load_dataset
from transformers import EsmTokenizer
from udev_public.models.esm.custom_esm import CustomEsmForMaskedLM
from udev_public.data.data_collators import AspectEvaluationCollator
from udev_public.data.nlp_dataset_classes import SequenceAnnotationDataset
from udev_public.metrics.language_modeling import compute_metrics_mlm_from_pred
from udev_public.data.data_utils import get_max_from_list_of_lists, process_column
from udev_public.trainers.huggingface import hf_trainer_from_args


def get_args():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--token', type=str)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--esm_path', type=str, default='facebook/esm2_t6_8m_UR50D')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=512)
    return parser.parse_args()


def main():
    args = get_args()
    paths = [
        'GleghornLab/EXP_test_ec_only',
        'GleghornLab/EXP_test_cc_only',
        'GleghornLab/EXP_test_mf_only',
        'GleghornLab/EXP_test_bp_only',
        'GleghornLab/EXP_test_3d_only',
        'GleghornLab/EXP_test_ip_only',
    ]
    tokenizer = EsmTokenizer.from_pretrained(args.esm_path)
    model = CustomEsmForMaskedLM.from_pretrained(args.model_path)
    print(summary(model))
    aspect_ranges = {
        'EC': (1, 5320),
        'MF': (5322, 14069),
        'BP': (14071, 15616),
        'CC': (15618, 21674),
        '3D': (34418, 38949),
        'IP': (21676, 34416)
    }
    for both in [False, True]:
        print(f"\nRunning evaluation with both={both}")
        for path in paths:
            aspect = path.split('_')[-2].upper()
            print(f"\nEvaluating {aspect} aspect")
            
            dataset = load_dataset(path, token=args.token)['train']
            process = process_column(col_name='combined')
            dataset = dataset.map(process)

            max_ann_value = get_max_from_list_of_lists(dataset['combined'])
            plm_vocab_size = tokenizer.vocab_size

            if both:
                vocab_size = plm_vocab_size + max_ann_value
                print(f'Total vocab size: {vocab_size} (PLM: {plm_vocab_size}, Annotations: {max_ann_value})')
            else:
                vocab_size = max_ann_value + 3  # +3 for special tokens
                print(f'Annotation vocab size: {vocab_size}')

            eval_dataset = SequenceAnnotationDataset(dataset, 'seqs', 'combined', max_length=args.max_length)
            data_collator = AspectEvaluationCollator(
                aspect_ranges=aspect_ranges,
                aspect=aspect,
                tokenizer=tokenizer,
                both=both
            )
            trainer = hf_trainer_from_args(args,
                                           model,
                                           eval_dataset=eval_dataset,
                                           compute_metrics=compute_metrics_mlm_from_pred,
                                           data_collator=data_collator)
            metrics = trainer.evaluate()
            print(f"{aspect} Evaluation Metrics (both={both}):")
            print(metrics)
            trainer.accelerator.free_memory()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

